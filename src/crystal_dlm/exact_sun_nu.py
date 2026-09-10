"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
from collections import defaultdict
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import uuid
from crystal_dlm.isolated_workers import isolated_results


PAIR_SCHEMA = "directed_structure_match_pair_v1"


def canonical_json(value):
    # Python's explicit NaN/Infinity tokens keep nonfinite geometry distinct
    # from null/string values. Hashing an irrelevant bad row must not require
    # evaluating it or block unrelated SUN contributors.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def matcher_contract():
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    path = Path(inspect.getsourcefile(StructureMatcher))
    return {
        "settings": matcher.as_dict(),
        "pymatgen_version": importlib.metadata.version("pymatgen"),
        "implementation": StructureMatcher.__module__ + "." + StructureMatcher.__qualname__,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _match_worker(connection, contract):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    if matcher_contract() != contract:
        raise ValueError("pair worker uses a different frozen matcher implementation")
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    connection.send({"ready": True})
    while True:
        record = connection.recv()
        if record is None:
            return
        try:
            matched = bool(
                matcher.fit(Structure.from_dict(record["left"]), Structure.from_dict(record["right"]))
            )
            result = {"status": "complete", "matched": matched}
        except Exception as error:
            result = {"status": "worker_error", "error": f"{type(error).__name__}: {error}", "matched": None}
        connection.send({"result": result})
        if result["status"] == "worker_error":
            return


class PairCache:
    def __init__(self, directory, contract):
        self.directory, self.contract = Path(directory), contract
        self.directory.mkdir(parents=True, exist_ok=True)
        self.contract_sha = fingerprint(contract)

    def descriptor(self, left, right):
        identity = {
            "schema": PAIR_SCHEMA,
            "left_sha256": fingerprint(left),
            "right_sha256": fingerprint(right),
            "matcher_sha256": self.contract_sha,
        }
        return fingerprint(identity), identity

    def get(self, key, identity):
        path = self.directory / key[:2] / (key + ".json")
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        if (
            saved.get("identity") != identity
            or saved.get("key") != key
            or type(saved.get("matched")) is not bool
        ):
            raise ValueError("directed pair cache identity or value is invalid")
        return saved["matched"]

    def put(self, key, identity, matched):
        if type(matched) is not bool:
            raise ValueError("an unresolved comparison cannot enter the exact pair cache")
        previous = self.get(key, identity)
        if previous is not None:
            if previous != matched:
                raise ValueError("the same directed geometry/matcher pair gave conflicting decisions")
            return
        path = self.directory / key[:2] / (key + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(canonical_json({"identity": identity, "key": key, "matched": matched}) + "\n")
        try:
            try:
                # An exclusive hard link publishes a complete immutable file.
                os.link(temporary, path)
            except FileExistsError:
                if self.get(key, identity) != matched:
                    raise ValueError("concurrent directed pair caches disagree")
        finally:
            temporary.unlink()


def conjunction(*values):
    if any(value is False for value in values):
        return False
    return None if any(value is None for value in values) else True


def evaluate_sun_predicates(
    structures,
    train_structures,
    train_formula_index,
    required_indices,
    *,
    cache_dir,
    frozen_nu_sha256,
    training_identity,
    workers=4,
    pair_timeout=30.0,
    startup_timeout=120.0,
    pair_runner=None,
    contract=None,
):
    """Return exact representative predicates without transitive clustering.

    U(i) compares i to *every* earlier same-formula structure. Earlier structures
    remain eligible witnesses regardless of their stability, novelty, or their
    own representative status. Only i's need for N/U is short-circuited.
    """
    required = sorted(set(required_indices))
    if len(required) != len(required_indices) or any(
        type(i) is not int or not 0 <= i < len(structures) for i in required
    ):
        raise ValueError("SUN contributors must identify distinct actual input structures")
    if not 1 <= workers <= 64:
        raise ValueError("invalid pair worker count")
    runtime = contract or matcher_contract()
    cache_contract = {"matcher": runtime, "frozen_nu_sha256": frozen_nu_sha256}
    cache = PairCache(cache_dir, cache_contract)
    payloads = [structure.as_dict() for structure in structures]
    input_hashes = [fingerprint(payload) for payload in payloads]
    formulas = [structure.composition.reduced_formula for structure in structures]
    formula_indices = defaultdict(list)
    for index, formula in enumerate(formulas):
        formula_indices[formula].append(index)
    training_payloads = {}
    counters = defaultdict(int)
    problems = []

    def process_groups(groups):
        """Joint N/U scheduling; a known false partner cancels irrelevant work."""
        states, tasks, identities, consumers = {}, {}, {}, defaultdict(set)

        def partner(index):
            return ("U" if index[0] == "N" else "N", index[1])

        for index, candidates in groups.items():
            states[index] = {"pending": set(), "order": [], "unknown": False, "value": True}
            for left, right in candidates:
                key, identity = cache.descriptor(left, right)
                matched = cache.get(key, identity)
                if matched is not None:
                    counters["pair_cache_hits"] += 1
                    if matched:
                        states[index]["value"] = False
                        break
                    continue
                states[index]["pending"].add(key)
                states[index]["order"].append(key)
                consumers[key].add(index)
                tasks[key] = {"left": left, "right": right}
                identities[key] = identity

        def admitted(key, payload):
            return any(
                states[index]["value"] is not False
                and states[partner(index)]["value"] is not False
                and key in states[index]["pending"]
                for index in consumers[key]
            )

        # Interleave predicates and inputs, so one long N list cannot hide a
        # quick U counterexample (or conversely). Pair direction/order remains
        # part of the exact predicate, independent of execution scheduling.
        active_tasks, scheduled = [], set()
        for offset in range(max((len(state["order"]) for state in states.values()), default=0)):
            for state in states.values():
                if offset < len(state["order"]):
                    key = state["order"][offset]
                    if key not in scheduled and admitted(key, tasks[key]):
                        active_tasks.append((key, tasks[key]))
                        scheduled.add(key)
        if pair_runner is None:
            results = isolated_results(
                active_tasks,
                worker_target=_match_worker,
                worker_arguments=[(runtime,)] * workers,
                task_timeout=pair_timeout,
                startup_timeout=startup_timeout,
                admitted=admitted,
            )
        else:
            results = pair_runner(active_tasks, admitted)
        for key, _, result in results:
            if result["status"] == "cancelled_irrelevant":
                counters["cancelled_irrelevant_pairs"] += 1
                continue
            matched = result.get("matched") if result["status"] == "complete" else None
            if matched is not None and type(matched) is not bool:
                raise ValueError("matcher worker returned a nonboolean comparison")
            if matched is None:
                counters["unresolved_pair_attempts"] += 1
                problems.append(
                    {"pair_key": key, "error": result.get("error"), "affected_inputs": sorted(consumers[key])}
                )
            else:
                counters["computed_pairs"] += 1
                cache.put(key, identities[key], matched)
            for index in consumers[key]:
                state = states[index]
                state["pending"].discard(key)
                if matched is True:
                    state["value"] = False
                elif matched is None:
                    state["unknown"] = True
        values = {}
        for index, state in states.items():
            if state["value"] is False:
                values[index] = False
            elif states[partner(index)]["value"] is False and state["pending"]:
                values[index] = None
            elif state["pending"]:
                raise ValueError("a still-relevant directed comparison disappeared from evaluation")
            else:
                values[index] = None if state["unknown"] else True
        return values

    groups = {}
    for index in required:
        candidates = []
        for ti in train_formula_index.get(formulas[index], []):
            if ti not in training_payloads:
                training_payloads[ti] = train_structures[ti].as_dict()
            candidates.append((payloads[index], training_payloads[ti]))
        groups[("N", index)] = candidates
        groups[("U", index)] = [
            (payloads[index], payloads[earlier])
            for earlier in formula_indices[formulas[index]]
            if earlier < index
        ]
    values = process_groups(groups)
    novel, unique = [None] * len(structures), [None] * len(structures)
    for index in required:
        novel[index], unique[index] = values[("N", index)], values[("U", index)]
    unresolved = [index for index in required if conjunction(novel[index], unique[index]) is None]
    return {
        "novel": novel,
        "unique": unique,
        "required_indices": required,
        "unresolved_sun_indices": unresolved,
        "pair_attempt_problems": problems,
        "counters": dict(counters),
        "complete_sun_predicates": not unresolved,
        "standalone_NU_complete": all(x is not None for x in novel + unique),
        "matcher_contract": cache_contract,
        "training_identity": training_identity,
        "input_geometry_order_sha256": fingerprint(input_hashes),
        "uniqueness_semantics": "later_to_every_earlier_same_formula_in_original_order",
        "novelty_semantics": "generated_to_each_same_formula_training_structure",
        "skipped_values_are_false": False,
    }
