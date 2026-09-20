"""Adapters to the verified baseline. Expensive imports occur only during real runs."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import gc
import math
import numpy as np
import torch
from .common import file_sha, read_json, read_rows, write_json, write_rows, digest, seed_for
from .conditions import composition_key
from .quality import Measurement, raw_utility
from .verifier import JointController


def _shard_call(name, args, kwargs):
    if str(kwargs.get('device','')).startswith('cuda'):
        torch.cuda.set_device(kwargs['device'])
    if name == "refine_drafts":
        from .evaluation import refine_drafts
        return refine_drafts(*args,**kwargs)
    return globals()[name](*args, **kwargs)


def _parallel_sources(name, config, assets, plans, folder, settings, budget, *,
                      drafts=None, guide=None, collect=False, forced=None):
    """Independent source workers; no DDP or changes to request seeds."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    runtime = settings.get("runtime", {})
    devices = runtime.get("devices", [])
    copies = runtime.get("generation_workers_per_device", 1) if drafts is None else 1
    workers = [d for d in devices for _ in range(copies)][:len(plans)]
    child_settings = deepcopy(settings)
    child_settings["runtime"]["devices"] = []
    child_settings["runtime"]["generation_workers_per_device"] = 1
    indexed = []
    with ProcessPoolExecutor(len(workers), mp_context=mp.get_context("spawn")) as pool:
        jobs = []
        for rank, device in enumerate(workers):
            indexes = list(range(rank, len(plans), len(workers)))
            subset = [plans[i] for i in indexes]
            args = (config, assets, subset)
            if drafts is not None: args += ([drafts[i] for i in indexes],)
            args += (Path(folder)/f"shard{rank}", child_settings, budget)
            kwargs = {"device":device}
            if drafts is None: kwargs.update(guide=guide, collect=collect, forced=forced)
            jobs.append((indexes, pool.submit(_shard_call, name, args, kwargs)))
        for indexes, job in jobs:
            indexed.extend(zip(indexes, job.result(), strict=True))
    rows = [row for _,row in sorted(indexed)]
    for i,row in enumerate(rows): write_json(Path(folder)/f"{i:06d}.json", row)
    budget.state = read_json(budget.path)
    return rows


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_drafts(config, assets, plans, folder, settings, budget, *, guide=None, collect=False,
                  forced=None, device="cuda:0"):
    from .inference_protocol import bind_inference_protocol
    config=bind_inference_protocol(config,assets)
    from dlm_iclr.runtime.models import load_model_and_tokenizer
    from dlm_iclr.runtime.config import backend_config
    from dlm_iclr.runtime.io import json_default
    from dlm_iclr.c1.generation import Constructor
    from dlm_iclr.c1.trainer import load_head
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    # Stored input identity prevents reading a previous model's outputs as fresh samples.
    identity = {"assets":assets, "plan_sources":[p["source_id"] for p in plans],
                "guide":digest(guide.saved) if guide else None, "forced":forced, "collect":collect}
    if 'inference_protocol' in assets:
        identity['ordered_plans']=digest(plans)
    record_states = settings.get("runtime", {}).get("record_construction_states", False)
    if record_states:
        identity["construction_trace_schema"] = "actual_all_geometry_commits_v1"
    manifest = folder / "inputs.json"
    if manifest.exists() and read_json(manifest) != identity:
        raise ValueError("Draft directory is bound to different inputs")
    if not manifest.exists() and any(folder.glob('[0-9]*.json')):
        raise ValueError('Cannot bind an unversioned draft cache to a new model/protocol')
    write_json(manifest,identity)
    cached = [folder / f"{i:06d}.json" for i in range(len(plans))]
    if all(p.exists() for p in cached):
        return [read_json(p) for p in cached]
    runtime = settings.get("runtime", {})
    if runtime.get("devices") and (len(runtime["devices"]) * runtime.get("generation_workers_per_device",1) > 1):
        return _parallel_sources("sample_drafts",config,assets,plans,folder,settings,budget,
                                 guide=guide,collect=collect,forced=forced)
    model, tokenizer = load_model_and_tokenizer(assets["base"],assets["draft"],torch.device(device),mean_resizing=False)
    head = load_head(assets["head"],device)
    constructor = Constructor(model,tokenizer,backend_config(config,stage="c1").inference,axis_head=head,
                              record_construction_states=record_states,
                              axis_confidence_policy=assets.get('axis_confidence_policy','legacy_unary'))
    if not hasattr(constructor,"joint_controller"):
        raise RuntimeError("Install the two exact-hash baseline hook patches first")
    result = []
    for i,plan in enumerate(plans):
        if cached[i].exists():
            result.append(read_json(cached[i])); continue
        budget.reserve("drafts")
        force = forced.get(plan["source_id"]) if forced else None
        # A frozen verifier carries its calibrated opportunity/feature policy.
        # Evaluation defaults must not silently replace this deployment policy.
        controller_settings=guide.saved['settings'] if guide is not None else settings['verifier']
        control = JointController(controller_settings,guide=guide,forced=force,collect=collect)
        constructor.joint_controller = control if (guide or collect or force) else None
        generated = constructor.generate(plan)
        if force and not (control.forced_hit and control.force_verified):
            raise RuntimeError("Exact counterfactual replay did not reach/commit the registered state-action")
        generated["process_probes"] = control.probes
        generated["process_decisions"] = control.decisions
        generated["forced_replay_verified"] = bool(force and control.force_verified)
        import json
        generated = json.loads(json.dumps(generated, default=json_default, allow_nan=False))
        write_json(cached[i],generated); result.append(generated)
        if (i+1)%8 == 0:
            print({"stage":"drafts", "completed":i+1,"requests":len(plans)},flush=True)
    del constructor, model, head
    release()
    return result


def _numeric_graph(graph):
    out = dict(graph)
    for key in ("edge_indices", "to_jimages", "a_type"):
        if key in out:
            out[key] = np.asarray(out[key])
    return out


def make_tracking_refiner(config, tokenizer, settings, device):
    from dlm_iclr.runtime.config import asset
    from dlm_iclr.diffusion.refinement import Refiner, lattices_to_parameters
    from pymatgen.core import Structure, Lattice

    class TrackingRefiner(Refiner):
        """Record existing sampler outputs without changing its RNG or endpoint."""
        def sample(self, graph, seed):
            self.selected_states = []
            original = self.model.sample
            def tracked(*args, **kwargs):
                result, trajectory = original(*args, **kwargs)
                for t in settings["teacher_times"]:
                    if not 0 < t < self.steps:
                        continue
                    index = self.steps-int(t)
                    lattice = trajectory["all_lattices"][index]
                    lengths, angles = lattices_to_parameters(lattice)
                    coords = trajectory["all_frac_coords"][index]
                    if not all(bool(torch.isfinite(v).all()) for v in (lengths, angles, coords)):
                        continue
                    try:
                        cell = Lattice.from_parameters(*lengths.detach().cpu().reshape(3).tolist(),
                                                       *angles.detach().cpu().reshape(3).tolist())
                        structure = Structure(cell, result["atom_types"].detach().cpu().reshape(-1).tolist(),
                                              coords.detach().cpu().tolist())
                        self.selected_states.append({"origin":f"F_state_t{t}_from_{self.steps}",
                                                      "structure":structure.as_dict()})
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                        # Recording an unusable intermediate must never invalidate a valid endpoint.
                        continue
                return result, trajectory
            self.model.sample = tracked
            try:
                return super().sample(graph,seed)
            finally:
                self.model.sample = original

        def sample_many(self, graphs, seeds):
            self.batch_states = [[] for _ in graphs]
            original = self.model.sample
            def tracked(*args, **kwargs):
                kwargs["save_trajectory"] = True
                result, trajectory = original(*args, **kwargs)
                sizes = [int(np.asarray(g["n_atom"]).reshape(-1)[0]) for g in graphs]
                species = result["atom_types"].detach().cpu().split(sizes)
                for t in settings["teacher_times"]:
                    if not 0 < t < self.steps: continue
                    index = self.steps-int(t)
                    lengths, angles = lattices_to_parameters(trajectory["all_lattices"][index])
                    coords = trajectory["all_frac_coords"][index].detach().cpu().split(sizes)
                    for i in range(len(graphs)):
                        try:
                            cell = Lattice.from_parameters(*lengths[i].detach().cpu().tolist(),
                                                           *angles[i].detach().cpu().tolist())
                            structure = Structure(cell,species[i].reshape(-1).tolist(),coords[i].tolist())
                            if np.isfinite(structure.lattice.matrix).all() and np.isfinite(structure.frac_coords).all():
                                self.batch_states[i].append({"origin":f"F_state_t{t}_from_{self.steps}",
                                                             "structure":structure.as_dict()})
                        except (ValueError,FloatingPointError,np.linalg.LinAlgError):
                            continue
                return result, trajectory
            self.model.sample = tracked
            try:
                return super().sample_many(graphs,seeds)
            finally:
                self.model.sample = original
    return TrackingRefiner(asset(config,"diffusion"),tokenizer,device,steps=settings["teacher_steps"],
                            reuse_fixed_geometry=config["diffusion"]["reuse_fixed_geometry"])


def quantized_record(plan, structure, tokenizer, origin, *, jitter_seed=None):
    from dlm_iclr._core.expert_edit_data import arrays_from_structure, quantize_arrays
    from dlm_iclr._core.r03_physics_transfer import build_repair_constraints, geometry_support_report
    from dlm_iclr.data.adapters import align_sites
    from dlm_iclr.c1.generation import make_record
    from dlm_iclr.evaluation.physics import record_key
    arrays, _ = align_sites(arrays_from_structure(structure),plan["plan_state"])
    if len(arrays["species"]) != plan["plan_state"]["N"]:
        raise ValueError("Teacher changes atom count")
    if jitter_seed is not None:
        rng = np.random.default_rng(jitter_seed)
        arrays = deepcopy(arrays)
        arrays["frac_coords"] = (np.asarray(arrays["frac_coords"])+rng.uniform(-0.005,0.005,(len(arrays["species"]),3)))%1
        arrays["lengths"] = (np.asarray(arrays["lengths"])+rng.uniform(-0.05,0.05,3)).tolist()
        arrays["angles"] = (np.asarray(arrays["angles"])+rng.uniform(-0.5,0.5,3)).tolist()
    ids, decoded, diagnostic = quantize_arrays(arrays, tokenizer.get_vocab())
    support = geometry_support_report(ids,constraints=build_repair_constraints(tokenizer))
    if not support["supported"]:
        raise ValueError("Quantized teacher does not meet the retained geometry support")
    from collections import Counter
    expected = Counter(dict(zip(plan["plan_state"]["elements"],plan["plan_state"]["counts"],strict=True)))
    if Counter(decoded["species"]) != expected:
        raise ValueError("Teacher changes fixed composition")
    inverse = {int(v):k for k,v in tokenizer.get_vocab().items()}
    record = make_record(plan,stage="draft_teacher",body="".join(inverse[i] for i in ids))
    record.update(body_token_ids=ids,body_prompt=plan["body_prompt"],origin=origin)
    record["exact_record_key"] = record_key(record)
    record["quantization"] = diagnostic
    return record


def collect_teacher_candidates(config, assets, plans, drafts, folder, settings, budget, *, device="cuda:0"):
    from transformers import AutoTokenizer
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=True)
    if all((folder/f"{i:06d}.json").exists() for i in range(len(plans))):
        return [read_json(folder/f"{i:06d}.json") for i in range(len(plans))]
    if len(settings.get("runtime",{}).get("devices",[])) > 1:
        return _parallel_sources("collect_teacher_candidates",config,assets,plans,folder,settings,budget,drafts=drafts)
    tokenizer = AutoTokenizer.from_pretrained(assets["draft"],trust_remote_code=True)
    refiner = make_tracking_refiner(config,tokenizer,settings,device)
    results = []
    variants = settings.get("quantized_variants",3)
    batch_size = settings.get("runtime",{}).get("teacher_batch_size",1)
    batched = {}
    pending = [i for i,d in enumerate(drafts) if d.get("graph") is not None and not (folder/f"{i:06d}.json").exists()]
    if batch_size > 1:
        for start in range(0,len(pending),batch_size):
            indexes = pending[start:start+batch_size]
            budget.reserve("teacher_calls",len(indexes))
            sampled = refiner.sample_many([_numeric_graph(drafts[i]["graph"]) for i in indexes],
                                          [plans[i]["refiner_noise_seed"] for i in indexes])
            for j,i in enumerate(indexes): batched[i] = (sampled[j],refiner.batch_states[j])
            print({"stage":"teacher_diffusion","completed":start+len(indexes),"requests":len(pending),"device":device},flush=True)
    for i,(plan,draft) in enumerate(zip(plans,drafts,strict=True)):
        path = folder / f"{i:06d}.json"
        if path.exists():
            results.append(read_json(path)); continue
        rows, rejected = [], []
        if draft.get("graph") is not None:
            source = dict(draft,graph=_numeric_graph(draft["graph"]))
            if i in batched:
                sampled, states = batched[i]
                refined = refiner.refine(plan,source,sampled=sampled)
            else:
                budget.reserve("teacher_calls")
                refined = refiner.refine(plan,source)
                states = getattr(refiner,"selected_states",[])
            candidates = []
            if refined["continuous_trace"].get("source") == "continuous_F":
                candidates.append({"origin":"F_endpoint", "structure":refined["record"]["structure"]})
            candidates.extend(states)
            keys = set()
            for state in candidates:
                for variant in range(variants):
                    try:
                        row = quantized_record(plan,state["structure"],tokenizer,
                              state["origin"]+f":quantized_variant{variant}",
                              jitter_seed=seed_for(settings["seed"],plan["source_id"],state["origin"],variant) if variant else None)
                        if row["exact_record_key"] not in keys:
                            rows.append(row); keys.add(row["exact_record_key"])
                    except (ValueError,KeyError,TypeError,IndexError) as error:
                        rejected.append({"origin":state["origin"],"variant":variant,"reason":str(error)})
        result = {"source_id":plan["source_id"],"candidates":rows,"rejected":rejected}
        write_json(path,result); results.append(result)
        if (i+1)%8 == 0:
            print({"stage":"teacher_candidates","completed":i+1},flush=True)
    del refiner, tokenizer
    release()
    return results


def _physical_config(config):
    return {"max_steps":config["evaluation"]["relaxation_steps"],
            "fmax":config["evaluation"]["fmax"],
            "stress_tolerance_GPa":config["evaluation"]["stress_tolerance_GPa"]}


def measure(config, records, folder, settings, budget, *, device="cuda:0", singlepoint=False):
    from dlm_iclr.runtime.config import run_root
    from dlm_iclr.evaluation.workflow import ensure_chgnet
    from dlm_iclr.evaluation.physics import label_records, record_key, physical_identity, Labeler
    from dlm_iclr.evaluation.relaxation import force_and_stress, finite_scalar, structure_from_record
    from dlm_iclr.evaluation.sun import HullReference
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=True)
    checkpoint = ensure_chgnet(config)
    protocol = _physical_config(config)
    protocol_key = digest(physical_identity(checkpoint,protocol))
    hull_path = run_root(config)/"hull"
    hull = HullReference(hull_path) if (hull_path/"official_slim_cache.jsonl").exists() else None
    if settings["require_hull"] and hull is None:
        raise FileNotFoundError("The existing MP-hull cache is required; no silent no-hull fallback")
    # The physical protocol is independent of unrelated chemical systems added
    # to the cache. Each label below binds the reference for its own chemistry.
    from dlm_iclr.evaluation.hull import chemical_system
    references = {}
    for r in records:
        elements = r.get("declared_composition")
        try:
            system = "-".join(sorted(elements)) if elements else chemical_system(r)
        except (ValueError,TypeError,KeyError):
            system = ""
        references[record_key(r)] = hull.rows.get(system) if hull else None
    identities = {"records":[record_key(r) for r in records], "physics":protocol_key,
                  "hull":digest(references), "singlepoint":singlepoint}
    cached = folder/"result.json"
    if cached.exists():
        old = read_json(cached)
        if old["identity"] != identities:
            raise ValueError("Physics cache identity changed")
        return old["rows"]
    previous_labels = None
    physical_folder = folder/'physical'
    if not singlepoint and (physical_folder/'labels.jsonl').exists() and (physical_folder/'protocol.json').exists():
        stored = read_rows(physical_folder/'labels.jsonl')
        if ([r['record_key'] for r in stored] == [record_key(r) for r in records]
            and digest(read_json(physical_folder/'protocol.json')) == protocol_key
            and all(r.get('status') not in ('worker_error','unknown','evaluation_error') for r in stored)):
            previous_labels = stored
    if previous_labels is None:
        budget.reserve("physics_records",len(records))
    if singlepoint:
        labeller = Labeler(checkpoint,device,protocol)
        labels = []
        for record in records:
            try:
                if not record.get("success"):
                    labels.append({"status":"generation_failure","verified":False}); continue
                structure = structure_from_record(record)
                raw = labeller.model.predict_structure(structure,task="efs")
                if isinstance(raw,list): raw=raw[0]
                labels.append({"status":"singlepoint","verified":False,
                               "raw_energy":finite_scalar(raw["e"]),
                               "raw":force_and_stress(raw["f"],raw["s"],stress_unit="GPa")})
            except (ValueError,KeyError,TypeError,FloatingPointError) as error:
                labels.append({"status":"worker_error","verified":False,"error":str(error)})
        del labeller
        release()
    else:
        runtime = settings.get("runtime",{})
        labels = previous_labels if previous_labels is not None else label_records(
            records,checkpoint,physical_folder,devices=runtime.get("devices") or [device],
            workers_per_device=runtime.get("physics_workers_per_device",1),
            cache=run_root(config)/"cache/physics",protocol=protocol)
    rows = []
    for record,label in zip(records,labels,strict=True):
        expected_key = record_key(record)
        if label.get("record_key",expected_key) != expected_key:
            raise ValueError("Physical label and exact token geometry do not match")
        comp = None; hull_energy = None; reference_identity = references[expected_key]
        try:
            structure = structure_from_record(record)
            counts = structure.composition.get_el_amt_dict()
            from functools import reduce
            from math import gcd
            integers = {e:int(n) for e,n in counts.items()}
            divisor = reduce(gcd,integers.values())
            comp = "|".join(f"{e}:{n//divisor}" for e,n in sorted(integers.items()))
            if hull is not None:
                system,hull_energy = hull.energy(structure)
                reference_identity = hull.rows.get(system)
        except (ValueError,KeyError,TypeError):
            counts = record.get("declared_composition") or {}
            if counts:
                from functools import reduce
                from math import gcd
                divisor=reduce(gcd,(int(n) for n in counts.values()))
                comp="|".join(f"{e}:{int(n)//divisor}" for e,n in sorted(counts.items()))
            else:
                comp=""
        raw = label.get("raw") or {}
        energy,terminal = label.get("raw_energy"),label.get("terminal_energy")
        bound_protocol = digest([protocol_key,reference_identity])
        m = Measurement(expected_key,bound_protocol,comp,label.get("status","unknown"),
                        bool(label.get("verified",False)),energy,raw.get("force_rms_eV_A"),
                        raw.get("force_max_eV_A"),raw.get("stress_max_GPa"),
                        energy-hull_energy if energy is not None and hull_energy is not None else None,
                        terminal-hull_energy if terminal is not None and hull_energy is not None else None,
                        label.get("actual_steps"))
        rows.append({"source_id":record["source_id"],"measurement":m.to_dict(),
                     "final_structure":label.get("final_structure"),"exact_token_remeasured":bool(record.get("body_token_ids"))})
    # Do not persist unresolved computations as stable false labels. Resume retries unknown results.
    if all(row["measurement"]["status"] not in ("worker_error","unknown","evaluation_error") for row in rows):
        write_json(cached,{"identity":identities,"rows":rows})
    else:
        write_json(folder/"unresolved.json",{"identity":identities,"rows":rows})
    return rows
