"""Offline CHGNet relaxation with explicit outcomes and geometry-bound caching."""

from __future__ import annotations
from functools import partial
import importlib.metadata
import os
from pathlib import Path
import time
import numpy as np
from crystal_dlm.terminal_energy_consistency import (
    COMMON_RELAXATION_PROTOCOL,
    LABEL_GEOMETRY_PROTOCOL,
    TERMINAL_VERIFICATION_PROTOCOL,
)
from crystal_dlm.isolated_workers import isolated_results
from .io import file_hash, fingerprint, read_json, write_json

RELAXATION_PROTOCOL = {
    **COMMON_RELAXATION_PROTOCOL,
    "max_steps": 1000,
    "optimizer_stop": "joint_atomic_force_and_stress_v1",
}


def physical_identity(checkpoint):
    return {
        "checkpoint_sha256": file_hash(checkpoint),
        "implementation_sha256": {
            name: file_hash(Path(__file__).with_name(name)) for name in ("physics.py", "physics_core.py")
        },
        "protocol": RELAXATION_PROTOCOL,
        "geometry": LABEL_GEOMETRY_PROTOCOL,
        "verification": TERMINAL_VERIFICATION_PROTOCOL,
        "packages": {
            name: importlib.metadata.version(name) for name in ("chgnet", "ase", "torch", "pymatgen")
        },
    }


def record_key(record):
    if record.get("success"):
        return fingerprint(
            record.get("structure") if record.get("structure") is not None else record.get("body")
        )
    return fingerprint({"source_id": record["source_id"], "generation_failure": record.get("reason")})


class Labeler:
    def __init__(self, checkpoint, device="cuda:0"):
        import torch
        from ase.optimize import FIRE
        from ase.filters import FrechetCellFilter
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import StructOptimizer
        from . import physics_core

        self.core = physics_core
        os.environ["RSI_JOINT_PHYSICAL_STOP"] = "1"
        os.environ["RSI_STRESS_TOLERANCE"] = str(RELAXATION_PROTOCOL["stress_tolerance_GPa"])
        torch.set_num_threads(1)
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = CHGNet.from_dict(saved["model"]).to(device)
        self.model.eval()

        class PinnedOptimizer(StructOptimizer):
            def relax(self, structure, **kwargs):
                kwargs.pop("ase_filter", None)
                return super().relax(
                    structure,
                    ase_filter=partial(
                        FrechetCellFilter,
                        mask=np.ones((3, 3)),
                        hydrostatic_strain=False,
                        constant_volume=False,
                        scalar_pressure=0.0,
                    ),
                    loginterval=1,
                    assign_magmoms=False,
                    dt=0.1,
                    maxstep=0.2,
                    **kwargs,
                )

        self.optimizer = PinnedOptimizer(
            model=self.model,
            optimizer_class=physics_core.recorded_fire_class(FIRE),
            use_device=str(device),
            stress_weight=1 / physics_core.EV_A3_TO_GPA,
        )

    def label(self, record):
        result = self.core.label_record(
            record,
            model=self.model,
            optimizer=self.optimizer,
            fmax=RELAXATION_PROTOCOL["fmax"],
            stress_tolerance=RELAXATION_PROTOCOL["stress_tolerance_GPa"],
            max_steps=RELAXATION_PROTOCOL["max_steps"],
            optimizer_status=self.core._OPT_STATUS,
        )
        if result["status"] == "evaluation_error":
            result["status"] = "worker_error"
        return result


def _worker(connection, checkpoint, device, cache):
    labeler = Labeler(checkpoint, device)
    connection.send({"ready": True})
    while True:
        record = connection.recv()
        if record is None:
            return
        result = labeler.label(record)
        if "relaxation_trajectory" in result:
            import gzip
            import json

            key = record_key(record)
            path = Path(cache) / key[:2] / (key + ".trajectory.json.gz")
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(result.pop("relaxation_trajectory"), stream, allow_nan=False)
            result["trajectory_artifact"] = {"path": str(path), "sha256": file_hash(path)}
        connection.send({"result": result})
        if result["status"] == "worker_error":
            return


def label_records(
    records,
    checkpoint,
    output,
    *,
    devices=("cuda:0",),
    workers_per_device=4,
    cache=None,
    record_timeout=600.0,
):
    """Evaluation errors remain unresolved; they never become unstable labels."""
    output = Path(output)
    cache = Path(cache) if cache is not None else output / "cache"
    identity = physical_identity(checkpoint)
    identity_key = fingerprint(identity)
    cache = cache / identity_key
    cache.mkdir(parents=True, exist_ok=True)
    tasks, values, unique = [], {}, {}
    for record in records:
        key = record_key(record)
        unique.setdefault(key, record)
    for key, record in unique.items():
        path = cache / key[:2] / (key + ".json")
        if path.exists():
            values[key] = read_json(path)["label"]
        elif not record.get("success"):
            values[key] = {
                "status": "generation_failure",
                "verified": False,
                "raw_energy": None,
                "terminal_energy": None,
                "gap": None,
                "raw": None,
                "terminal": None,
                "actual_steps": None,
                "final_structure": None,
            }
        else:
            tasks.append((key, record))
    arguments = [
        (str(checkpoint), device, str(cache)) for device in devices for _ in range(workers_per_device)
    ]
    started = time.monotonic()
    for key, record, result in isolated_results(
        tasks,
        worker_target=_worker,
        worker_arguments=arguments,
        task_timeout=record_timeout,
        startup_timeout=180.0,
    ):
        values[key] = result
        if result["status"] != "worker_error":
            write_json(
                cache / key[:2] / (key + ".json"),
                {"label": result, "identity": identity_key, "record_key": key},
            )
        if len(values) % 20 == 0:
            progress = {
                "unique_completed": len(values),
                "unique_requested": len(unique),
                "seconds": time.monotonic() - started,
            }
            write_json(output / "PROGRESS.json", progress)
            print(progress, flush=True)
    result = [
        dict(
            values[record_key(record)],
            source_id=record["source_id"],
            ordinal=record["ordinal"],
            record_key=record_key(record),
        )
        for record in records
    ]
    from .io import write_rows

    write_rows(output / "labels.jsonl", result)
    write_json(output / "protocol.json", identity)
    return result
