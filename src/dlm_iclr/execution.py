"""Process workers used equally by a shell, Slurm allocation, or Python caller."""

from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import time
from .config import load_config
from .io import read_rows, read_json, write_json


def setup_device(device=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    return torch.device(device)


def run_commands(commands, log_dir):
    """Run only the supplied children; failures stop their sibling processes."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    running, logs = [], []
    try:
        for name, arguments in commands:
            log = (log_dir / f"{name}.log").open("a", encoding="utf-8")
            logs.append(log)
            env = os.environ.copy()
            env.update(
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            running.append(
                (
                    name,
                    subprocess.Popen(
                        [sys.executable, "-m", "dlm_iclr", *map(str, arguments)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        creationflags=flags,
                    ),
                )
            )
        while running:
            finished = []
            for name, process in running:
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"{name} exited with {code}; see {log_dir / (name + '.log')}")
                    finished.append((name, process))
            running = [value for value in running if value not in finished]
            if running:
                time.sleep(0.25)
    finally:
        for _, process in running:
            if process.poll() is None:
                process.terminate()
        for _, process in running:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def inference_worker(stage, config_path, plans_path, output, *, rank=0, world=1, device="cuda:0"):
    device = setup_device(device)
    import torch

    config, plans, output = load_config(config_path), read_rows(plans_path), Path(output)
    indices = [i for i in range(rank, len(plans), world) if not (output / stage / f"{i:05d}.json").exists()]
    if not indices:
        return
    if stage == "G":
        from .generation import Constructor
        from .models import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(
            config.assets.base_model, config.assets.generator, device, mean_resizing=False
        )
        generator = Constructor(model, tokenizer, config.inference)
        for completed, i in enumerate(indices, 1):
            value = generator.generate(plans[i])
            graph = value.pop("graph")
            value["graph_available"] = graph is not None
            if graph is not None:
                path = output / "graphs" / f"{i:05d}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(graph, path)
            write_json(output / "G" / f"{i:05d}.json", value)
            print(
                {"stage": stage, "worker": rank, "completed": completed, "requests": len(indices)}, flush=True
            )
    elif stage == "F":
        from transformers import AutoTokenizer
        from .refinement import Refiner

        tokenizer = AutoTokenizer.from_pretrained(config.assets.generator, trust_remote_code=True)
        refiner = Refiner(config.assets.refiner, tokenizer, device, steps=config.inference.refiner_steps)
        for completed, i in enumerate(indices, 1):
            generated = read_json(output / "G" / f"{i:05d}.json")
            generated["graph"] = (
                torch.load(output / "graphs" / f"{i:05d}.pt", map_location="cpu", weights_only=False)
                if generated["graph_available"]
                else None
            )
            write_json(output / "F" / f"{i:05d}.json", refiner.refine(plans[i], generated))
            print(
                {"stage": stage, "worker": rank, "completed": completed, "requests": len(indices)}, flush=True
            )
    elif stage == "E":
        from .models import load_editor
        from .value import load_value
        from .editing import Editor

        model, tokenizer = load_editor(config.assets.base_model, config.assets.editor, device)
        editor = Editor(model, tokenizer, load_value(config.assets.value), config.inference)
        # Bounded chunks allow a resumed job to retain completed structures.
        for offset in range(0, len(indices), config.inference.editor_batch_size):
            chosen = indices[offset : offset + config.inference.editor_batch_size]
            current = [read_json(output / "F" / f"{i:05d}.json") for i in chosen]
            result, _ = editor.edit([plans[i] for i in chosen], current)
            for i, value in zip(chosen, result, strict=True):
                write_json(output / "E" / f"{i:05d}.json", value)
            print(
                {
                    "stage": stage,
                    "worker": rank,
                    "completed": min(offset + len(chosen), len(indices)),
                    "requests": len(indices),
                },
                flush=True,
            )
    else:
        raise ValueError(f"Unknown inference stage: {stage}")
