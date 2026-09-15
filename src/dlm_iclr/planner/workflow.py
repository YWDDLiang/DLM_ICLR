"""H1A2 training: fresh LoRA, then a separate optimizer/scheduler stage."""

import subprocess
import sys
from pathlib import Path
from ..runtime.config import asset, run_root


def train(config, *, resume=False, device="cuda:0"):
    root = run_root(config)
    recipe = config["planner"]["training"]
    for stage in range(1, recipe["stages"] + 1):
        output = root / "planner" / f"epoch{stage}"
        if resume and (output / "train_metrics.json").exists() and (output / "final").exists():
            continue
        command = [
            sys.executable,
            "-m",
            "dlm_iclr.planner.trainer",
            "--model-path",
            asset(config, "planner_base"),
            "--data-dir",
            str(root / "data/planner"),
            "--output-dir",
            str(output),
        ]
        for key, value in recipe.items():
            if key != "stages":
                command += ["--" + key.replace("_", "-"), str(value)]
        command += ["--device", device]
        if stage > 1:
            command += ["--checkpoint-path", str(root / "planner" / f"epoch{stage - 1}" / "final")]
        if resume and (output / "checkpoints/last/state.pt").exists():
            command += ["--resume-from", str(output / "checkpoints/last")]
        subprocess.run(command, check=True)
    return {"checkpoint": str(root / "planner" / f"epoch{recipe['stages']}" / "final")}
