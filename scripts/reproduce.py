#!/usr/bin/env python
"""Run the reproducible workflow, with a fresh process at each completed stage."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dlm_iclr.runtime.config import load, path, run_root  # noqa: E402
from dlm_iclr.runtime.datasets import canonical_name  # noqa: E402
from dlm_iclr.runtime.io import fingerprint, read_json, write_json  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all",
                   choices=["all", "prepare", "planner", "train-planner", "constructor", "periodic", "feedback", "inference"],
                   metavar="{all,prepare,planner,train-planner,constructor,periodic,feedback,inference}")
    p.add_argument("--dataset", help="mp20 (default), perov-5, mpts-52")
    p.add_argument("--config", type=Path)
    p.add_argument("--plans", type=Path, help="Saved Plan JSONL; required for non-MP20 inference")
    p.add_argument("--num-samples", type=int, help="Override the configured request count (default: 1000)")
    p.add_argument("--output", type=Path, help="Run root, including data, checkpoints and samples")
    p.add_argument("--device", help="Torch device, e.g. cuda:0")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-prepare", action="store_true", help="Reuse data already prepared in the run root")
    p.add_argument("--dry-run", action="store_true", help="Print resolved config and commands without execution")
    return p


def resolve(args):
    config_path = args.config or REPO / "configs" / (canonical_name(args.dataset or "mp20") + ".json")
    c = load(config_path, args.overrides, dataset=args.dataset)
    requests = c["sampling"]["requests"] if args.num_samples is None else args.num_samples
    if type(requests) is not int or requests < 1:
        raise ValueError("The request count must be a positive integer")
    c["sampling"]["requests"] = requests
    if args.num_samples is not None:
        c["planner"]["sampling"]["requests"] = requests
    if args.output:
        c["output"] = str(args.output.resolve())
    if args.device:
        c["runtime"]["devices"] = [args.device]
    if args.plans:
        c["sampling"]["plans"] = str(args.plans.resolve())
    root = run_root(c).resolve()
    # Snapshot has absolute external paths, so relocating it does not alter assets.
    result = deepcopy(c)
    result["output"] = str(root)
    result["dataset"]["splits"] = {k: str(path(c, v).resolve()) for k, v in c["dataset"]["splits"].items()}
    for key, value in c["models"].items():
        if value and not value.startswith(("hf:", "@run/")):
            result["models"][key] = str(path(c, value).resolve())
    for section, key in [(result["sampling"], "plans"), (result["feedback"]["training"], "plans")]:
        value = section.get(key)
        if value and not value.startswith(("preset:", "@run/")):
            section[key] = str(path(c, value).resolve())
    result.pop("_config_dir", None)
    return result, root


def commands(args, config, root):
    common = ["--config", str(root / "reproduce.config.json")]
    base = [sys.executable, "-m", "dlm_iclr"]
    steps = []
    needs_panel = args.stage in ("all", "planner", "inference")
    if needs_panel and not config["sampling"].get("plans"):
        raise ValueError("Pass --plans for Perov-5/MPTS-52; the packaged MP-20 Plan set is MP20 only")
    if args.stage in ("all", "prepare", "train-planner", "constructor", "periodic", "feedback", "inference") and not args.skip_prepare:
        steps.append([*base, "prepare", *common, *(["--with-planner"] if args.stage == "train-planner" else [])])
    if needs_panel:
        steps.append([*base, "plans", *common])
    modules = ["constructor", "periodic", "feedback"] if args.stage == "all" else ["planner"] if args.stage == "train-planner" else [args.stage] if args.stage in ("constructor", "periodic", "feedback") else []
    for module in modules:
        steps.append([*base, "train", module, *common, *(["--resume"] if args.resume else [])])
    if args.stage in ("all", "inference"):
        steps.append([*base, "run", *common, "--plans", str(root / "plans/evaluation.jsonl")])
    return steps


def main(argv=None):
    args = parser().parse_args(argv)
    config, root = resolve(args)
    steps = commands(args, config, root)
    if args.dry_run:
        print(json.dumps({"config": config, "commands": steps}, indent=2))
        return 0
    if args.stage in ("all", "feedback", "inference"):
        value = config["models"]["diffusion"]
        checkpoint = root / value[5:] if value.startswith("@run/") else Path(value)
        if not checkpoint.is_file():
            raise FileNotFoundError("Set models.diffusion to the dataset's frozen diffusion checkpoint before running")
    snapshot = root / "reproduce.config.json"
    if snapshot.exists() and fingerprint(read_json(snapshot)) != fingerprint(config):
        raise ValueError("Run configuration changed. Choose a new --output directory.")
    write_json(snapshot, config)
    import os
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for i, command in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True, env=env, cwd=REPO)
        write_json(root / "reproduction.progress.json", {"completed": i, "total": len(steps), "command": command})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
