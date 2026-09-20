"""Atomic artifacts, reproducible identities, and bounded execution."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any
from dlm_iclr.runtime.io import json_default

BASE_COMMIT = "2e78750dc4ca9a9bfce0b7cef7eda1d657e24402"
SCHEMA = "verified_draft_loop_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False,default=json_default).encode()).hexdigest()


def seed_for(seed: int, *parts: Any) -> int:
    return int(digest([seed, *parts])[:15], 16)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False,default=json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,default=json_default) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_identity(base: str, adapter: str | Path, head: str | Path) -> str:
    adapter = Path(adapter)
    if not adapter.exists() or not Path(head).is_file():
        raise FileNotFoundError("A real local draft checkpoint and C1 head are required")
    files = sorted(p for p in adapter.rglob("*") if p.is_file()
                   and (p.suffix in (".json", ".safetensors", ".bin", ".model") or p.name == "merges.txt"))
    if not files:
        raise ValueError("Empty model checkpoint")
    return digest({"base": base, "files": [(str(p.relative_to(adapter)), file_sha(p)) for p in files],
                   "head": file_sha(head)})


def shuffled(rows: list[dict], seed: int) -> list[dict]:
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result


@contextmanager
def exclusive_run(root: str | Path):
    """Fail rather than launching a second learner on the same output tree."""
    from filelock import FileLock, Timeout
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(root / ".lock")
    try:
        lock.acquire(timeout=0)
    except Timeout as error:
        raise RuntimeError("Another draft-loop process owns this output directory") from error
    try:
        yield
    finally:
        lock.release()


class Budget:
    """Persistent total budget; every attempted request is charged before work."""
    def __init__(self, path: Path, limits: dict):
        self.path, self.limits = path, limits
        from filelock import FileLock
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            self.state = read_json(path) if path.exists() else {"started": time.time(), "counts": {}}
            if not path.exists(): write_json(path, self.state)

    def reserve(self, kind: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("Negative resource reservation")
        from filelock import FileLock
        with FileLock(str(self.path) + ".lock"):
            self.state = read_json(self.path)
            if time.time() - self.state["started"] >= self.limits["max_wall_seconds"]:
                raise RuntimeError("BUDGET_STOP: elapsed wall-clock limit")
            used = self.state["counts"].get(kind, 0)
            cap = self.limits.get("max_" + kind)
            if cap is not None and used + amount > cap:
                raise RuntimeError(f"BUDGET_STOP: {kind} budget {used}+{amount}>{cap}")
            self.state["counts"][kind] = used + amount
            write_json(self.path, self.state)
