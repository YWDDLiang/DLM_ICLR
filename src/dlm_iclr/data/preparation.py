"""Content identities for reusing prepared crystal datasets."""

from importlib.metadata import version
from pathlib import Path

from ..runtime.io import file_hash, fingerprint, read_json
from ..runtime.config import path


def tokenizer_identity(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not callable(getattr(backend, "to_str", None)):
        return None
    return fingerprint(
        {
            "backend": backend.to_str(),
            "special_tokens": tokenizer.special_tokens_map,
            "chat_template": getattr(tokenizer, "chat_template", None),
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
        }
    )


def input_identity(config, tokenizer):
    token_hash = tokenizer_identity(tokenizer)
    if token_hash is None:
        return None
    sources = {}
    for split, source in config["dataset"]["splits"].items():
        source = path(config, source).resolve()
        members = sorted(source.glob("*.cif")) if source.is_dir() else [source]
        sources[split] = {"path": str(source), "files": [(p.name, file_hash(p)) for p in members]}
    package = Path(__file__).parents[1]
    implementations = sorted(package.rglob("*.py"))
    return fingerprint(
        {
            "dataset": {k: v for k, v in config["dataset"].items() if k != "splits"},
            "sources": sources,
            "seed": config["constructor"]["seed"],
            "tokenizer": token_hash,
            "implementations": {p.relative_to(package).as_posix(): file_hash(p) for p in implementations},
            "packages": {
                p: version(p)
                for p in ("transformers", "tokenizers", "pymatgen", "smact", "numpy", "spglib", "monty")
            },
        }
    )


def read_receipt(root):
    try:
        value = read_json(root / "preparation.json")
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def outputs(root, paths):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}


def matches(root, receipt, identity):
    if not identity or not isinstance(receipt, dict) or receipt.get("identity") != identity:
        return False
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        return False
    try:
        tokenizer_files = {n for n in files if n.startswith("tokenizer/")}
        if tokenizer_files and tokenizer_files != {
            p.relative_to(root).as_posix() for p in (root / "tokenizer").rglob("*") if p.is_file()
        }:
            return False
        for relative, expected in files.items():
            p = (root / relative).resolve()
            if not p.is_relative_to(root.resolve()) or not p.is_file() or file_hash(p) != expected:
                return False
    except (OSError, ValueError, TypeError):
        return False
    return True
