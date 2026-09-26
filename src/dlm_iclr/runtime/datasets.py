"""Dataset presets shared by the command line and subprocess launchers."""
import os
import sys

PROFILES = {
    "mp20": (1, 20, "MP-20", [0.4, 10.0]),
    "perov-5": (5, 5, "Perov-5", [0.2, 4.0]),
    "mpts-52": (1, 52, "MPTS-52", [0.4, 10.0]),
}
ALIASES = {"mp-20": "mp20", "perov5": "perov-5", "perovskite": "perov-5",
           "mpts52": "mpts-52"}


def canonical_name(name):
    name = name.lower()
    return ALIASES.get(name, name)


def coverage_cutoffs(name):
    """Return an independent copy of the dataset's default coverage thresholds."""
    name = canonical_name(name)
    if name == "carbon":
        return [0.2, 4.0]
    if name not in PROFILES:
        raise ValueError(f"No coverage preset for {name!r}; supply explicit coverage_cutoffs")
    return list(PROFILES[name][3])


def profile(name):
    name = canonical_name(name)
    if name not in PROFILES:
        raise ValueError(f"Unsupported dataset {name!r}; choose {', '.join(PROFILES)}")
    minimum, maximum, label, _ = PROFILES[name]
    return {
        "dataset": {"name": name, "label": label, "min_atoms": minimum, "max_atoms": maximum,
                    "length_max_bin": 500,
                    "splits": {s: f"../datasets/{name}/{s}.csv" for s in ("train", "val", "test")}},
        "output": f"../outputs/{name}",
        "sampling": {"plans": "preset:mp20_default" if name == "mp20" else None, "requests": 1000},
        "evaluation": {"coverage_cutoffs": coverage_cutoffs(name)},
    }


def activate(config):
    dataset = config["dataset"]
    minimum, maximum = dataset.get("min_atoms", 1), dataset["max_atoms"]
    if not 1 <= minimum <= maximum <= 52:
        raise ValueError("Dataset atom limits must satisfy 1 <= min_atoms <= max_atoms <= 52")
    values = {"MAX_ATOMS": maximum, "MIN_ATOMS": minimum,
              "LENGTH_MAX_BIN": dataset.get("length_max_bin", 500),
              "DATASET_LABEL": dataset.get("label", dataset["name"])}
    loaded = sys.modules.get("dlm_iclr.runtime.capacity")
    if loaded and any(getattr(loaded, k) != v for k, v in values.items() if k != "DATASET_LABEL"):
        raise RuntimeError("Dataset capacity is already loaded. Use a new process for another dataset.")
    for key, value in values.items():
        os.environ["DLM_" + key] = str(value)
