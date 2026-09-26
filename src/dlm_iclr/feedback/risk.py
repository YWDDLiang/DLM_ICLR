"""Cheap periodic-contact features and a source-balanced failure readout."""

from dlm_iclr.runtime.capacity import MAX_ATOMS

from collections import Counter
import json
from pathlib import Path
import numpy as np
from pymatgen.core import Structure


def structure(record):
    from dlm_iclr._core.continuous_keep_edit import structure_of

    return structure_of(record)


def focus_sites(positions, n):
    return sorted({(p - 8) // 4 for p in positions if p >= 8 and (p - 8) % 4 < 3}) or list(range(n))


def contacts(lattice, coords, focus):
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 1:
        # No coordinate repair is attempted for the rigid one-atom case.
        minimum = float(min(lattice.abc))
        return np.full((len(focus), 1), minimum), np.full(len(focus), minimum)
    distances = lattice.get_all_distances(coords[focus], coords)
    distances[np.arange(len(focus)), focus] = np.inf
    return distances, distances.min(1)


def contact_features(before, after, positions):
    n = len(before)
    focus = focus_sites(positions, n)
    d0, m0 = contacts(before.lattice, before.frac_coords, focus)
    d1, m1 = contacts(after.lattice, after.frac_coords, focus)
    low0 = float(m0.min())
    low1 = float(m1.min())
    crowd = float(np.exp(-np.square(d1 / np.maximum(m0[:, None], 0.05))).sum(1).mean())
    return [np.log1p(low0), np.log1p(low1), np.log((low0 + 0.05) / (low1 + 0.05)), crowd]


def features(before_record, after_record, positions):
    from dlm_iclr.feedback.value import geometry_features

    before = structure(before_record)
    after = structure(after_record)
    return np.asarray(
        geometry_features(before_record, after_record, positions, len(before))
        + contact_features(before, after, positions),
        dtype=float,
    )


def scalar_feature_grid(before, proposal, positions, site, axis, values):
    """Same 13 features for one-site coordinate alternatives, without 8B calls."""
    n = len(before)
    values = np.asarray(values, dtype=float)
    xyz = np.broadcast_to(proposal.frac_coords[site], (len(values), 3)).copy()
    xyz[:, axis] = values
    neighbors = np.delete(proposal.frac_coords, site, axis=0)
    d1 = before.lattice.get_all_distances(xyz, neighbors)
    d0 = before.lattice.get_all_distances(
        before.frac_coords[site : site + 1], np.delete(before.frac_coords, site, axis=0)
    )
    low0 = float(d0.min())
    low1 = d1.min(1)
    delta = xyz - before.frac_coords[site]
    delta -= np.round(delta)
    frac = np.linalg.norm(delta, axis=1)
    cart = np.linalg.norm(delta @ before.lattice.matrix, axis=1) / 10
    scope = len(set((p - 8) // 4 for p in positions if p >= 8))
    return np.column_stack(
        [
            np.full(len(values), n / MAX_ATOMS),
            np.full(len(values), scope / n),
            np.full(len(values), len(positions) / (6 + 3 * n)),
            np.zeros(len(values)),
            frac / np.sqrt(n),
            frac,
            cart / np.sqrt(n),
            cart,
            np.ones(len(values)),
            np.full(len(values), np.log1p(low0)),
            np.log1p(low1),
            np.log((low0 + 0.05) / (low1 + 0.05)),
            np.exp(-np.square(d1 / max(low0, 0.05))).sum(1),
        ]
    )


def target(score):
    status = score.get("terminal_status", score.get("status"))
    if status == "verified":
        return 0
    if status in ("invalid_raw", "invalid_terminal", "not_converged"):
        return 1
    return None


class RiskModel:
    def __init__(self, saved):
        self.saved = saved
        self.mean = np.asarray(saved["mean"])
        self.scale = np.asarray(saved["scale"])
        self.weight = np.asarray(saved["weight"])
        self.bias = float(saved["bias"])

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        logits = np.clip(((x - self.mean) / self.scale) @ self.weight + self.bias, -40, 40)
        return 1 / (1 + np.exp(-logits))

    def score(self, before, after, positions):
        return float(self.predict(features(before, after, positions)))


def fit_risk(x, y, sources, *, seed=20260915):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    x = np.asarray(x)
    y = np.asarray(y)
    sources = np.asarray(sources)
    unique = np.asarray(sorted(set(sources)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cut = max(1, int(len(unique) * 0.8))
    train_sources = set(unique[:cut])
    train = np.asarray([s in train_sources for s in sources])
    counts = Counter(sources[train])
    w = np.asarray([1 / counts[s] for s in sources[train]])
    w *= len(w) / w.sum()
    scaler = StandardScaler().fit(x[train], sample_weight=w)
    model = LogisticRegression(C=0.5, max_iter=400, solver="lbfgs", random_state=seed)
    model.fit(scaler.transform(x[train]), y[train], sample_weight=w)
    prob = model.predict_proba(scaler.transform(x[~train]))[:, 1]
    metrics = {
        "validation_rows": int((~train).sum()),
        "failure_prevalence": float(y[~train].mean()),
        "mean_absolute_error": float(np.abs(prob - y[~train]).mean()),
        "brier": float(brier_score_loss(y[~train], prob)),
        "auc": float(roc_auc_score(y[~train], prob)),
        "average_precision": float(average_precision_score(y[~train], prob)),
    }
    saved = {
        "schema": "periodic_contact_risk_v1",
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weight": model.coef_[0].tolist(),
        "bias": float(model.intercept_[0]),
        "seed": seed,
        "feature_width": x.shape[1],
        "training_sources": unique[:cut].tolist(),
        "validation_sources": unique[cut:].tolist(),
        "metrics": metrics,
        "target": "invalid_or_not_converged_under_unchanged_physics_budget",
        "physical_inputs_at_inference": False,
    }
    return saved
