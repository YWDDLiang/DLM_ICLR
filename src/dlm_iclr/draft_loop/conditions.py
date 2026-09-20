"""Fresh Planner conditions, deduplication, and leakage-free reservations (not a stability filter)."""
from __future__ import annotations
from collections import Counter
from functools import reduce
from math import gcd
import sqlite3
from pathlib import Path
from .common import digest


def composition_key(plan: dict) -> str:
    counts = Counter()
    for e, n in zip(plan["elements"], plan["counts"], strict=True):
        counts[e] += n
    d = reduce(gcd, counts.values())
    return "|".join(f"{e}:{n // d}" for e, n in sorted(counts.items()))


def condition_key(plan: dict) -> str:
    names = ("N", "elements", "counts", "anion_framework", "charge_bucket", "lattice_system",
             "spacegroup_bucket", "volume_per_atom_bin")
    return digest({name: plan[name] for name in names})


class ConditionLedger:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS conditions(k TEXT PRIMARY KEY, composition TEXT, round INTEGER, source TEXT);
        CREATE TABLE IF NOT EXISTS reservations(composition TEXT PRIMARY KEY, role TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attempts(k TEXT PRIMARY KEY, reason TEXT, round INTEGER);
        """)

    def reserve(self, plans: list[dict], role: str) -> None:
        with self.db:
            for row in plans:
                self.db.execute("INSERT OR IGNORE INTO reservations VALUES(?,?)",
                                (composition_key(row["plan_state"]), role))

    def admit(self, row: dict, round_id: int, cap: int, round_counts: Counter) -> tuple[bool, str]:
        """Every draw is auditable; no energy/force/evaluation field is used for admission."""
        source = row["source_id"]
        attempt_key = digest([round_id, source])
        existing = self.db.execute("SELECT reason FROM attempts WHERE k=?", (attempt_key,)).fetchone()
        if existing is not None:
            if existing[0] == "accepted":
                round_counts[composition_key(row["plan_state"])] += 1
            return existing[0] == "accepted", existing[0]
        if not row.get("body_eligible") or not isinstance(row.get("plan_state"), dict):
            reason, key, comp = "invalid_plan", None, None
        else:
            comp = composition_key(row["plan_state"])
            key = condition_key(row["plan_state"])
            if self.db.execute("SELECT 1 FROM reservations WHERE composition=?", (comp,)).fetchone():
                reason = "reserved_composition"
            elif self.db.execute("SELECT 1 FROM conditions WHERE k=?", (key,)).fetchone():
                reason = "duplicate_condition"
            elif round_counts[comp] >= cap:
                reason = "per_round_composition_cap"
            else:
                reason = "accepted"
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO attempts VALUES(?,?,?)", (attempt_key, reason, round_id))
            if reason == "accepted":
                self.db.execute("INSERT INTO conditions VALUES(?,?,?,?)", (key, comp, round_id, source))
                round_counts[comp] += 1
        return reason == "accepted", reason

    def close(self) -> None:
        self.db.close()
