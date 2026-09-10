"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
import hashlib


SCHEMA = "sun_gated_token_revision_v1"


def derived_seed(source_id: str, stage: str, round_index: int = 0) -> int:
    if not source_id or not stage or type(round_index) is not int or round_index < 0:
        raise ValueError("a source, stage and nonnegative round are required")
    value = f"{SCHEMA}|{source_id}|{stage}|{round_index}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63)
