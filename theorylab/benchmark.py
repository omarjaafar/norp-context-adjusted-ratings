"""Benchmark theories and the review of a Planner spec against its gold answer (CP4).

Assumption 1 says the Planner's specs pass the schema check AND our review: right
variables, sign and unit of analysis. The TA script measured schema validity only
(CP4 audit W6). review_spec() makes the review mechanical: a spec agrees with a
theory when its outcome, exposure, exposure type, expected sign and exposure unit
equal the gold answer written down before any Planner run.
"""

from __future__ import annotations

import json
from pathlib import Path

from theorylab.catalog import REPO_ROOT

THEORIES_PATH = REPO_ROOT / "benchmark" / "cp4_theories.json"
REVIEWED = ("outcome", "exposure", "exposure_type", "expected_sign", "exposure_unit")


def load_theories(path: Path = THEORIES_PATH) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))["theories"]


def runnable(theory: dict, columns) -> bool:
    """True when the gold outcome and exposure are in the panel (chr_* need V2's rebuild)."""
    return {theory["gold"]["outcome"], theory["gold"]["exposure"]} <= set(columns)


def review_spec(spec: dict, gold: dict) -> list:
    """Where the spec departs from the gold answer; empty when it agrees."""
    got = dict(spec)
    got.setdefault("exposure_type", "binary")
    return [f"{k}: Planner {got.get(k)!r}, gold {gold[k]!r}" for k in REVIEWED if got.get(k) != gold[k]]
