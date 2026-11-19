"""Recheck the reported results from the committed files, in about a minute.

    python scripts/verify_outputs.py            # checks that need only the repository
    python scripts/verify_outputs.py --panel    # also hashes the rebuilt panel (after build_panel.py)

Each check recomputes something from committed data or outputs and compares it with what the
reports claim. It prints [PASS], [FAIL] or [SKIP] per check, then one line:

    N/N checks passed -> data/output/validation_report.json

and exits 0 only when every check that ran passed. A [SKIP] means the input is not there yet
(for example the panel before build_panel.py); skipped checks are listed, never counted as passed.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.llm import LLMClient  # noqa: E402
from theorylab.spec import GATE_POLICY  # noqa: E402

OUT = REPO_ROOT / "data" / "output"
REPORT_PATH = OUT / "validation_report.json"


class Skip(Exception):
    """The input this check needs is not present."""


def _load(rel: str) -> dict:
    path = REPO_ROOT / rel
    if not path.exists():
        raise Skip(f"{rel} not found")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- checks: each returns a one-line detail on success and raises AssertionError on failure ---

def check_extracts_match_manifest() -> str:
    """Every committed NCCS extract has the sha256 and row count MANIFEST.json records (CP2)."""
    files = _load("data/extracts/nccs/MANIFEST.json")["files"]
    bad = [f["path"] for f in files if _sha256(REPO_ROOT / f["path"]) != f["sha256"]]
    assert not bad, f"sha256 differs: {bad}"
    return f"{len(files)} extracts, sha256 match"


def check_llm_cache_intact() -> str:
    """Every cached LLM response matches its stored hashes, so --replay reproduces the specs (CP4)."""
    result = LLMClient(replay_only=True).verify_cache()
    assert result["ok"] > 0, "no cache entries"
    assert not result["bad"], f"entries failing their hash: {result['bad']}"
    return f"{result['ok']} entries, 0 bad"


def check_cp4_verdicts_follow() -> str:
    """Each CP4 theory's verdict and family q-values recomputed from its stored p-values and
    effects (scripts/review_theory_runs.py), and they match the committed review.json."""
    review = _script("review_theory_runs").review(OUT / "theory_runs" / "cp4")
    assert review["all_follow"], "a verdict or family q does not follow from the stored numbers"
    assert review["all_current_gate"], "a run is not on the current gate"
    stored = {t["theory_id"]: t["verdict"] for t in _load("data/output/theory_runs/cp4/review.json")["theories"]}
    now = {t["theory_id"]: t["verdict_recomputed"] for t in review["theories"]}
    assert stored == now, f"review.json {stored} vs recomputed {now}"
    return f"{len(now)} theories: " + ", ".join(sorted(set(now.values())))


def check_outcome_tails() -> str:
    """CP4 finding: T2, T3, T5 outcomes are heavy-tailed (SD > 600x robust SD); T1, T4 are not (< 2x)."""
    tails = _load("data/output/outcome_tails.json")
    ratio = {k: v["sd"] / v["robust_sd"] for k, v in tails.items()}
    for k, v in tails.items():
        assert abs(ratio[k] - v["sd_over_robust_sd"]) < 1e-9 * max(1.0, ratio[k]), k
    heavy = sorted(k for k, r in ratio.items() if r > 600)
    light = sorted(k for k, r in ratio.items() if r < 2)
    assert len(heavy) == 3 and len(light) == 2, f"heavy {heavy}, light {light}"
    return f"heavy: {', '.join(heavy)}"


def check_sweep_cells() -> str:
    """Main sweep (rerun in CP5 under the outcome rule): 28 cells; a cell whose H1 and H2 effects are both under the floor must be
    unsupported, and none is above it."""
    cells = _load("data/output/sweep.json")["main_cells"]
    floor = GATE_POLICY["effect_floor_sd"]
    assert len(cells) == 28, f"{len(cells)} cells"
    below = [c for c in cells if abs(c["h1_r"]) < floor and abs(c["h2_std_effect"]) < floor]
    assert all(c["verdict"] == "unsupported" for c in below), "a cell under the floor is not unsupported"
    assert len(below) == len(cells), "a cell reaches the floor"
    biggest = max(max(abs(c["h1_r"]), abs(c["h2_std_effect"])) for c in cells)
    return f"28/28 unsupported, largest effect {biggest:.3f} SD"


def check_sweep_rule_before_after() -> str:
    """CP5 sweep rerun under the outcome rule: the cells the rule did not clip have exactly
    their CP4 effects (the rule leaves them alone), and only heavy-tailed outcomes were clipped."""
    log = _load("data/output/sweep/outcome_rule_log.json")["by_cell"]
    now = {c["cell"]: c for c in _load("data/output/sweep/main.json")["cells"]}
    cp4 = {c["cell"]: c for c in _load("data/output/sweep/main_cp4.json")["cells"]}
    assert set(log) == set(now) == set(cp4), "the three files cover different cells"
    untouched = [k for k, v in log.items() if not v["applied"]]
    for k in untouched:
        for f in ("h1_r", "h2_std_effect"):
            assert now[k][f] == cp4[k][f], f"{k} {f} moved without clipping"
    clipped = [k for k, v in log.items() if v["applied"]]
    assert all(log[k]["sd_over_robust_sd"] > log[k]["threshold"] for k in clipped)
    return f"{len(untouched)} unclipped cells identical to CP4; {len(clipped)} clipped"


def check_panel_hash() -> str:
    """The rebuilt panel's uncompressed CSV has the sha256 and row count in panel_join_log.json."""
    log = _load("data/output/panel_join_log.json")["panel"]
    path = REPO_ROOT / log["path"]
    if not path.exists():
        raise Skip(f"{log['path']} not built (python scripts/build_panel.py)")
    digest, rows, _ = _script("check_panel_hash").csv_hash_and_rows(path)
    assert (digest, rows) == (log["sha256_csv"], log["rows"]), f"{digest[:8]}..., {rows} rows"
    return f"{rows:,} rows, sha256_csv {digest[:8]}..."


CHECKS = {
    "extracts_match_manifest": check_extracts_match_manifest,
    "llm_cache_intact": check_llm_cache_intact,
    "cp4_verdicts_follow": check_cp4_verdicts_follow,
    "outcome_tails": check_outcome_tails,
    "sweep_cells": check_sweep_cells,
    "sweep_rule_before_after": check_sweep_rule_before_after,
}
SLOW = {"panel_hash": check_panel_hash}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="store_true", help="also hash the rebuilt panel (~1 min)")
    args = ap.parse_args()
    checks = CHECKS | (SLOW if args.panel else {})
    results = {}
    for name, fn in checks.items():
        try:
            status, detail = "PASS", fn()
        except Skip as exc:
            status, detail = "SKIP", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        results[name] = {"status": status, "detail": detail, "what": (fn.__doc__ or "").strip()}
        print(f"[{status}] {name}: {detail}")
    ran = [r for r in results.values() if r["status"] != "SKIP"]
    passed = sum(r["status"] == "PASS" for r in ran)
    REPORT_PATH.write_text(json.dumps({
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "passed": passed, "ran": len(ran),
        "skipped": sorted(k for k, r in results.items() if r["status"] == "SKIP"),
        "checks": results}, indent=2), encoding="utf-8")
    print(f"{passed}/{len(ran)} checks passed -> {REPORT_PATH.relative_to(REPO_ROOT).as_posix()}")
    return 0 if passed == len(ran) else 1


if __name__ == "__main__":
    sys.exit(main())
