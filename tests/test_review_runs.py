"""scripts/review_theory_runs.py: stale-gate detection and gold matching (CP4 V6)."""

import importlib.util

from theorylab.benchmark import load_theories
from theorylab.catalog import REPO_ROOT
from theorylab.spec import GATE_POLICY

_spec = importlib.util.spec_from_file_location("review_runs", REPO_ROOT / "scripts" / "review_theory_runs.py")
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


def _validator(**over):
    v = {"alpha": GATE_POLICY["alpha"], "effect_floor_sd": GATE_POLICY["effect_floor_sd"],
         "n_permutations": GATE_POLICY["n_permutations"], "permutation_keep_paths": True,
         "permutation_studentized": True}
    return {"validator": {**v, **over}}


def test_current_gate_has_no_problems_and_old_runs_are_stale():
    assert review.gate_problems(_validator()) == []
    stale = review.gate_problems(_validator(permutation_studentized=False, n_permutations=199))
    assert any("rerun" in s for s in stale) and any("n_permutations" in s for s in stale)


def test_gold_is_found_even_when_the_planner_rewrites_the_text():
    theories = load_theories()
    gid, gold = review.gold_for({"theory": "reworded", "exposure": "chr_some_college",
                                 "outcome": "fundraising_efficiency"}, theories)
    assert gid == "education_fundraising" and gold["expected_sign"] == -1
