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


def _bench_report(tid, p, effect, family, n_perm):
    fold = {"test": {"effect": effect}, "train": {"effect": effect}}
    return {"theory_id": tid, "verdict": "x", "verdict_within_theory": "x",
            "validator": {**_validator(n_permutations=n_perm)["validator"], "family": family},
            "H1": {"effect": effect, "standardized_effect": effect, "permutation": {"p_value": p}},
            "H2": {"fit": {"effect": effect, "standardized_effect": effect}, "permutation": {"p_value": p}},
            "replication": {"replicated": True, "folds": [fold] * 5, "folds_with_expected_sign": 5,
                            "pooled_p_value": 0.001, "pooled_effect": effect}, "population": {}}


def test_benchmark_folder_is_reviewed_in_two_families(tmp_path):
    """CP5 V7: q-values recomputed per family; a control filed in the literature family is caught;
    an entry left at the screening count is not flagged for it."""
    import json
    from theorylab.benchmark_run import apply_families, load_entries
    entries = {e["id"]: e for e in load_entries()}
    ids = {"ratings_fairness": 0.03, "lit_poverty_reserves": 0.5, "pos_highneed_rating": 0.001,
           "neg_population_admin": 0.6}
    kinds = {i: entries[i]["kind"] for i in ids}
    signs = {i: entries[i]["gold"]["expected_sign"] for i in ids}
    raw = {i: _bench_report(i, p, 0.3 * signs[i], None, 199 if i != "pos_highneed_rating" else 999)
           for i, p in ids.items()}
    judged = apply_families(raw, signs, kinds)
    judged["neg_population_admin"]["validator"]["family"] = "literature"  # filed in the wrong family
    for i, r in judged.items():
        (tmp_path / f"{i}.json").write_text(json.dumps(r))
        (tmp_path / f"{i}.spec.json").write_text(json.dumps(entries[i]["gold"]))
    (tmp_path / "benchmark.json").write_text(json.dumps(
        {"permutations": {"screen": 199, "final": None}, "rerun_at_full_permutations": ["pos_highneed_rating"]}))
    out = review.review(tmp_path)
    rows = {x["theory_id"]: x for x in out["theories"]}
    assert out["family_size"] == {"literature": 4, "controls": 4}
    assert out["all_follow"]
    assert not rows["neg_population_admin"]["own_family"] and not out["all_in_own_family"]
    assert rows["ratings_fairness"]["own_family"] and rows["ratings_fairness"]["gate_problems"] == []
    assert rows["pos_highneed_rating"]["gate_problems"] == []  # rerun at the full count
    assert all(not x["gold_disagreements"] for x in rows.values())
