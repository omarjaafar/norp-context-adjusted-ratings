"""Benchmark entries, control preparation and the three metrics (CP5), without the LLM or the real panel."""

import copy

import pytest

import numpy as np
import pandas as pd

from theorylab.benchmark import review_spec
from theorylab.outcomes import apply_outcome_rule
from theorylab.benchmark_run import KINDS, apply_families, load_entries, metrics, prepare_population
from theorylab.gate import apply_family
from theorylab.sweep import cell_spec
from theorylab.theory_run import available_columns


def test_the_benchmark_file_has_the_promised_shape():
    entries = load_entries()
    assert len(entries) == 20
    assert [sum(e["kind"] == k for e in entries) for k in KINDS] == [10, 5, 5]
    assert len({e["id"] for e in entries}) == 20
    assert len({e["text"] for e in entries}) == 20
    columns = set(available_columns())
    for e in entries:
        g = e["gold"]
        assert {g["outcome"], g["exposure"]} <= columns, e["id"]
        assert g["expected_sign"] in (-1, 1) and g["exposure_unit"] == "county_fips"
        assert g["exposure_type"] in ("binary", "continuous")
        if e["kind"] == "positive_control":
            assert e["control"]["effect_sd"] > 0.1  # above the gate's floor
        if e["kind"] == "negative_control":
            assert isinstance(e["control"]["placebo_seed"], int)
    # a gold answer that names a column in its own text would inflate Assumption 1 (CP4 audit W5)
    for e in entries:
        assert e["gold"]["outcome"] not in e["text"] and e["gold"]["exposure"] not in e["text"]
    seeds = [e["control"]["placebo_seed"] for e in entries if e["kind"] == "negative_control"]
    assert len(set(seeds)) == 5


def _panel(counties=40, per=10, seed=0):
    rng = np.random.default_rng(seed)
    x_county = rng.normal(size=counties)
    rows = []
    for t in (2019, 2020, 2021):
        for c in range(counties):
            for o in range(per):
                rows.append({"ein": f"{c:04d}{o:05d}", "tax_year": t, "county_fips": f"{c + 1000:05d}",
                             "state": f"S{c % 8}", "category": ["Health", "Education"][o % 2],
                             "return_type": "990", "has_county": "True", "has_need": "True",
                             "revenue_total": float(np.exp(13 + rng.normal())),
                             "program_expense_ratio": float(rng.normal()),
                             "poverty_rate": float(x_county[c])})
    return pd.DataFrame(rows)


def _entry(kind, **control):
    return {"id": f"x_{kind}", "kind": kind, "control": control}


def _spec(sign=-1):
    return cell_spec("program_expense_ratio", "poverty_rate", sign, n_permutations=99)


def test_positive_control_plants_the_effect_along_the_exposure():
    panel, spec = _panel(), _spec(-1)
    base, _ = prepare_population({"kind": "literature"}, spec, panel)
    pop, log = prepare_population(_entry("positive_control", effect_sd=0.5), spec, panel)
    sd = float(base["program_expense_ratio"].std(ddof=1))
    assert log["planted"]["effect_sd"] == 0.5
    delta = pop["program_expense_ratio"] - base["program_expense_ratio"]
    z = (base["poverty_rate"] - base["poverty_rate"].mean()) / base["poverty_rate"].std(ddof=1)
    assert np.allclose(delta, -0.5 * sd * z)
    assert base["poverty_rate"].equals(pop["poverty_rate"])  # the exposure is untouched


def test_positive_control_in_a_heavy_tailed_ratio_is_planted_raw_at_the_post_rule_sd():
    """CP5 V2 (audit X3): plant in the raw outcome at effect_sd x the SD after the rule; the rule runs once, later."""
    panel = _panel()
    rng = np.random.default_rng(11)
    panel["operating_margin"] = rng.normal(0, 0.2, len(panel))
    panel.loc[rng.random(len(panel)) < 0.01, "operating_margin"] = -1e4  # a few tiny-revenue filers
    spec = cell_spec("operating_margin", "poverty_rate", -1, n_permutations=99)
    base, base_log = prepare_population({"kind": "literature"}, spec, panel)
    assert "outcome_rule" not in base_log or not base_log["outcome_rule"]["applied"]  # raw here
    assert base["operating_margin"].min() == -1e4
    pop, log = prepare_population(_entry("positive_control", effect_sd=0.25), spec, panel)
    clipped, rule_log = apply_outcome_rule(base, "operating_margin", spec["validator"]["outcome_rule"])
    assert rule_log["applied"]
    post_sd = float(clipped["operating_margin"].std(ddof=1))
    assert log["planted"]["sd_after_rule_unplanted"] == pytest.approx(post_sd)
    assert post_sd < 0.01 * float(base["operating_margin"].std(ddof=1))  # not the raw SD
    z = (base["poverty_rate"] - base["poverty_rate"].mean()) / base["poverty_rate"].std(ddof=1)
    assert np.allclose(pop["operating_margin"] - base["operating_margin"], -0.25 * post_sd * z)


def test_negative_control_replaces_the_exposure_with_whole_county_histories():
    panel, spec = _panel(), _spec(1)
    base, _ = prepare_population({"kind": "literature"}, spec, panel)
    a, log = prepare_population(_entry("negative_control", placebo_seed=7), spec, panel)
    b, _ = prepare_population(_entry("negative_control", placebo_seed=7), spec, panel)
    c, _ = prepare_population(_entry("negative_control", placebo_seed=8), spec, panel)
    assert log["placebo_seed"] == 7
    assert a["poverty_rate"].equals(b["poverty_rate"])  # seeded
    assert not a["poverty_rate"].equals(c["poverty_rate"])
    assert not a["poverty_rate"].equals(base["poverty_rate"])
    assert a["program_expense_ratio"].equals(base["program_expense_ratio"])  # outcome untouched
    assert (a.groupby("county_fips")["poverty_rate"].nunique() == 1).all()  # one history per county


def _report(verdict):
    return {"verdict": verdict}


def test_metrics_count_plan_failures_and_controls_separately():
    entries = ([{"id": f"l{i}", "kind": "literature"} for i in range(2)]
               + [{"id": f"p{i}", "kind": "positive_control"} for i in range(4)]
               + [{"id": f"n{i}", "kind": "negative_control"} for i in range(4)])
    reviews = {"l0": [], "l1": ["outcome: Planner 'a', gold 'b'"], "p0": [], "p1": [], "p2": [], "p3": [],
               "n0": [], "n1": [], "n2": [], "n3": []}
    failures = [{"id": "p3"}]
    del reviews["p3"]
    run = {"reports": {"l0": _report("weak"), "l1": _report("unsupported"),
                       "p0": _report("supported"), "p1": _report("supported"), "p2": _report("weak"),
                       "n0": _report("unsupported"), "n1": _report("supported"),
                       "n2": _report("unsupported")},
           "run_errors": [{"id": "n3"}]}
    m = metrics(entries, reviews, failures, run)
    assert m["plan_validity"]["valid"] == 7 and m["plan_validity"]["of"] == 10
    assert m["plan_validity"]["plan_failures"] == ["p3"]
    assert m["plan_validity"]["disagrees_with_gold"] == ["l1"]
    assert m["plan_validity"]["run_errors"] == ["n3"]
    assert m["control_recovery"] == {"recovered": 2, "of": 4, "rate": 0.5, "missed": ["p2", "p3"],
                                     "supported_but_off_gold": []}
    fp = m["false_positive_rate"]
    assert (fp["false_positives"], fp["of"], fp["ids"], fp["not_run"]) == (1, 3, ["n1"], ["n3"])
    assert m["literature_verdicts"]["weak"] == ["l0"]


def test_review_spec_accepts_a_spec_that_matches_gold():
    gold = {"outcome": "program_expense_ratio", "exposure": "poverty_rate", "exposure_type": "continuous",
            "expected_sign": -1, "exposure_unit": "county_fips"}
    spec = copy.deepcopy(_spec(-1))
    assert review_spec(spec, gold) == []
    assert review_spec({**spec, "expected_sign": 1}, gold)


def test_a_supported_control_on_the_wrong_variable_is_not_recovered():
    # CP5 audit X6: the effect is planted along the Planner's exposure, so a spec that picked
    # the wrong exposure is "supported" without telling us anything about the gate.
    entries = [{"id": "p0", "kind": "positive_control"}, {"id": "p1", "kind": "positive_control"}]
    reviews = {"p0": [], "p1": ["exposure: Planner 'poverty_rate', gold 'unemployment_rate'"]}
    run = {"reports": {"p0": _report("supported"), "p1": _report("supported")}, "run_errors": []}
    m = metrics(entries, reviews, [], run)
    assert m["control_recovery"]["recovered"] == 1 and m["control_recovery"]["missed"] == ["p1"]
    assert m["control_recovery"]["supported_but_off_gold"] == ["p1"]
    assert m["plan_validity"]["valid"] == 1


def _gate_report(tid, p, effect, replicated):
    fold = {"test": {"effect": -0.2}, "train": {"effect": -0.2}}
    return {"theory_id": tid, "verdict": "x", "validator": {"effect_floor_sd": 0.1, "alpha": 0.05},
            "H1": {"effect": effect, "standardized_effect": effect, "permutation": {"p_value": p}},
            "H2": {"fit": {"effect": effect, "standardized_effect": effect}, "permutation": {"p_value": p}},
            "replication": {"replicated": replicated, "folds": [fold] * 5,
                            "pooled_p_value": 0.001, "pooled_effect": -0.2}}


def test_controls_do_not_loosen_the_literature_family():
    # CP5 audit X2. Literature: one borderline theory (p 0.03) and one null. Alone they are
    # 4 hypotheses, so the borderline one gets q = 0.03 * 4 / 2 = 0.06 and fails. Five planted
    # controls (p 0.001) in the same family make it 14 hypotheses with the borderline pair
    # ranked 11-12: q = 0.03 * 14 / 12 = 0.035, and it would pass only because we planted controls.
    reports = {"lit": _gate_report("lit", 0.03, -0.3, True), "lit_null": _gate_report("lit_null", 0.5, 0.0, False)}
    reports |= {f"pos{i}": _gate_report(f"pos{i}", 0.001, -0.3, True) for i in range(5)}
    kinds = {"lit": "literature", "lit_null": "literature"} | {f"pos{i}": "positive_control" for i in range(5)}
    signs = {i: -1 for i in reports}

    shared = {r["theory_id"]: r for r in apply_family(list(reports.values()), signs)}
    assert shared["lit"]["verdict"] == "supported"  # the leak the split removes

    fam = apply_families(reports, signs, kinds)
    assert fam["lit"]["H1"]["family_q_value"] == pytest.approx(0.06)
    assert fam["lit"]["verdict"] == "unsupported"
    assert fam["lit"]["validator"]["family"] == "literature"
    assert fam["pos0"]["validator"]["family"] == "controls"
    assert fam["pos0"]["H1"]["family_q_value"] == pytest.approx(0.001)  # 10 equal p-values
    assert fam["pos0"]["verdict"] == "supported"
    assert set(fam) == set(reports)
