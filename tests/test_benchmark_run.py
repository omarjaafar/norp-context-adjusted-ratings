"""Benchmark entries, control preparation and the three metrics (CP5), without the LLM or the real panel."""

import copy

import numpy as np
import pandas as pd

from theorylab.benchmark import review_spec
from theorylab.benchmark_run import KINDS, load_entries, metrics, prepare_population
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
    assert m["control_recovery"] == {"recovered": 2, "of": 4, "rate": 0.5, "missed": ["p2", "p3"]}
    fp = m["false_positive_rate"]
    assert (fp["false_positives"], fp["of"], fp["ids"], fp["not_run"]) == (1, 3, ["n1"], ["n3"])
    assert m["literature_verdicts"]["weak"] == ["l0"]


def test_review_spec_accepts_a_spec_that_matches_gold():
    gold = {"outcome": "program_expense_ratio", "exposure": "poverty_rate", "exposure_type": "continuous",
            "expected_sign": -1, "exposure_unit": "county_fips"}
    spec = copy.deepcopy(_spec(-1))
    assert review_spec(spec, gold) == []
    assert review_spec({**spec, "expected_sign": 1}, gold)
