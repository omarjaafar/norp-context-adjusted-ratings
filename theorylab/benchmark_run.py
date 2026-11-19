"""The CP5 benchmark runner: 20 entries through Scout, Planner and the locked gate, three metrics.

    entry (benchmark/cp5_theories.json)
      -> Orchestrator.plan()          Scout + Planner, schema check; a rejection is a plan failure
      -> review_spec()                spec against the gold answer written before any run
      -> prepare_population()         literature: the theory's own population
                                      positive control: effect_sd outcome SDs planted along the exposure
                                      negative control: exposure replaced by another county's history
      -> run_spec()                   the same gate for every entry
      -> gate.apply_family()          TWO BH families: literature theories, and controls (CP5 audit X2)
      -> drill_down() by state        literature theories that survive only

Families (CP5 audit X2): the planted controls have tiny p-values, and under BH strong true
positives raise the thresholds for everything ranked after them. Sharing one family would let
the number of controls we planted make literature theories easier to pass, so literature
theories and controls are corrected in separate families.

Metrics (defined in docs/validator.md 2.9; the numbers are in data/output/benchmark_cp5.json):
  plan_validity         entries whose spec passed the schema, agreed with the gold answer AND ran
                        without error / all entries (CP4 A1: a spec that crashes is not usable; X5)
  control_recovery      positive controls the gate calls "supported" whose spec agreed with the
                        gold answer / positive controls (X6: planting along a wrong exposure is
                        not a recovery)
  false_positive_rate   negative controls the gate does not call "unsupported" / negative controls
A control whose plan failed counts as not recovered (positive) and is left out of the false-positive
rate (negative), and is listed, so a Planner failure is never mistaken for a gate failure.

Screening (CP4 Assumption 2): with screen_permutations set, every entry runs at that count first
(199), entries with a family q below rerun_q on H1 or H2 are rerun at the gate's full count
(999), and the family is applied again over the final reports. Which entries were rerun is recorded.
"""

from __future__ import annotations

import copy
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from theorylab.benchmark import review_spec
from theorylab.catalog import REPO_ROOT
from theorylab.drilldown import SURVIVOR_VERDICTS, drill_down
from theorylab.executor import to_float, to_jsonable
from theorylab.gate import apply_family
from theorylab.outcomes import apply_outcome_rule
from theorylab.spec import spec_columns
from theorylab.theory_run import add_derived, run_spec, select_population
from theorylab.validator import UnitPermuter

ENTRIES_PATH = REPO_ROOT / "benchmark" / "cp5_theories.json"
OUT_PATH = REPO_ROOT / "data" / "output" / "benchmark_cp5.json"
KINDS = ("literature", "positive_control", "negative_control")


def load_entries(path: Path = ENTRIES_PATH) -> list:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))["entries"]
    bad = [e["id"] for e in entries if e["kind"] not in KINDS]
    if bad:
        raise ValueError(f"unknown kind for {bad}")
    return entries


def _without_outcome_rule(spec: dict) -> dict:
    raw = copy.deepcopy(spec)
    (raw.get("validator") or {}).pop("outcome_rule", None)
    return raw


def prepare_population(entry: dict, spec: dict, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The rows this entry is gated on, with the RAW outcome. Controls change the population, never the gate.

    The outcome rule is not applied here: run_spec applies it once, after this (CP5 V2). Applying it
    here as well clipped every entry twice. A positive control is planted in the raw outcome, before
    the rule, at effect_sd x the outcome's SD AFTER the rule, measured on the population with nothing
    planted (decided 2026-11-19, docs/validator.md 2.8). The gate measures effects in post-rule SD, so
    that is the size it should see; the raw SD of a heavy-tailed ratio would plant a huge effect, and
    planting after the rule (the TA version) never tested whether the rule recovers an effect hidden
    in a raw heavy-tailed ratio (audit X3, CP4 feedback F4).
    """
    pop, log = select_population(add_derived(panel, spec_columns(spec)), _without_outcome_rule(spec))
    if pop.empty:
        raise ValueError("the population is empty")
    kind, ctl = entry["kind"], entry.get("control", {})
    outcome, exposure, sign = spec["outcome"], spec["exposure"], int(spec["expected_sign"])
    if kind == "positive_control":
        y = to_float(pop[outcome])
        x = to_float(pop[exposure])
        if spec.get("exposure_type", "binary") == "continuous":
            x = (x - x.mean()) / x.std(ddof=1)  # per SD of the exposure, as run_spec scales it
        else:
            x = x.eq(1.0).astype(float)
        after_rule, rule_log = apply_outcome_rule(pop, outcome, (spec.get("validator") or {}).get("outcome_rule"))
        sd_after_rule = float(to_float(after_rule[outcome]).std(ddof=1))
        shift = sign * float(ctl["effect_sd"]) * sd_after_rule
        pop = pop.copy()
        pop[outcome] = y + shift * x
        log["planted"] = {"effect_sd": float(ctl["effect_sd"]), "shift_per_unit_exposure": shift,
                          "planted_in": "raw outcome, before the outcome rule",
                          "scale": "outcome SD after the rule, unplanted population",
                          "sd_after_rule_unplanted": sd_after_rule, "raw_sd_unplanted": float(y.std(ddof=1)),
                          "rule_applied_to_unplanted": bool(rule_log["applied"])}
    elif kind == "negative_control":
        within = spec.get("exposure_varies_by")
        permuter = UnitPermuter(pop, exposure, unit=spec["exposure_unit"], within=within,
                                keep_paths=within is not None)
        pop = pop.copy()
        pop[exposure] = permuter.draw(np.random.default_rng(int(ctl["placebo_seed"])))
        log["placebo_seed"] = int(ctl["placebo_seed"])
    return pop, log


def plan_entries(orch, entries: list, variant: str = "text") -> tuple[list, list]:
    """[(entry, spec)] for entries the agents planned, and plan-failure rows for the rest.
    Every entry gets a row in `reviews`: schema validity and the gold-answer review."""
    planned, failures = [], []
    for e in entries:
        try:
            spec = copy.deepcopy(orch.plan(e[variant]))
        except ValueError as exc:
            failures.append({"id": e["id"], "kind": e["kind"], "error": str(exc)[:500]})
            continue
        spec["theory_id"] = e["id"]  # the Planner picks its own id; the benchmark's id is unique
        planned.append((e, spec))
    return planned, failures


def _run_one(entry: dict, spec: dict, panel: pd.DataFrame, n_permutations: int | None) -> dict:
    pop, log = prepare_population(entry, spec, panel)
    report = run_spec(pop, spec, n_permutations=n_permutations, include_robustness=False)
    report["benchmark"] = {"id": entry["id"], "kind": entry["kind"], "preparation": log}
    return report


FAMILIES = {"literature": ("literature",), "controls": ("positive_control", "negative_control")}


def apply_families(reports: dict, signs: dict, kinds: dict) -> dict:
    """BH within each family (FAMILIES), never across them. {id: report} in, {id: report} out."""
    out = {}
    for name, members in FAMILIES.items():
        group = [r for i, r in reports.items() if kinds[i] in members]
        if not group:
            continue
        for r in apply_family(group, signs):
            r["validator"]["fdr_family"] = f"{name}: {2 * len(group)} hypotheses ({len(group)} entries x H1, H2)"
            r["validator"]["family"] = name
            out[r["theory_id"]] = r
    return out


def _near_miss(report: dict, q: float) -> bool:
    return min(report["H1"]["family_q_value"], report["H2"]["family_q_value"]) < q


def run_planned(planned: list, panel: pd.DataFrame, *, n_permutations: int | None = None,
                screen_permutations: int | None = None, rerun_q: float = 0.25,
                drill_by: str | None = "state") -> dict:
    """Run every planned entry, apply one family, optionally screen then rerun near-misses."""
    errors, reports, entries_by_id = [], {}, {}
    n_first = screen_permutations or n_permutations
    seconds = {}
    for entry, spec in planned:
        t0 = time.time()
        try:
            reports[entry["id"]] = _run_one(entry, spec, panel, n_first)
        except (ValueError, KeyError) as exc:  # after the spec exists: a run error, not a Planner failure
            errors.append({"id": entry["id"], "kind": entry["kind"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        seconds[entry["id"]] = time.time() - t0
        entries_by_id[entry["id"]] = (entry, spec)

    signs = {i: int(spec["expected_sign"]) for i, (_, spec) in entries_by_id.items()}
    kinds = {i: entry["kind"] for i, (entry, _) in entries_by_id.items()}
    family = apply_families(reports, signs, kinds)
    rerun = []
    if screen_permutations and family:
        rerun = [i for i, r in family.items() if _near_miss(r, rerun_q)]
        for i in rerun:
            entry, spec = entries_by_id[i]
            t0 = time.time()
            reports[i] = _run_one(entry, spec, panel, n_permutations)
            seconds[i] += time.time() - t0
        family = apply_families(reports, signs, kinds)

    survivors = []
    for i, r in family.items():
        entry, spec = entries_by_id[i]
        if drill_by and entry["kind"] == "literature" and r["verdict"] in SURVIVOR_VERDICTS:
            r["drill_down"] = drill_down(panel, spec, r, drill_by, n_permutations=n_permutations)
            survivors.append(i)
    return {"reports": family, "run_errors": errors, "rerun_at_full_permutations": rerun,
            "drilled": survivors, "seconds": seconds,
            "kinds": {i: entries_by_id[i][0]["kind"] for i in family}}


def metrics(entries: list, reviews: dict, plan_failures: list, run: dict) -> dict:
    """The three benchmark metrics, from reviews (per entry), plan failures and the family verdicts."""
    kinds = {e["id"]: e["kind"] for e in entries}
    failed = {f["id"] for f in plan_failures}
    errored = {f["id"] for f in run["run_errors"]}
    verdict = {i: r["verdict"] for i, r in run["reports"].items()}

    def ids(kind):
        return [i for i, k in kinds.items() if k == kind]

    valid = [i for i in kinds if i not in failed and i not in errored and not reviews.get(i)]
    pos, neg = ids("positive_control"), ids("negative_control")
    recovered = [i for i in pos if verdict.get(i) == "supported" and not reviews.get(i)]
    neg_ran = [i for i in neg if i in verdict]
    false_pos = [i for i in neg_ran if verdict[i] != "unsupported"]
    lit = ids("literature")
    return {
        "plan_validity": {"valid": len(valid), "of": len(kinds), "rate": len(valid) / len(kinds),
                          "by_kind": {k: {"valid": sum(i in valid for i in ids(k)), "of": len(ids(k))}
                                      for k in KINDS},
                          "plan_failures": sorted(failed), "disagrees_with_gold": sorted(
                              i for i in kinds if i not in failed and reviews.get(i)),
                          "run_errors": sorted(errored)},
        "control_recovery": {"recovered": len(recovered), "of": len(pos),
                             "rate": len(recovered) / len(pos) if pos else None,
                             "missed": sorted(set(pos) - set(recovered)),
                             "supported_but_off_gold": sorted(
                                 i for i in pos if verdict.get(i) == "supported" and reviews.get(i))},
        "false_positive_rate": {"false_positives": len(false_pos), "of": len(neg_ran),
                                "rate": len(false_pos) / len(neg_ran) if neg_ran else None,
                                "ids": sorted(false_pos),
                                "not_run": sorted(set(neg) - set(neg_ran))},
        "literature_verdicts": {v: sorted(i for i in lit if verdict.get(i) == v)
                                for v in ("supported", "weak", "reversed", "unsupported")},
        "literature_not_run": sorted(i for i in lit if i not in verdict),
    }


def run_benchmark(orch, panel: pd.DataFrame | None = None, entries: list | None = None, *,
                  variant: str = "text",
                  n_permutations: int | None = None, screen_permutations: int | None = None,
                  rerun_q: float = 0.25, out_dir: Path | None = None) -> dict:
    """Plan and run the whole benchmark; write one report (and each entry's report) when out_dir is given."""
    entries = entries if entries is not None else load_entries()
    t0 = time.time()
    planned, failures = plan_entries(orch, entries, variant)
    planning_seconds = time.time() - t0
    reviews = {e["id"]: review_spec(spec, e["gold"]) for e, spec in planned}
    if panel is None and planned:  # load once, only the columns every planned spec needs
        panel = orch.panel_for([spec for _, spec in planned])
    run = run_planned(planned, panel, n_permutations=n_permutations,
                      screen_permutations=screen_permutations, rerun_q=rerun_q)
    result = to_jsonable({
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "variant": variant,
        "entries": len(entries),
        "permutations": {"final": n_permutations, "screen": screen_permutations, "rerun_q": rerun_q},
        "metrics": metrics(entries, reviews, failures, run),
        "plan_failures": failures,
        "reviews": reviews,
        "run_errors": run["run_errors"],
        "rerun_at_full_permutations": run["rerun_at_full_permutations"],
        "drilled": run["drilled"],
        "seconds_per_entry": run["seconds"],
        "planning_seconds": planning_seconds,
        "llm_usage": vars(orch.llm.usage).copy(),
        "table": [{"id": i, "kind": run["kinds"][i], "family": r["validator"].get("family"),
                   "verdict": r["verdict"],
                   "verdict_within_theory": r["verdict_within_theory"],
                   "H1_std": r["H1"]["standardized_effect"], "H1_family_q": r["H1"]["family_q_value"],
                   "H2_std": r["H2"]["fit"]["standardized_effect"], "H2_family_q": r["H2"]["family_q_value"],
                   "replicated": r["replication"]["replicated"],
                   "outcome_rule_applied": r["population"]["outcome_rule"]["applied"]}
                  for i, r in run["reports"].items()],
    })
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, r in run["reports"].items():
            (out_dir / f"{i}.json").write_text(json.dumps(to_jsonable(r), indent=2), encoding="utf-8")
        for e, spec in planned:
            (out_dir / f"{e['id']}.spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
