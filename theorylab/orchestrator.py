"""Orchestrator (CP4): theory text -> Scout -> Planner -> spec -> Executor/Validator -> verdict -> drill-down.

    run_theory(text)   one theory through every stage; each stage's output is kept
    run_theories(...)  many theories, then one BH family over all of them (gate.apply_family)

The LLM agents are only used for the first two stages and are cached; everything
after the spec is deterministic Python (theory_run, validator, drilldown, gate).

CP4 audit change (W8): every theory is planned first and the panel is loaded once
with the columns all of them need (the TA version loaded only the first theory's
columns and reused them). A spec the agents fail to produce is a plan failure; an
error after the spec exists is a run error, so it never counts against the Planner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from theorylab.agents import DataScout, Planner
from theorylab.catalog import REPO_ROOT
from theorylab.drilldown import STRATIFIERS, SURVIVOR_VERDICTS, drill_down
from theorylab.gate import apply_family
from theorylab.llm import LLMClient
from theorylab.spec import validate_spec
from theorylab.theory_run import (available_columns, load_panel, run_spec, spec_columns,
                                  write_report, RUNS_DIR)

SPEC_TEMPLATE = REPO_ROOT / "specs" / "ratings_fairness.json"
DEFAULT_DRILL = "state"
DRILL_COLUMNS = {"state": ["state"], "sector": ["category"], "size": ["log_revenue"]}


class Orchestrator:
    def __init__(self, llm: LLMClient, panel: pd.DataFrame | None = None, *,
                 template_path: Path = SPEC_TEMPLATE, use_scout: bool = True,
                 drill_by: str | None = DEFAULT_DRILL, n_permutations: int | None = None):
        if drill_by is not None and drill_by not in STRATIFIERS:
            raise ValueError(f"drill-down by {drill_by!r}; choose one of {sorted(STRATIFIERS)}")
        self.llm = llm
        self._panel = panel
        self.columns = available_columns()
        self.scout = DataScout(llm, self.columns) if use_scout else None
        self.planner = Planner(llm, str(template_path))
        self.drill_by = drill_by
        self.n_permutations = n_permutations

    def needed_columns(self, specs: list) -> list:
        cols = set()
        for spec in specs:
            cols |= spec_columns(spec)
        if self.drill_by:
            cols |= set(DRILL_COLUMNS[self.drill_by])
        return sorted(cols)

    def panel_for(self, specs: list) -> pd.DataFrame:
        if self._panel is None:
            self._panel = load_panel(columns=self.needed_columns(specs))
        return self._panel

    def plan(self, theory: str) -> dict:
        """Theory text -> schema-checked spec. Raises ValueError when an agent's output is rejected."""
        if self.scout is not None:
            mapping = self.scout.map_variables(theory)
        else:  # fallback in force if CP4 slips: the Planner reads the column list directly
            mapping = {"fields_available": self.columns}
        spec = self.planner.plan(theory, mapping, self.columns)
        assert validate_spec(spec, self.columns) == []
        return spec

    def execute(self, spec: dict, panel: pd.DataFrame) -> dict:
        """Spec -> report (+ drill-down for survivors). Deterministic."""
        report = run_spec(panel, spec, n_permutations=self.n_permutations)
        if self.drill_by and report["verdict"] in SURVIVOR_VERDICTS:
            report["drill_down"] = drill_down(panel, spec, report, self.drill_by,
                                              n_permutations=self.n_permutations)
        return report

    def run_theory(self, theory: str) -> dict:
        spec = self.plan(theory)
        return {"spec": spec, "report": self.execute(spec, self.panel_for([spec]))}

    def run_theories(self, theories: list, out_dir: Path = RUNS_DIR) -> dict:
        planned, plan_failures = [], []
        for text in theories:
            try:
                planned.append((text, self.plan(text)))
            except ValueError as exc:  # a rejected plan is a result, not a crash
                plan_failures.append({"theory": text, "error": f"{type(exc).__name__}: {exc}"})
        results, run_errors = [], []
        if planned:
            panel = self.panel_for([spec for _, spec in planned])
            for text, spec in planned:
                try:
                    results.append({"theory": text, "spec": spec, "report": self.execute(spec, panel)})
                except (ValueError, KeyError) as exc:
                    run_errors.append({"theory": text, "theory_id": spec.get("theory_id"),
                                       "error": f"{type(exc).__name__}: {exc}"})
        signs = {r["spec"]["theory_id"]: int(r["spec"]["expected_sign"]) for r in results}
        family = apply_family([r["report"] for r in results], signs) if results else []
        out_dir = Path(out_dir)
        for r, fam in zip(results, family):
            r["report"] = fam
            write_report(fam, out_dir / f"{fam['theory_id']}.json")
            (out_dir / f"{fam['theory_id']}.spec.json").write_text(json.dumps(r["spec"], indent=2),
                                                                    encoding="utf-8")
        summary = {"theories": [{"theory_id": r["spec"]["theory_id"], "verdict": r["report"]["verdict"],
                                 "verdict_within_theory": r["report"]["verdict_within_theory"],
                                 "gate_overrides": r["spec"].get("gate_overrides", [])}
                                for r in results],
                   "plan_failures": plan_failures,
                   "run_errors": run_errors,
                   "llm_usage": vars(self.llm.usage).copy()}
        write_report(summary, out_dir / "summary.json")
        return summary
