"""Orchestrator (CP4): theory text -> Scout -> Planner -> spec -> Executor/Validator -> verdict -> drill-down.

    run_theory(text)   one theory through every stage; each stage's output is kept
    run_theories(...)  many theories, then one BH family over all of them (gate.apply_family)

The LLM agents are only used for the first two stages and are cached; everything
after the spec is deterministic Python (theory_run, validator, drilldown, gate).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from theorylab.agents import DataScout, Planner
from theorylab.catalog import REPO_ROOT
from theorylab.drilldown import SURVIVOR_VERDICTS, drill_down
from theorylab.gate import apply_family
from theorylab.llm import LLMClient
from theorylab.spec import validate_spec
from theorylab.theory_run import (available_columns, load_panel, run_spec, spec_columns,
                                  write_report, RUNS_DIR)

SPEC_TEMPLATE = REPO_ROOT / "specs" / "ratings_fairness.json"
DEFAULT_DRILL = "state"


class Orchestrator:
    def __init__(self, llm: LLMClient, panel: pd.DataFrame | None = None, *,
                 template_path: Path = SPEC_TEMPLATE, use_scout: bool = True,
                 drill_by: str | None = DEFAULT_DRILL, n_permutations: int | None = None):
        self.llm = llm
        self._panel = panel
        self.columns = available_columns()
        self.scout = DataScout(llm, self.columns) if use_scout else None
        self.planner = Planner(llm, str(template_path))
        self.drill_by = drill_by
        self.n_permutations = n_permutations

    def panel(self, spec: dict | None = None) -> pd.DataFrame:
        if self._panel is None:
            self._panel = load_panel(columns=sorted(spec_columns(spec))) if spec else load_panel()
        return self._panel

    def plan(self, theory: str) -> dict:
        """Theory text -> schema-checked spec. Raises ValueError when an agent's output is rejected."""
        if self.scout is not None:
            mapping = self.scout.map_variables(theory)
        else:  # fallback in force if CP4 slips: the Planner reads the catalog fields directly
            mapping = {"fields_available": self.columns}
        spec = self.planner.plan(theory, mapping, self.columns)
        assert validate_spec(spec, self.columns) == []
        return spec

    def run_theory(self, theory: str) -> dict:
        spec = self.plan(theory)
        panel = self.panel(spec)
        report = run_spec(panel, spec, n_permutations=self.n_permutations)
        if self.drill_by and report["verdict"] in SURVIVOR_VERDICTS:
            report["drill_down"] = drill_down(panel, spec, report, self.drill_by,
                                              n_permutations=self.n_permutations)
        return {"spec": spec, "report": report}

    def run_theories(self, theories: list, out_dir: Path = RUNS_DIR) -> dict:
        results, failed = [], []
        for text in theories:
            try:
                results.append(self.run_theory(text))
            except (ValueError, KeyError) as exc:  # a rejected plan is a result, not a crash
                failed.append({"theory": text, "error": f"{type(exc).__name__}: {exc}"})
        signs = {r["spec"]["theory_id"]: int(r["spec"]["expected_sign"]) for r in results}
        family = apply_family([r["report"] for r in results], signs) if results else []
        out_dir = Path(out_dir)
        for r, fam in zip(results, family):
            r["report"] = fam
            write_report(fam, out_dir / f"{fam['theory_id']}.json")
            (out_dir / f"{fam['theory_id']}.spec.json").write_text(json.dumps(r["spec"], indent=2),
                                                                    encoding="utf-8")
        summary = {"theories": [{"theory_id": r["spec"]["theory_id"], "verdict": r["report"]["verdict"],
                                 "verdict_within_theory": r["report"]["verdict_within_theory"]}
                                for r in results],
                   "plan_failures": failed,
                   "llm_usage": vars(self.llm.usage).copy()}
        write_report(summary, out_dir / "summary.json")
        return summary
