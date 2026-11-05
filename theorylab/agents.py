"""Planner and Data Scout agents (CP4).

Both are LLM roles with their own prompt, their own validation and their own
state; neither computes a statistic. Their output is checked by deterministic
code and rejected, not repaired, when it fails:
- Scout: theory text -> {variable role: catalog field}. Every field must be a
  canonical name in catalog/catalog.yaml or a panel column.
- Planner: theory text + Scout mapping -> a JSON spec that passes
  theorylab.spec.validate_spec().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from theorylab.catalog import load_catalog
from theorylab.llm import LLMClient, extract_json
from theorylab.spec import validate_spec
from theorylab.theory_run import DERIVED, available_columns

SCOUT_SYSTEM = (
    "You are the Data Scout of a nonprofit-finance theory-testing pipeline. You map the variables "
    "of a theory to fields of a data catalog. You may only use field names from the list you are "
    "given. You never compute or estimate anything. Reply with one JSON object and nothing else.")

PLANNER_SYSTEM = (
    "You are the Planner of a nonprofit-finance theory-testing pipeline. You turn a theory into a "
    "JSON test spec in the exact format shown. You choose variables, the expected sign, the unit "
    "the exposure is assigned to, controls and the hold-out; you never compute a statistic and you "
    "cannot change the validation gate. Use only the column names listed. Reply with one JSON "
    "object and nothing else.")


def catalog_fields(catalog: dict | None = None) -> dict:
    """{canonical field: 'description (source)'} for every field the Scout may request."""
    catalog = catalog or load_catalog()
    out = {}
    for name, src in catalog.items():
        for canonical in src.canonical_names():
            out.setdefault(canonical, f"{src.description} [{name}]")
    return out


@dataclass
class DataScout:
    llm: LLMClient
    panel_columns: list
    history: list = field(default_factory=list)

    def prompt(self, theory: str) -> str:
        cols = sorted(set(self.panel_columns) | set(DERIVED))
        return (f"Theory: {theory}\n\nPanel columns you may use: {cols}\n\n"
                'Return {"outcome": <column>, "exposure": <column>, "exposure_type": "binary" or '
                '"continuous", "exposure_unit": <column the exposure is assigned at, e.g. '
                'county_fips>, "controls": [<columns>], "notes": <one sentence>}.')

    def map_variables(self, theory: str) -> dict:
        mapping = extract_json(self.llm.complete(SCOUT_SYSTEM, self.prompt(theory)))
        problems = self.check(mapping)
        self.history.append({"theory": theory, "mapping": mapping, "problems": problems})
        if problems:
            raise ValueError("Data Scout output rejected: " + "; ".join(problems))
        return mapping

    def check(self, mapping: dict) -> list:
        allowed = set(self.panel_columns) | set(DERIVED)
        problems = [f"missing {k!r}" for k in ("outcome", "exposure", "exposure_type", "exposure_unit")
                    if k not in mapping]
        named = [mapping.get(k) for k in ("outcome", "exposure", "exposure_unit")]
        named += list(mapping.get("controls", []))
        problems += [f"{c!r} is not a catalog or panel field" for c in named if c and c not in allowed]
        if mapping.get("exposure_type") not in ("binary", "continuous"):
            problems.append("exposure_type must be binary or continuous")
        return problems


@dataclass
class Planner:
    llm: LLMClient
    template_path: str
    max_attempts: int = 2
    history: list = field(default_factory=list)

    def prompt(self, theory: str, mapping: dict, errors: list | None = None) -> str:
        with open(self.template_path, encoding="utf-8") as f:
            example = f.read()
        text = (f"Theory: {theory}\n\nScout mapping: {json.dumps(mapping)}\n\n"
                f"Example spec for a binary exposure (follow this format exactly):\n{example}\n\n"
                'For exposure_type "continuous", set hypotheses H1.method to "correlation". '
                "Give a fresh theory_id (snake_case).")
        if errors:
            text += f"\n\nYour previous spec was rejected: {errors}. Fix exactly these problems."
        return text

    def plan(self, theory: str, mapping: dict, columns=None) -> dict:
        columns = columns if columns is not None else available_columns()
        errors = None
        for attempt in range(1, self.max_attempts + 1):
            spec = extract_json(self.llm.complete(PLANNER_SYSTEM, self.prompt(theory, mapping, errors)))
            spec.setdefault("source", "planner")
            spec["written_by"] = f"Planner agent ({self.llm.model})"
            errors = validate_spec(spec, columns)
            self.history.append({"attempt": attempt, "theory_id": spec.get("theory_id"),
                                 "errors": errors})
            if not errors:
                return spec
        raise ValueError("Planner spec rejected after " f"{self.max_attempts} attempts: {errors}")
