"""Planner and Data Scout agents (CP4).

Both are LLM roles with their own prompt, their own validation and their own
state; neither computes a statistic. Their output is checked by deterministic
code and rejected, not repaired, when it fails:
- Scout: theory text -> {variable role: panel column}. Every column must be a
  panel column or a derived column.
- Planner: theory text + Scout mapping -> a JSON spec that passes
  theorylab.spec.validate_spec() and uses the Scout's outcome, exposure and unit.

CP4 audit changes (W2, W4): the Planner's validator block is replaced by the fixed
GATE_POLICY (spec.lock_gate) and every difference is kept as an override, so an
agent cannot loosen the gate. Both agents see what each column means
(catalog/panel_columns.yaml), not just its name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import yaml

from theorylab.catalog import REPO_ROOT, load_catalog
from theorylab.llm import LLMClient, extract_json
from theorylab.spec import lock_gate, validate_spec
from theorylab.theory_run import DERIVED, available_columns

COLUMNS_PATH = REPO_ROOT / "catalog" / "panel_columns.yaml"
MAPPING_KEYS = ("outcome", "exposure", "exposure_unit")

SCOUT_SYSTEM = (
    "You are the Data Scout of a nonprofit-finance theory-testing pipeline. You map the variables "
    "of a theory to columns of an organization-year panel. You may only use column names from the "
    "list you are given. You never compute or estimate anything. Reply with one JSON object and "
    "nothing else.")

PLANNER_SYSTEM = (
    "You are the Planner of a nonprofit-finance theory-testing pipeline. You turn a theory into a "
    "JSON test spec in the exact format shown. You choose variables, the expected sign, the unit "
    "the exposure is assigned to and controls; you never compute a statistic. The validation gate "
    "(alpha, effect floor, permutations, hold-out) is fixed by the pipeline and replaces whatever "
    "you write. Use only the column names listed. Reply with one JSON object and nothing else.")


def column_dictionary(columns, path=COLUMNS_PATH) -> dict:
    """{column: 'description (level)'} for the given panel columns."""
    with open(path, encoding="utf-8") as f:
        described = yaml.safe_load(f)["columns"]
    out = {}
    for c in sorted(set(columns)):
        d = described.get(c)
        out[c] = f"{d['description']} ({d['level']})" if d else "(no description)"
    return out


def catalog_fields(catalog: dict | None = None) -> dict:
    """{canonical field: 'description (source)'} for every field in the data catalog."""
    catalog = catalog or load_catalog()
    out = {}
    for name, src in catalog.items():
        for canonical in src.canonical_names():
            out.setdefault(canonical, f"{src.description} [{name}]")
    return out


def scout_disagreements(spec: dict, mapping: dict) -> list:
    """Problems where the spec departs from the Scout's mapping (none without a Scout)."""
    if not all(k in mapping for k in MAPPING_KEYS):
        return []  # --no-scout: the Planner reads the column list directly
    problems = [f"{k} is {spec.get(k)!r} but the Scout mapped {mapping[k]!r}"
                for k in MAPPING_KEYS if spec.get(k) != mapping[k]]
    etype = spec.get("exposure_type", "binary")
    if "exposure_type" in mapping and etype != mapping["exposure_type"]:
        problems.append(f"exposure_type is {etype!r} but the Scout mapped {mapping['exposure_type']!r}")
    return problems


@dataclass
class DataScout:
    llm: LLMClient
    panel_columns: list
    history: list = field(default_factory=list)

    def prompt(self, theory: str) -> str:
        listing = "\n".join(f"- {c}: {d}" for c, d in
                            column_dictionary(set(self.panel_columns) | set(DERIVED)).items())
        return (f"Theory: {theory}\n\nPanel columns you may use (name: meaning (level)):\n{listing}\n\n"
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
        named = [mapping.get(k) for k in MAPPING_KEYS]
        named += list(mapping.get("controls", []))
        problems += [f"{c!r} is not a panel column" for c in named if c and c not in allowed]
        if mapping.get("exposure_type") not in ("binary", "continuous"):
            problems.append("exposure_type must be binary or continuous")
        return problems


@dataclass
class Planner:
    llm: LLMClient
    template_path: str
    max_attempts: int = 2
    history: list = field(default_factory=list)

    def prompt(self, theory: str, mapping: dict, errors: list | None = None,
               previous: dict | None = None) -> str:
        with open(self.template_path, encoding="utf-8") as f:
            example = f.read()
        text = (f"Theory: {theory}\n\nScout mapping: {json.dumps(mapping)}\n\n"
                f"Example spec for a binary exposure (follow this format exactly):\n{example}\n\n"
                'For exposure_type "continuous", set hypotheses H1.method to "correlation". '
                "Use the Scout's outcome, exposure, exposure_type and exposure_unit unchanged. "
                "Give a fresh theory_id (snake_case).")
        if errors:
            text += (f"\n\nYour previous spec:\n{json.dumps(previous)}\n\nIt was rejected: {errors}. "
                     "Return it with exactly these problems fixed and everything else unchanged.")
        return text

    def plan(self, theory: str, mapping: dict, columns=None) -> dict:
        columns = columns if columns is not None else available_columns()
        errors, proposed = None, None
        for attempt in range(1, self.max_attempts + 1):
            proposed = extract_json(self.llm.complete(PLANNER_SYSTEM, self.prompt(theory, mapping, errors, proposed)))
            spec, overrides = lock_gate(proposed)
            spec.setdefault("source", "planner")
            spec["written_by"] = f"Planner agent ({self.llm.model})"
            spec["gate_overrides"] = overrides
            errors = validate_spec(spec, columns) + scout_disagreements(spec, mapping)
            self.history.append({"attempt": attempt, "theory_id": spec.get("theory_id"),
                                 "errors": errors, "gate_overrides": overrides})
            if not errors:
                return spec
        raise ValueError("Planner spec rejected after " f"{self.max_attempts} attempts: {errors}")
