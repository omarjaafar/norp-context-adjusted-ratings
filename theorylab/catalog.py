"""Data catalog and data contracts.

The catalog (catalog/catalog.yaml) lists every source the pipeline may read.
check_contract() compares a loaded table against its source's contract and
returns measurements that the build scripts write into their logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "catalog" / "catalog.yaml"

REQUIRED_KEYS = ("description", "origin_kind", "origin", "years", "keys", "fields",
                 "attributes", "verified", "expected_rows", "extract")
ORIGIN_KINDS = ("url", "norp_raw", "census_api")


class ContractError(ValueError):
    """A table does not satisfy its source's data contract."""


@dataclass(frozen=True)
class Source:
    name: str
    description: str
    origin_kind: str
    origin: str
    years: tuple
    keys: dict          # canonical key -> raw column
    fields: dict        # raw numeric column -> canonical name
    attributes: dict    # raw text column -> canonical name
    verified: tuple
    expected_rows: tuple
    extract: str | None

    def columns(self) -> list:
        """Raw columns this source must provide, in a stable order."""
        cols = list(self.keys.values()) + list(self.fields) + list(self.attributes)
        return list(dict.fromkeys(cols))

    def canonical_names(self) -> list:
        return list(self.keys) + list(self.fields.values()) + list(self.attributes.values())

    def origin_for(self, year: int | None = None) -> str:
        return self.origin.format(year=year) if year is not None else self.origin

    def extract_path(self, year: int | None = None) -> Path | None:
        if self.extract is None:
            return None
        rel = self.extract.format(year=year) if year is not None else self.extract
        return REPO_ROOT / rel


def _parse_source(name: str, raw: dict) -> Source:
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"catalog source {name!r} is missing {missing}")
    if raw["origin_kind"] not in ORIGIN_KINDS:
        raise ValueError(f"catalog source {name!r}: origin_kind must be one of {ORIGIN_KINDS}")
    rows = raw["expected_rows"]
    if not (isinstance(rows, list) and len(rows) == 2 and rows[0] <= rows[1]):
        raise ValueError(f"catalog source {name!r}: expected_rows must be [min, max]")
    src = Source(
        name=name,
        description=str(raw["description"]),
        origin_kind=raw["origin_kind"],
        origin=str(raw["origin"]),
        years=tuple(int(y) for y in (raw["years"] or [])),
        keys=dict(raw["keys"] or {}),
        fields=dict(raw["fields"] or {}),
        attributes=dict(raw["attributes"] or {}),
        verified=tuple(raw["verified"] or []),
        expected_rows=(int(rows[0]), int(rows[1])),
        extract=raw["extract"],
    )
    names = src.canonical_names()
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"catalog source {name!r}: canonical names used twice: {dupes}")
    unknown = [c for c in src.verified if c not in src.columns()]
    if unknown:
        raise ValueError(f"catalog source {name!r}: verified columns not in contract: {unknown}")
    return src


def load_catalog(path: Path | str = CATALOG_PATH) -> dict:
    """Load and validate the catalog. Returns {source name: Source}."""
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or "sources" not in doc:
        raise ValueError(f"{path}: expected a top-level 'sources' mapping")
    return {name: _parse_source(name, raw) for name, raw in doc["sources"].items()}


def missing_columns(columns, source: Source) -> list:
    present = set(columns)
    return [c for c in source.columns() if c not in present]


def check_contract(df: pd.DataFrame, source: Source, label: str | None = None) -> dict:
    """Check a raw table (raw column names) against its contract.

    Raises ContractError when contracted columns are missing. Everything else
    is measured and returned; `violations` lists what broke the contract so the
    caller can fail after logging it.
    """
    label = label or source.name
    missing = missing_columns(df.columns, source)
    if missing:
        raise ContractError(f"{label}: missing contracted columns {missing}")

    lo, hi = source.expected_rows
    key_cols = list(source.keys.values())
    null_keys = int(df[key_cols].isna().any(axis=1).sum())
    duplicate_keys = int(df.duplicated(key_cols, keep="first").sum())
    fill = {source.fields.get(c, source.attributes.get(c, c)): round(float(df[c].notna().mean()), 4)
            if len(df) else 0.0
            for c in list(source.fields) + list(source.attributes)}

    violations = []
    if not lo <= len(df) <= hi:
        violations.append(f"{label}: {len(df)} rows outside expected [{lo}, {hi}]")

    return {
        "label": label,
        "rows": int(len(df)),
        "expected_rows": [lo, hi],
        "null_key_rows": null_keys,
        "duplicate_key_rows": duplicate_keys,
        "fill_rates": fill,
        "violations": violations,
    }
