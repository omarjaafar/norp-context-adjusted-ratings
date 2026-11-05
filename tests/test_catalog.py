"""Catalog row bounds: per-year expected_rows (CP3 feedback: the NCCS bounds were 100k-1M)."""

import json

import pandas as pd
import pytest
import yaml

from theorylab.catalog import REPO_ROOT, check_contract, load_catalog


def _catalog(tmp_path, expected_rows, years=(2018, 2019)):
    doc = {"sources": {"s": {
        "description": "test", "origin_kind": "url", "origin": "http://x/{year}", "years": list(years),
        "keys": {"ein": "EIN", "tax_year": "YEAR"}, "fields": {"V": "v"}, "attributes": {},
        "verified": [], "expected_rows": expected_rows, "extract": None}}}
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return load_catalog(path)["s"]


def _table(n):
    return pd.DataFrame({"EIN": [f"{i:09d}" for i in range(n)], "YEAR": ["2018"] * n, "V": ["1"] * n})


def test_per_year_bounds_are_checked_per_year(tmp_path):
    src = _catalog(tmp_path, {2018: [3, 3], 2019: [5, 5]})
    assert src.row_bounds(2019) == (5, 5)
    assert check_contract(_table(3), src, year=2018)["violations"] == []
    assert check_contract(_table(3), src, year=2019)["violations"]  # one changed file is caught
    with pytest.raises(ValueError, match="per year"):
        check_contract(_table(3), src)  # a yearly source needs the year


def test_per_year_bounds_must_cover_every_year(tmp_path):
    with pytest.raises(ValueError, match="cover exactly"):
        _catalog(tmp_path, {2018: [3, 3]})


def test_single_range_still_works(tmp_path):
    src = _catalog(tmp_path, [1, 10])
    assert src.row_bounds() == (1, 10) == src.row_bounds(2018)
    assert check_contract(_table(11), src)["violations"]


def test_nccs_bounds_match_the_measured_extracts():
    """The committed bounds are the row counts measured when the extracts were built."""
    catalog = load_catalog()
    manifest = json.loads((REPO_ROOT / "data" / "extracts" / "nccs" / "MANIFEST.json").read_text())
    files = [f for f in manifest["files"] if f["source"].startswith("nccs_")]
    assert len(files) == 15
    for f in files:
        assert catalog[f["source"]].row_bounds(f["year"]) == (f["rows"], f["rows"])
