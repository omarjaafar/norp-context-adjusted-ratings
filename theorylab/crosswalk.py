"""EIN -> county FIPS crosswalk.

Geography comes from the NORP NGO table (STATE, COUNTY per EIN). County names
are resolved against the Census 2020 county reference file instead of the NORP
county_fips_lookup table, which has no Florida rows (found in Checkpoint 0).
Connecticut is handled both ways: legacy county names resolve to the 2020
county codes (09001-09015), and planning-region names resolve to the 2022+
planning-region codes (09110-09190) that the 2022 ACS uses.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from theorylab.ids import normalize_ein

# Longest first, so "CITY AND BOROUGH" is stripped before "BOROUGH".
_SUFFIXES = ("CITY AND BOROUGH", "PLANNING REGION", "CENSUS AREA", "MUNICIPALITY",
             "MUNICIPIO", "COUNTY", "PARISH", "BOROUGH")


def county_key(name) -> str:
    """Compact matching key for a county name.

    "Miami-Dade County" -> "MIAMIDADE", "St. Louis city" -> "STLOUISCITY",
    "De Soto Parish" and "DeSoto Parish" -> "DESOTO", "Doña Ana" -> "DONAANA".
    The word CITY is kept so Virginia/Missouri/Maryland independent cities do
    not collide with the county of the same name.
    """
    if name is None or (not isinstance(name, str) and pd.isna(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = s.upper().replace("&", " AND ")
    s = re.sub(r"[.'`,]", "", s)
    s = re.sub(r"[-/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\bSAINTE\b", "STE", s)
    s = re.sub(r"\bSAINT\b", "ST", s)
    for suffix in _SUFFIXES:
        if s.endswith(" " + suffix):
            s = s[: -len(suffix) - 1]
            break
    return s.replace(" ", "")


_CT_REGION_CODES = {
    "Capitol": "09110",
    "Greater Bridgeport": "09120",
    "Lower Connecticut River Valley": "09130",
    "Naugatuck Valley": "09140",
    "Northeastern Connecticut": "09150",
    "Northwest Hills": "09160",
    "South Central Connecticut": "09170",
    "Southeastern Connecticut": "09180",
    "Western Connecticut": "09190",
}
CT_PLANNING_REGIONS = {county_key(name): code for name, code in _CT_REGION_CODES.items()}

# Old or alternative names that the 2020 reference file spells differently.
ALIASES = {
    ("FL", "DADE"): "MIAMIDADE",
    ("SD", "SHANNON"): "OGLALALAKOTA",
    ("AK", "WADEHAMPTON"): "KUSILVAK",
}


def load_census_counties(path) -> pd.DataFrame:
    """Read the pipe-delimited Census county reference file."""
    try:
        ref = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        ref = pd.read_csv(path, sep="|", dtype=str, encoding="latin-1")
    ref["county_fips"] = ref["STATEFP"].str.zfill(2) + ref["COUNTYFP"].str.zfill(3)
    return ref


def build_reference_index(ref: pd.DataFrame):
    """(state abbreviation, county_key) -> 5-digit FIPS, plus state -> state FIPS."""
    index, collisions = {}, []
    for state, name, fips in ref[["STATE", "COUNTYNAME", "county_fips"]].itertuples(index=False):
        key = (state, county_key(name))
        if key in index and index[key] != fips:
            collisions.append({"state": state, "key": key[1], "kept": index[key], "dropped": fips})
            continue
        index[key] = fips
    state_fips = dict(zip(ref["STATE"], ref["STATEFP"].str.zfill(2)))
    return index, state_fips, collisions


def resolve_county(state, county, index: dict, state_fips: dict):
    """Return (county FIPS or None, method)."""
    if not isinstance(state, str) or not isinstance(county, str) or not county.strip():
        return None, "missing"
    state = state.strip().upper()
    raw = county.strip()
    if raw.isdigit():
        if len(raw) == 5:
            return raw, "fips_code"
        if len(raw) <= 3 and state in state_fips:
            return state_fips[state] + raw.zfill(3), "fips_code"
        return None, "unresolved"
    key = county_key(raw)
    if state == "CT" and key in CT_PLANNING_REGIONS:
        return CT_PLANNING_REGIONS[key], "ct_planning_region"
    if (state, key) in index:
        return index[(state, key)], "name"
    alias = ALIASES.get((state, key))
    if alias and (state, alias) in index:
        return index[(state, alias)], "alias"
    if (state, key + "CITY") in index:
        return index[(state, key + "CITY")], "name_city"
    return None, "unresolved"


def build_ein_crosswalk(ngo: pd.DataFrame, ref: pd.DataFrame, ein_col: str = "EIN",
                        state_col: str = "STATE", county_col: str = "COUNTY",
                        category_col: str = "CATEGORY"):
    """One row per EIN: ein, state, county_name, county_fips, geo_method, category.

    When an EIN appears more than once in the NGO table, a row whose county
    resolves is preferred, then the first row in file order. Returns
    (crosswalk, diagnostics).
    """
    index, state_fips, collisions = build_reference_index(ref)
    df = pd.DataFrame({
        "ein": normalize_ein(ngo[ein_col]).to_numpy(),
        "state": ngo[state_col].astype("string").str.strip().str.upper().to_numpy(),
        "county_name": ngo[county_col].to_numpy(),
        "category": ngo[category_col].to_numpy(),
    })

    pairs = df[["state", "county_name"]].drop_duplicates().reset_index(drop=True)
    resolved = [resolve_county(s if isinstance(s, str) else None, c, index, state_fips)
                for s, c in pairs.itertuples(index=False)]
    pairs["county_fips"] = [r[0] for r in resolved]
    pairs["geo_method"] = [r[1] for r in resolved]
    df = df.merge(pairs, on=["state", "county_name"], how="left", validate="many_to_one")

    rows_with_valid_ein = int(df["ein"].notna().sum())
    df = df[df["ein"].notna()]
    n_fips = df.dropna(subset=["county_fips"]).groupby("ein")["county_fips"].nunique()
    conflicting = int((n_fips > 1).sum())
    df = df.assign(_unresolved=df["county_fips"].isna())
    df = df.sort_values("_unresolved", kind="stable").drop_duplicates("ein", keep="first")
    xw = df.drop(columns="_unresolved").reset_index(drop=True)

    unresolved = pairs[pairs["county_fips"].isna()].merge(
        df.loc[df["county_fips"].isna()].groupby(["state", "county_name"], dropna=False)
        .size().rename("eins").reset_index(),
        on=["state", "county_name"], how="inner")
    top_unresolved = unresolved.sort_values("eins", ascending=False).head(25)

    by_state = xw.groupby("state", dropna=False)["county_fips"].apply(lambda s: s.notna().mean())
    diagnostics = {
        "ngo_rows": int(len(ngo)),
        "rows_with_valid_ein": rows_with_valid_ein,
        "unique_eins": int(len(xw)),
        "eins_with_conflicting_counties": conflicting,
        "eins_resolved": int(xw["county_fips"].notna().sum()),
        "resolved_share": round(float(xw["county_fips"].notna().mean()), 4) if len(xw) else 0.0,
        "method_counts": {str(k): int(v) for k, v in xw["geo_method"].value_counts(dropna=False).items()},
        "resolved_share_by_state": {str(k): round(float(v), 4) for k, v in by_state.items()},
        "reference_collisions": collisions,
        "top_unresolved_pairs": [
            {"state": None if pd.isna(r.state) else str(r.state),
             "county_name": None if pd.isna(r.county_name) else str(r.county_name),
             "method": r.geo_method, "eins": int(r.eins)}
            for r in top_unresolved.itertuples(index=False)
        ],
    }
    return xw, diagnostics


def match_rate(eins: pd.Series, crosswalk: pd.DataFrame) -> dict:
    """Share of unique EINs that have a resolved county in the crosswalk."""
    unique = pd.Series(normalize_ein(eins).dropna().unique())
    resolved = set(crosswalk.loc[crosswalk["county_fips"].notna(), "ein"])
    hits = int(unique.isin(resolved).sum())
    return {
        "unique_eins": int(len(unique)),
        "with_county": hits,
        "match_rate": round(hits / len(unique), 4) if len(unique) else 0.0,
    }
