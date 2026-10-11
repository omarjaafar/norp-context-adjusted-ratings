"""County need table from ACS 5-year estimates plus NORP tract-level need.

One row per county per ACS year with population, median household income,
poverty rate and unemployment rate, plus a composite need index computed by
pandas (z-scores within each year).

Connecticut: the 2022 ACS reports it by planning region (09110-09190) and
2018-2021 by legacy county (09001-09015), but the EIN crosswalk places every
Connecticut organization in a planning region. ct_region_rows() rebuilds the
planning regions for 2018-2021 from town-level ACS (a region is a set of
whole towns), so Connecticut joins in every year. Those rows carry
acs_estimated = True and are scored against the published counties without
shifting the index for anyone else.

NORP's disadvantaged-communities table adds housing burden and food desert
share. It is one tract-level snapshot with no year, so the county aggregate
is the same for every ACS year. It uses legacy Connecticut counties, so the
planning-region rows get no NORP values.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from theorylab.catalog import Source

NEED_COLUMNS = (
    "county_fips", "acs_year", "acs_county_name", "population", "median_household_income",
    "poverty_universe", "below_poverty", "civilian_labor_force", "unemployed",
    "poverty_rate", "unemployment_rate", "need_index", "high_need", "acs_estimated",
)
# Added by add_norp_need(); not part of need_index.
NORP_NEED_COLUMNS = ("avg_housing_burden", "food_desert_share")

# Lower edges (dollars) of the 16 household-income brackets in ACS table B19001;
# the last bracket ($200,000 or more) is open-ended.
INCOME_BRACKET_EDGES = (0, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 45_000,
                        50_000, 60_000, 75_000, 100_000, 125_000, 150_000, 200_000)
INCOME_BRACKET_COLUMNS = tuple(f"hh_income_bracket_{i:02d}" for i in range(1, 17))
_COUNT_COLUMNS = ("population", "poverty_universe", "below_poverty",
                  "civilian_labor_force", "unemployed")


def acs_url(source: Source, year: int, api_key: str | None = None,
            geo_for: str = "county:*", geo_in: str | None = None) -> str:
    raw_vars = list(source.attributes) + list(source.fields)
    params = {"get": ",".join(raw_vars), "for": geo_for}
    if geo_in:
        params["in"] = geo_in
    if api_key:
        params["key"] = api_key
    return source.origin_for(year) + "?" + urllib.parse.urlencode(
        params, safe=",:*", quote_via=urllib.parse.quote)


def fetch_acs_rows(source: Source, year: int, api_key: str | None = None,
                   geo_for: str = "county:*", geo_in: str | None = None) -> list:
    req = urllib.request.Request(acs_url(source, year, api_key, geo_for, geo_in),
                                 headers={"User-Agent": "theorylab-cp2"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # The API answers 200 with an HTML "Missing Key" page when no key is sent.
        raise RuntimeError(f"Census API returned no data for {year}; "
                           "set CENSUS_API_KEY or pass --api-key") from None


def acs_rows_to_frame(rows: list, source: Source, year: int) -> pd.DataFrame:
    """Turn the Census API's list-of-lists into a typed frame (raw column names kept)."""
    header, body = rows[0], rows[1:]
    df = pd.DataFrame(body, columns=header)
    for col in source.fields:
        values = pd.to_numeric(df[col], errors="coerce")
        # Census uses large negative sentinels (e.g. -666666666) for "not available".
        df[col] = values.where(values >= 0)
    df["acs_year"] = int(year)
    return df


def to_canonical(acs: pd.DataFrame, source: Source) -> pd.DataFrame:
    """Rename a raw ACS frame's keys, fields and attributes to canonical names."""
    keys = {raw: name for name, raw in source.keys.items()}
    return acs.rename(columns={**source.fields, **source.attributes, **keys})


def to_need_table(acs: pd.DataFrame, source: Source,
                  estimated: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rename to canonical names and add rates and the index. acs may hold several years.

    estimated holds extra rows we built ourselves (ct_region_rows); they are
    appended with acs_estimated = True.
    """
    out = acs.rename(columns={**source.fields, **source.attributes})
    out["county_fips"] = (acs[source.keys["statefp"]].astype(str).str.zfill(2)
                          + acs[source.keys["countyfp"]].astype(str).str.zfill(3))
    out["acs_estimated"] = False
    if estimated is not None and len(estimated):
        out = pd.concat([out, estimated.assign(acs_estimated=True)], ignore_index=True)
        # estimated rows take the area's published name from the years that have one
        names = (out.dropna(subset=["acs_county_name"]).drop_duplicates("county_fips")
                 .set_index("county_fips")["acs_county_name"])
        out["acs_county_name"] = out["acs_county_name"].fillna(out["county_fips"].map(names))
    out = add_need_index(_add_rates(out))
    return out[list(NEED_COLUMNS)].sort_values(["acs_year", "county_fips"]).reset_index(drop=True)


def _add_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pu = out["poverty_universe"].astype("float64")
    lf = out["civilian_labor_force"].astype("float64")
    out["poverty_rate"] = out["below_poverty"].astype("float64") / pu.where(pu > 0)
    out["unemployment_rate"] = out["unemployed"].astype("float64") / lf.where(lf > 0)
    return out


def _z(s: pd.Series, ref: pd.Series) -> pd.Series:
    """z-score of s using the mean and spread of the reference rows only."""
    base = s[ref]
    sd = base.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - base.mean()) / sd


def add_need_index(df: pd.DataFrame) -> pd.DataFrame:
    """need_index = mean of z(poverty_rate), z(unemployment_rate), -z(log income), per year.

    high_need marks the top quartile of need_index within each ACS year
    (<NA> when the index is missing). Means, spreads and the quartile cut come
    from the published rows only; rows with acs_estimated = True are scored
    against them, so adding estimated rows never changes a published row.
    """
    out = df.copy()
    if "acs_estimated" not in out.columns:
        out["acs_estimated"] = False
    income = out["median_household_income"].astype("float64")
    out["_log_income"] = np.log(income.where(income > 0))
    out["need_index"] = np.nan
    out["high_need"] = pd.array([pd.NA] * len(out), dtype="boolean")
    for _, idx in out.groupby("acs_year").groups.items():
        g = out.loc[idx]
        ref = ~g["acs_estimated"].astype(bool)
        index = (_z(g["poverty_rate"], ref) + _z(g["unemployment_rate"], ref)
                 - _z(g["_log_income"], ref)) / 3
        out.loc[idx, "need_index"] = index
        high = (index >= index[ref].quantile(0.75)).astype("boolean").where(index.notna())
        out.loc[idx, "high_need"] = high
    return out.drop(columns="_log_income")


def grouped_median(counts: pd.DataFrame, edges=INCOME_BRACKET_EDGES) -> pd.Series:
    """Median of a bracketed distribution, by linear interpolation inside the bracket.

    counts has one column per bracket, in order; edges are the brackets' lower
    bounds. NaN when a row has no households or its median falls in the
    open-ended top bracket.
    """
    c = counts.astype("float64").to_numpy()
    lower = np.asarray(edges, dtype="float64")
    width = np.append(np.diff(lower), np.nan)
    half = c.sum(axis=1) / 2
    cum = c.cumsum(axis=1)
    k = (cum >= half[:, None]).argmax(axis=1)
    rows = np.arange(len(c))
    below = np.where(k > 0, cum[rows, np.maximum(k - 1, 0)], 0.0)
    in_bracket = c[rows, k]
    with np.errstate(divide="ignore", invalid="ignore"):
        median = lower[k] + (half - below) / in_bracket * width[k]
    median[~(half > 0)] = np.nan
    return pd.Series(median, index=counts.index)


def town_region_map(towns_2022: pd.DataFrame) -> pd.Series:
    """Connecticut town code -> planning-region county FIPS, from 2022 town rows.

    The 2022 ACS nests each town (county subdivision) under its planning
    region, so the mapping comes from the Census itself. Code 00000 is water
    with no town and is left out.
    """
    t = towns_2022[towns_2022["cousub"] != "00000"]
    fips = t["statefp"].str.zfill(2) + t["countyfp"].str.zfill(3)
    return pd.Series(fips.to_numpy(), index=t["cousub"].to_numpy(), name="county_fips")


def ct_region_rows(towns: pd.DataFrame, region_of_town: pd.Series) -> pd.DataFrame:
    """Sum town-level ACS rows (canonical names, any years) to planning regions.

    Counts add up exactly. The median household income cannot be added, so it
    is interpolated from the summed B19001 income brackets. Towns missing from
    the mapping are dropped; the caller checks for them.
    """
    t = towns[towns["cousub"] != "00000"].copy()
    t["county_fips"] = t["cousub"].map(region_of_town)
    t = t[t["county_fips"].notna()]
    sums = (t.groupby(["county_fips", "acs_year"])[[*_COUNT_COLUMNS, *INCOME_BRACKET_COLUMNS]]
            .sum(min_count=1).reset_index())
    sums["median_household_income"] = grouped_median(sums[list(INCOME_BRACKET_COLUMNS)]).round()
    return sums[["county_fips", "acs_year", *_COUNT_COLUMNS, "median_household_income"]]


def vintage_correlation(need: pd.DataFrame, columns, pairs) -> dict:
    """Pearson correlation of each column between two ACS years, over counties in both."""
    wide = need.pivot(index="county_fips", columns="acs_year")
    out = {}
    for a, b in pairs:
        both = wide.loc[wide[(columns[0], a)].notna() & wide[(columns[0], b)].notna()]
        out[f"{a}_vs_{b}"] = {"counties": int(len(both)),
                              **{c: round(float(both[(c, a)].corr(both[(c, b)])), 4)
                                 for c in columns}}
    return out


def load_norp_tracts(path, source: Source) -> pd.DataFrame:
    """Read the NORP tract table with raw column names; codes stay text."""
    codes = list(source.keys.values()) + list(source.attributes)
    return pd.read_csv(path, usecols=source.columns(), dtype={c: str for c in codes})


def _weighted_mean(values: pd.Series, weights: pd.Series, by: pd.Series) -> pd.Series:
    """Weighted mean per group over rows with a value and a positive weight."""
    w = weights.where(values.notna() & (weights > 0))
    total = w.groupby(by).sum(min_count=1)
    return (values * w).groupby(by).sum(min_count=1) / total


def aggregate_tract_need(tracts: pd.DataFrame, source: Source) -> pd.DataFrame:
    """Population-weighted county means of the NORP tract need measures.

    One row per county. A tract with zero population or a missing value does
    not count toward that measure; a county with no usable tract gets NaN.
    """
    t = tracts.rename(columns={**source.fields, **source.attributes})
    t["county_fips"] = t["county_fips"].astype(str).str.strip().str.zfill(5)
    pop = t["tract_population"].astype("float64")
    by = t["county_fips"]
    out = pd.DataFrame({
        "norp_tracts": t.groupby(by).size(),
        "norp_population": pop.groupby(by).sum(),
        "avg_housing_burden": _weighted_mean(t["avg_housing_burden"].astype("float64"), pop, by),
        "food_desert_share": _weighted_mean(t["food_desert_pct"].astype("float64"), pop, by),
    })
    return out.rename_axis("county_fips").reset_index()


def add_norp_need(need: pd.DataFrame, county_norp: pd.DataFrame) -> pd.DataFrame:
    """Attach the NORP county measures to every year of the need table."""
    cols = ["county_fips", *NORP_NEED_COLUMNS]
    return need.merge(county_norp[cols], on="county_fips", how="left", validate="many_to_one")
