"""County need table from ACS 5-year estimates plus NORP tract-level need.

One row per county per ACS year with population, median household income,
poverty rate and unemployment rate, plus a composite need index computed by
pandas (z-scores within each year). The 2022 ACS reports Connecticut by
planning region (09110-09190); 2018-2021 use the legacy counties.

NORP's disadvantaged-communities table adds housing burden and food desert
share. It is one tract-level snapshot with no year, so the county aggregate
is the same for every ACS year. It uses legacy Connecticut counties, so the
2022 Connecticut planning-region rows get no NORP values.
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
    "poverty_rate", "unemployment_rate", "need_index", "high_need",
)
# Added by add_norp_need(); not part of need_index.
NORP_NEED_COLUMNS = ("avg_housing_burden", "food_desert_share")


def acs_url(source: Source, year: int, api_key: str | None = None) -> str:
    raw_vars = list(source.attributes) + list(source.fields)
    params = {"get": ",".join(raw_vars), "for": "county:*"}
    if api_key:
        params["key"] = api_key
    return source.origin_for(year) + "?" + urllib.parse.urlencode(params, safe=",:*")


def fetch_acs_rows(source: Source, year: int, api_key: str | None = None) -> list:
    req = urllib.request.Request(acs_url(source, year, api_key),
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


def to_need_table(acs: pd.DataFrame, source: Source) -> pd.DataFrame:
    """Rename to canonical names and add rates. acs may hold several years."""
    out = acs.rename(columns={**source.fields, **source.attributes})
    out["county_fips"] = (acs[source.keys["statefp"]].astype(str).str.zfill(2)
                          + acs[source.keys["countyfp"]].astype(str).str.zfill(3))
    pu = out["poverty_universe"].astype("float64")
    lf = out["civilian_labor_force"].astype("float64")
    out["poverty_rate"] = out["below_poverty"].astype("float64") / pu.where(pu > 0)
    out["unemployment_rate"] = out["unemployed"].astype("float64") / lf.where(lf > 0)
    out = add_need_index(out)
    return out[list(NEED_COLUMNS)].sort_values(["acs_year", "county_fips"]).reset_index(drop=True)


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def add_need_index(df: pd.DataFrame) -> pd.DataFrame:
    """need_index = mean of z(poverty_rate), z(unemployment_rate), -z(log income), per year.

    high_need marks the top quartile of need_index within each ACS year
    (<NA> when the index is missing).
    """
    out = df.copy()
    income = out["median_household_income"].astype("float64")
    out["_log_income"] = np.log(income.where(income > 0))
    g = out.groupby("acs_year")
    z_pov = g["poverty_rate"].transform(_z)
    z_unemp = g["unemployment_rate"].transform(_z)
    z_inc = g["_log_income"].transform(_z)
    out["need_index"] = (z_pov + z_unemp - z_inc) / 3
    q75 = out.groupby("acs_year")["need_index"].transform(lambda s: s.quantile(0.75))
    out["high_need"] = (out["need_index"] >= q75).astype("boolean").where(out["need_index"].notna())
    return out.drop(columns="_log_income")


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
