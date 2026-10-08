"""County need table from ACS 5-year estimates.

One row per county per ACS year with population, median household income,
poverty rate and unemployment rate, plus a composite need index computed by
pandas (z-scores within each year). The 2022 ACS reports Connecticut by
planning region (09110-09190); 2018-2021 use the legacy counties.
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
        return json.loads(r.read().decode("utf-8"))


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
