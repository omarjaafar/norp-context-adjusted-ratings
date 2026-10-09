"""Organization-year panel: NCCS Parts I/IX/X + county + county need + ratios.

Every merge is recorded in a join log (rows in, rows matched, per tax year),
so the CP2 report can cite match rates instead of describing them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from theorylab.catalog import ContractError, Source, missing_columns
from theorylab.ids import normalize_ein
from theorylab.ratios import add_ratios, add_revenue_growth

KEYS = ["ein", "tax_year"]


def prepare_part(raw: pd.DataFrame, source: Source, keep_meta=()):
    """Raw NCCS extract -> canonical columns, one row per (ein, tax_year).

    Rows without a usable EIN or tax year are dropped. For repeated
    (ein, tax_year) pairs the filing with the latest RETURN_TIME_STAMP is kept,
    since an amended or corrected return is filed after the one it replaces;
    rows with no parseable timestamp rank lowest, and exact ties fall back to
    file order. (The TA version kept the last row in file order.) Filing-metadata
    columns are dropped unless named in keep_meta. Returns (frame, log).
    """
    missing = missing_columns(raw.columns, source)
    if missing:
        raise ContractError(f"{source.name}: missing contracted columns {missing}")
    raw = raw.reset_index(drop=True)
    out = pd.DataFrame(index=raw.index)
    out["ein"] = normalize_ein(raw[source.keys["ein"]])
    out["tax_year"] = pd.to_numeric(raw[source.keys["tax_year"]], errors="coerce").astype("Int64")
    for col, name in source.fields.items():
        out[name] = pd.to_numeric(raw[col], errors="coerce").astype("float64")
    for col, name in source.attributes.items():
        out[name] = raw[col]
    meta = source.filing_meta or {}
    for col, name in meta.items():
        out[name] = raw[col]

    rows_in = len(out)
    out = out.dropna(subset=KEYS)
    rows_with_keys = len(out)

    if "return_time_stamp" in out.columns:
        ts = pd.to_datetime(out["return_time_stamp"], errors="coerce", utc=True)
    else:
        ts = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    order = (out.assign(_ts=ts, _pos=range(len(out)))
             .sort_values(["_ts", "_pos"], na_position="first", kind="stable"))
    dup_mask = order.duplicated(KEYS, keep="last")
    dup_groups = order.loc[order.duplicated(KEYS, keep=False)]
    kept = order.loc[~dup_mask]
    log = {
        "rows_in": int(rows_in),
        "dropped_missing_keys": int(rows_in - rows_with_keys),
        "dropped_duplicate_keys": int(dup_mask.sum()),
        "rows_out": int(len(kept)),
    }
    if meta:
        winners = kept.loc[kept.set_index(KEYS).index.isin(dup_groups.set_index(KEYS).index)]
        amended = winners.get("return_amended")
        log["duplicate_resolution"] = {
            "rule": "latest RETURN_TIME_STAMP wins; missing timestamp ranks lowest; ties by file order",
            "duplicate_key_groups": int(dup_groups.drop_duplicates(KEYS).shape[0]),
            "kept_rows_marked_amended": int(amended.notna().sum()) if amended is not None else 0,
            "rows_missing_timestamp": int(ts.isna().sum()),
        }
    out = (kept.sort_values("_pos", kind="stable")
           .drop(columns=["_ts", "_pos"] + [n for n in meta.values() if n not in keep_meta])
           .reset_index(drop=True))
    return out, log


def _longest_run(years) -> int:
    ys = np.unique(np.asarray(years, dtype="int64"))
    if ys.size == 0:
        return 0
    best = run = 1
    for a, b in zip(ys[:-1], ys[1:]):
        run = run + 1 if b - a == 1 else 1
        best = max(best, run)
    return int(best)


def build_panel(p01: pd.DataFrame, p09: pd.DataFrame, p10: pd.DataFrame,
                crosswalk: pd.DataFrame, need: pd.DataFrame):
    """Join prepared parts, county and need into an org-year panel with ratios.

    p01 defines the panel rows (one per filing in Part I). Returns (panel, log).
    """
    log = {"inputs": {"p01_rows": int(len(p01)), "p09_rows": int(len(p09)),
                      "p10_rows": int(len(p10)), "crosswalk_eins": int(len(crosswalk)),
                      "need_rows": int(len(need))}}

    panel = p01.copy()
    for label, part in (("p09", p09), ("p10", p10)):
        panel = panel.merge(part, on=KEYS, how="left", validate="one_to_one", indicator=True)
        panel[f"has_{label}"] = panel["_merge"].eq("both")
        panel = panel.drop(columns="_merge")

    xw = crosswalk[["ein", "county_fips", "state", "category"]]
    panel = panel.merge(xw, on="ein", how="left", validate="many_to_one")
    panel["has_county"] = panel["county_fips"].notna()

    need_cols = need.drop(columns=["acs_county_name"], errors="ignore").rename(
        columns={"acs_year": "tax_year"})
    need_cols["tax_year"] = need_cols["tax_year"].astype("Int64")
    panel = panel.merge(need_cols, on=["county_fips", "tax_year"], how="left",
                        validate="many_to_one", indicator=True)
    panel["has_need"] = panel["_merge"].eq("both") & panel["has_county"]
    panel = panel.drop(columns="_merge")

    panel = add_ratios(panel)
    panel = add_revenue_growth(panel)

    by_year = {}
    for year, g in panel.groupby("tax_year"):
        n = len(g)
        by_year[str(int(year))] = {
            "rows": int(n),
            "with_p09": int(g["has_p09"].sum()),
            "with_p10": int(g["has_p10"].sum()),
            "with_county": int(g["has_county"].sum()),
            "with_need": int(g["has_need"].sum()),
            "county_match_rate": round(float(g["has_county"].mean()), 4),
            "need_match_rate": round(float(g["has_need"].mean()), 4),
        }
    log["by_tax_year"] = by_year

    linked = panel.loc[panel["has_county"], ["ein", "tax_year"]]
    runs = linked.groupby("ein")["tax_year"].agg(lambda s: _longest_run(s.dropna()))
    log["eins_in_panel"] = int(panel["ein"].nunique())
    log["eins_with_county"] = int(linked["ein"].nunique())
    log["eins_with_3plus_consecutive_linked_years"] = int((runs >= 3).sum())
    log["ratio_fill_rates"] = {
        c: round(float(panel[c].notna().mean()), 4) if len(panel) else 0.0
        for c in ("program_expense_ratio", "admin_expense_ratio", "fundraising_expense_ratio",
                  "fundraising_efficiency", "operating_margin", "equity_ratio",
                  "liabilities_to_assets", "revenue_concentration", "revenue_growth")
    }
    return panel, log


def assumption_4_check(log: dict, years=(2018, 2019, 2020, 2021, 2022), threshold: float = 0.8) -> dict:
    """Assumption 4: >= 80% of filers link to a county in every year 2018-2022."""
    rates = {y: log["by_tax_year"].get(str(y), {}).get("county_match_rate") for y in years}
    measured = [r for r in rates.values() if r is not None]
    return {
        "threshold": threshold,
        "county_match_rate_by_year": {str(y): r for y, r in rates.items()},
        "years_missing": [str(y) for y, r in rates.items() if r is None],
        "min_rate": min(measured) if measured else None,
        "eins_with_3plus_consecutive_linked_years": log["eins_with_3plus_consecutive_linked_years"],
        "holds": bool(measured) and len(measured) == len(rates) and min(measured) >= threshold,
    }
