"""Data checks carried into CP3: Part IX coverage by return type, and Assumption 5.

Part IX: in 2018 the panel has 351,711 rows with Part IX but only 263,339 full
Form 990 filings (data/output/panel_join_log.json vs ratio_audit.json). The
coverage table says which return types the extra Part IX rows belong to, so
the theory runs can restrict to the right filers (specs filter return_type).

Assumption 5: statsmodels MixedLM fits a random intercept by county on the
full-990 filers in under 10 minutes. assumption_5_check() turns a measured fit
(executor.multilevel) into the register's Holding / Weakened / Broken status.
"""

from __future__ import annotations

import pandas as pd

from theorylab.catalog import Source
from theorylab.panel import KEYS, prepare_part
from theorylab.ratios import functional_split_reported

A5_THRESHOLD_SECONDS = 600


def load_part(source: Source, keep_meta=("return_type",)) -> tuple[pd.DataFrame, dict]:
    """All years of one NCCS extract, one filing per (ein, tax_year), as build_panel.py does it."""
    frames = []
    for year in source.years:
        path = source.extract_path(year)
        if path is not None and path.exists():
            frames.append(pd.read_csv(path, dtype=str))
    if not frames:
        raise FileNotFoundError(f"no extracts for {source.name}; run build_nccs_extracts.py first")
    return prepare_part(pd.concat(frames, ignore_index=True), source, keep_meta=keep_meta)


def part9_coverage(p01: pd.DataFrame, p09: pd.DataFrame) -> dict:
    """Part IX rows by return type, per tax year.

    p01 and p09 come from prepare_part(..., keep_meta=("return_type",)). The
    return type is read from both tables: the Part I filing the panel keeps and
    the Part IX row itself, so disagreements show up too.
    """
    split_cols = ["p09_total_expense", "p09_program_expense", "p09_management_expense",
                  "p09_fundraising_expense"]
    left = p01[KEYS + ["return_type"]]
    right = p09[KEYS + ["return_type"] + split_cols].rename(columns={"return_type": "p09_return_type"})
    m = left.merge(right, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    m["usable_split"] = functional_split_reported(*(m[c] for c in split_cols))

    def counts(s: pd.Series) -> dict:
        return {str(k): int(v) for k, v in s.fillna("missing").value_counts().sort_index().items()}

    by_year = {}
    for year, g in m.groupby("tax_year"):
        both = g[g["_merge"] == "both"]
        by_year[str(int(year))] = {
            "p01_rows_by_return_type": counts(g.loc[g["_merge"] != "right_only", "return_type"]),
            "p09_rows_by_own_return_type": counts(g.loc[g["_merge"] != "left_only", "p09_return_type"]),
            "panel_rows_with_p09_by_p01_return_type": counts(both["return_type"]),
            "p09_rows_without_p01": int((g["_merge"] == "right_only").sum()),
            "full_990_without_p09": int(((g["_merge"] == "left_only") & g["return_type"].eq("990")).sum()),
            "return_type_disagreements": int((both["return_type"] != both["p09_return_type"]).sum()),
            "usable_split_share_by_p01_return_type": {
                str(rt): round(float(s.mean()), 4) for rt, s in both.groupby("return_type")["usable_split"]},
        }
    with_p09 = m[m["_merge"] == "both"]
    non_990 = with_p09[~with_p09["return_type"].eq("990")]
    return {
        "by_tax_year": by_year,
        "panel_rows_with_p09": int(len(with_p09)),
        "panel_rows_with_p09_not_full_990": int(len(non_990)),
        "not_full_990_return_types": counts(non_990["return_type"]),
        "rule": "analyses of Part IX ratios restrict to return_type == '990' (specs/*.json population)",
    }


def assumption_5_check(fit_details: dict, n_obs: int, n_groups: int,
                       threshold_seconds: float = A5_THRESHOLD_SECONDS) -> dict:
    """Register status for Assumption 5 from one measured MixedLM fit.

    Holding: converged within the threshold. Weakened: converged but slower, or
    fast but not converged. Broken: neither.
    """
    seconds = float(fit_details["fit_seconds"])
    converged = bool(fit_details["converged"])
    fast = seconds <= threshold_seconds
    status = "Holding (measured)" if converged and fast else (
        "Weakened (measured)" if converged or fast else "Broken (measured)")
    return {
        "statement": "statsmodels MixedLM fits a random intercept by county on full-990 filers "
                     f"in under {threshold_seconds / 60:g} minutes on a laptop",
        "threshold_seconds": threshold_seconds,
        "fit_seconds": seconds,
        "converged": converged,
        "n_obs": int(n_obs),
        "n_groups": int(n_groups),
        "status": status,
        "holds": converged and fast,
    }
