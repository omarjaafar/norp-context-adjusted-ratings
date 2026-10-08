"""Derived financial ratio library.

All ratios are plain pandas arithmetic on the canonical column names from
catalog/catalog.yaml. Undefined values (zero or negative denominators,
missing inputs, non-consecutive years) are NaN, never 0 or inf.

Ratios follow the nonprofit-finance literature the plan cites:
- program / admin / fundraising expense shares (Part IX functional expenses),
  the ratios raters such as Charity Navigator build on;
- Tuckman & Chang (1991) vulnerability indicators: equity ratio, revenue
  concentration, administrative cost ratio, operating margin;
- liabilities-to-assets (Part X) and year-over-year revenue growth.
"""

from __future__ import annotations

import pandas as pd

RATIO_INPUTS = (
    "revenue_total", "expenses_total", "contributions", "program_revenue",
    "net_assets_eoy", "p09_total_expense", "p09_program_expense",
    "p09_management_expense", "p09_fundraising_expense",
    "total_assets_eoy", "total_liabilities_eoy",
)

RATIO_COLUMNS = (
    "program_expense_ratio", "admin_expense_ratio", "fundraising_expense_ratio",
    "fundraising_efficiency", "operating_margin", "equity_ratio",
    "liabilities_to_assets", "revenue_concentration",
)


def safe_ratio(num, den, positive_den: bool = True) -> pd.Series:
    """num / den as float; NaN where den is missing, zero, or (by default) negative."""
    num = pd.to_numeric(num, errors="coerce").astype("float64")
    den = pd.to_numeric(den, errors="coerce").astype("float64")
    den = den.where(den > 0) if positive_den else den.where(den != 0)
    return num / den


def revenue_concentration(contributions, program_revenue, revenue_total) -> pd.Series:
    """Herfindahl index over contributions, program revenue and all other revenue.

    1.0 = a single revenue source; 1/3 = evenly split. Other revenue is
    total minus the two named sources, floored at 0 (negative investment
    income can push it below). NaN when the components sum to <= 0.
    """
    contr = pd.to_numeric(contributions, errors="coerce").astype("float64").clip(lower=0)
    prog = pd.to_numeric(program_revenue, errors="coerce").astype("float64").clip(lower=0)
    total = pd.to_numeric(revenue_total, errors="coerce").astype("float64")
    other = (total - contr - prog).clip(lower=0)
    parts = pd.concat([contr, prog, other], axis=1)
    denom = parts.sum(axis=1, min_count=3)
    shares = parts.div(denom.where(denom > 0), axis=0)
    return (shares ** 2).sum(axis=1, min_count=3)


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with every column in RATIO_COLUMNS added."""
    missing = [c for c in RATIO_INPUTS if c not in df.columns]
    if missing:
        raise KeyError(f"add_ratios needs columns {missing}")
    out = df.copy()
    out["program_expense_ratio"] = safe_ratio(out["p09_program_expense"], out["p09_total_expense"])
    out["admin_expense_ratio"] = safe_ratio(out["p09_management_expense"], out["p09_total_expense"])
    out["fundraising_expense_ratio"] = safe_ratio(out["p09_fundraising_expense"], out["p09_total_expense"])
    # cost of raising one dollar of contributions
    out["fundraising_efficiency"] = safe_ratio(out["p09_fundraising_expense"], out["contributions"])
    rev = pd.to_numeric(out["revenue_total"], errors="coerce").astype("float64")
    exp = pd.to_numeric(out["expenses_total"], errors="coerce").astype("float64")
    out["operating_margin"] = safe_ratio(rev - exp, rev)
    out["equity_ratio"] = safe_ratio(out["net_assets_eoy"], rev)
    out["liabilities_to_assets"] = safe_ratio(out["total_liabilities_eoy"], out["total_assets_eoy"])
    out["revenue_concentration"] = revenue_concentration(
        out["contributions"], out["program_revenue"], out["revenue_total"])
    return out


def add_revenue_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Add revenue_growth = (rev_t - rev_{t-1}) / rev_{t-1} per EIN.

    Only defined when the previous filing is exactly one tax year earlier and
    its revenue is positive. Row order of df is preserved.
    """
    for c in ("ein", "tax_year", "revenue_total"):
        if c not in df.columns:
            raise KeyError(f"add_revenue_growth needs column {c!r}")
    out = df.sort_values(["ein", "tax_year"], kind="stable").copy()
    grouped = out.groupby("ein", sort=False)
    prev_rev = grouped["revenue_total"].shift(1)
    prev_year = grouped["tax_year"].shift(1)
    year = pd.to_numeric(out["tax_year"], errors="coerce").astype("float64")
    consecutive = (year - pd.to_numeric(prev_year, errors="coerce").astype("float64")) == 1
    rev = pd.to_numeric(out["revenue_total"], errors="coerce").astype("float64")
    growth = safe_ratio(rev - prev_rev, prev_rev)
    out["revenue_growth"] = growth.where(consecutive)
    return out.loc[df.index]
