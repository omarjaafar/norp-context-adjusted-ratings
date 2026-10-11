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

Audit of the TA formulas (CP2). Checked against the sources above:
- Expense shares were 0 for filers that left the Part IX functional columns
  empty (only 501(c)(3) and 501(c)(4) organizations must fill them). A blank
  split is not a 0% program ratio, so shares and fundraising efficiency are
  now NaN unless the split adds up to total expenses (functional_split_reported).
- Impossible values are NaN, not kept: an expense share outside 0-1, or a
  negative fundraising expense or liabilities figure.
- Tuckman & Chang's four indicators match as coded. Two caveats: revenue
  concentration uses three sources (contributions, program revenue, other),
  fewer than their breakdown, so it understates concentration inside "other";
  and the administrative ratio divides by total expenses, where some later
  replications divide by total revenue.
- operating_margin and equity_ratio may be negative; that is real (a deficit,
  negative net assets), so they are kept.

standard_rating() is the context-blind rating the ratings-fairness theory
tests. Its definition and the reasons for it are in docs/ratings_theory.md.
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

# Part IX columns B-D must add up to column A within this share of the total.
SPLIT_TOLERANCE = 0.01

# Standard rating: ratio -> (value that scores 0, value that scores 10); linear in between.
RATING_ANCHORS = {
    "program_expense_ratio": (0.65, 0.85),
    "admin_expense_ratio": (0.35, 0.15),
    "fundraising_efficiency": (0.35, 0.10),
    "liabilities_to_assets": (1.00, 0.00),
}
RATING_MIN_COMPONENTS = 3
RATING_COLUMNS = ("standard_rating", "rating_components")


def safe_ratio(num, den, positive_den: bool = True) -> pd.Series:
    """num / den as float; NaN where den is missing, zero, or (by default) negative."""
    num = pd.to_numeric(num, errors="coerce").astype("float64")
    den = pd.to_numeric(den, errors="coerce").astype("float64")
    den = den.where(den > 0) if positive_den else den.where(den != 0)
    return num / den


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def functional_split_reported(total, program, management, fundraising) -> pd.Series:
    """True where Part IX program + management + fundraising add up to total expenses.

    A blank column counts as 0, so a filer with no fundraising still passes.
    False when total is missing or not positive, or when the three are off
    from the total by more than SPLIT_TOLERANCE (most often all three empty).
    """
    total = _num(total)
    parts = _num(program).fillna(0) + _num(management).fillna(0) + _num(fundraising).fillna(0)
    return (total > 0) & ((parts - total).abs() <= SPLIT_TOLERANCE * total)


def _share(part: pd.Series, total) -> pd.Series:
    """part / total, NaN outside 0-1 (a share of expenses cannot be negative or exceed 1)."""
    r = safe_ratio(part, total)
    return r.where((r >= 0) & (r <= 1))


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
    total = out["p09_total_expense"]
    split = functional_split_reported(total, out["p09_program_expense"],
                                      out["p09_management_expense"], out["p09_fundraising_expense"])
    program = _num(out["p09_program_expense"]).fillna(0).where(split)
    management = _num(out["p09_management_expense"]).fillna(0).where(split)
    fundraising = _num(out["p09_fundraising_expense"]).fillna(0).where(split)
    out["program_expense_ratio"] = _share(program, total)
    out["admin_expense_ratio"] = _share(management, total)
    out["fundraising_expense_ratio"] = _share(fundraising, total)
    # cost of raising one dollar of contributions
    out["fundraising_efficiency"] = safe_ratio(fundraising.where(fundraising >= 0),
                                               out["contributions"])
    rev = _num(out["revenue_total"])
    exp = _num(out["expenses_total"])
    out["operating_margin"] = safe_ratio(rev - exp, rev)
    out["equity_ratio"] = safe_ratio(out["net_assets_eoy"], rev)
    liabilities = _num(out["total_liabilities_eoy"])
    out["liabilities_to_assets"] = safe_ratio(liabilities.where(liabilities >= 0),
                                              out["total_assets_eoy"])
    out["revenue_concentration"] = revenue_concentration(
        out["contributions"], out["program_revenue"], out["revenue_total"])
    return add_standard_rating(out)


def rating_scores(df: pd.DataFrame, anchors: dict = RATING_ANCHORS) -> pd.DataFrame:
    """Score each rated ratio from 0 to 10 against fixed anchors.

    A ratio at or beyond its zero anchor scores 0, at or beyond its full
    anchor scores 10, and in between the score is linear. A missing ratio has
    a missing score. Columns are named score_<ratio>.
    """
    scores = {}
    for ratio, (zero, full) in anchors.items():
        value = _num(df[ratio])
        scores[f"score_{ratio}"] = ((value - zero) / (full - zero)).clip(0, 1) * 10
    return pd.DataFrame(scores, index=df.index)


def standard_rating(df: pd.DataFrame, anchors: dict = RATING_ANCHORS,
                    min_components: int = RATING_MIN_COMPONENTS) -> pd.Series:
    """Context-blind 0-10 rating: the plain mean of the available ratio scores.

    NaN when fewer than min_components of the ratios are defined. The rating
    never looks at where an organization is or whom it serves; that is the
    point of it (docs/ratings_theory.md).
    """
    scores = rating_scores(df, anchors)
    return scores.mean(axis=1).where(scores.notna().sum(axis=1) >= min_components)


def add_standard_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard_rating and rating_components (how many ratios it averages)."""
    out = df.copy()
    out["standard_rating"] = standard_rating(out)
    out["rating_components"] = rating_scores(out).notna().sum(axis=1)
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
