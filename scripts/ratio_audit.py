"""Checkpoint 2: audit of the ratio library and the standard rating on real filings.

Reads the committed NCCS extracts one tax year at a time (full Form 990 filers
only, since Parts IX and X come from that form), computes every ratio and the
standard rating, and writes data/output/ratio_audit.json: how many filings
report a usable functional expense split, how many impossible values were set
to missing, how often each ratio is defined, and how the rating is distributed.

Usage:
    python scripts/ratio_audit.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT, load_catalog  # noqa: E402
from theorylab.panel import KEYS, prepare_part  # noqa: E402
from theorylab.ratios import (RATING_ANCHORS, RATING_MIN_COMPONENTS, RATIO_COLUMNS,  # noqa: E402
                              add_ratios, functional_split_reported, rating_scores)

REPORT = REPO_ROOT / "data" / "output" / "ratio_audit.json"


def load_year(catalog, year: int) -> pd.DataFrame:
    """Full-990 filings for one tax year with Parts I, IX and X side by side."""
    parts = {}
    for name in ("nccs_p01", "nccs_p09", "nccs_p10"):
        raw = pd.read_csv(catalog[name].extract_path(year), dtype=str)
        keep = ("return_type",) if name == "nccs_p01" else ()
        parts[name], _ = prepare_part(raw, catalog[name], keep_meta=keep)
    p01 = parts["nccs_p01"]
    df = p01[p01["return_type"].eq("990")]
    for name in ("nccs_p09", "nccs_p10"):
        df = df.merge(parts[name], on=KEYS, how="left", validate="one_to_one")
    return df


def audit_year(df: pd.DataFrame) -> dict:
    n = len(df)
    total = df["p09_total_expense"]
    parts = df[["p09_program_expense", "p09_management_expense", "p09_fundraising_expense"]]
    has_total = total > 0
    split = functional_split_reported(total, *(parts[c] for c in parts))
    empty = has_total & parts.fillna(0).eq(0).all(axis=1)
    rated = add_ratios(df)
    scores = rating_scores(rated)
    rating = rated["standard_rating"]
    share = lambda mask: round(float(mask.mean()), 4)  # noqa: E731
    return {
        "full_990_filings": int(n),
        "functional_split": {
            "positive_total_expense": int(has_total.sum()),
            "split_adds_up": int(split.sum()),
            "split_all_empty_or_zero": int(empty.sum()),
            # entered as zeros, not left blank: the TA formulas scored these as a 0% program ratio
            "split_all_zero_not_blank": int((empty & parts.notna().all(axis=1)).sum()),
            "split_other_mismatch": int((has_total & ~split & ~empty).sum()),
            "share_without_usable_split": share(has_total & ~split) if n else None,
        },
        "impossible_values_set_missing": {
            "negative_fundraising_expense": int((df["p09_fundraising_expense"] < 0).sum()),
            "negative_liabilities": int((df["total_liabilities_eoy"] < 0).sum()),
        },
        "ratio_defined_share": {c: share(rated[c].notna()) for c in RATIO_COLUMNS},
        "standard_rating": {
            "rated": int(rating.notna().sum()),
            "rated_share": share(rating.notna()),
            "by_components": {str(k): int(v) for k, v in
                              rated["rating_components"].value_counts().sort_index().items()},
            "mean": round(float(rating.mean()), 3),
            "quantiles": {f"p{int(q * 100)}": round(float(rating.quantile(q)), 3)
                          for q in (0.10, 0.25, 0.50, 0.75, 0.90)},
            "share_scoring_10": {c.removeprefix("score_"): share(scores[c].dropna() == 10)
                                 for c in scores},
            "share_scoring_0": {c.removeprefix("score_"): share(scores[c].dropna() == 0)
                                for c in scores},
        },
    }


def main() -> int:
    catalog = load_catalog()
    by_year = {}
    for year in catalog["nccs_p01"].years:
        by_year[str(year)] = audit_year(load_year(catalog, year))
        print(year, by_year[str(year)]["standard_rating"]["rated_share"])
    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "scope": "full Form 990 filers in the committed NCCS extracts, one filing per EIN and tax year",
        "rating_anchors_zero_full": {k: list(v) for k, v in RATING_ANCHORS.items()},
        "rating_min_components": RATING_MIN_COMPONENTS,
        "by_tax_year": by_year,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
