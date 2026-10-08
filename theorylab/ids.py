"""Identifier normalization shared by every join."""

from __future__ import annotations

import pandas as pd


def normalize_ein(values: pd.Series) -> pd.Series:
    """Return EINs as 9-digit strings ("012345678"); anything unusable becomes <NA>.

    NCCS, the NORP NGO table and the NORP 990 extract do not agree on dashes or
    leading zeros, so every source goes through this before a merge.
    """
    digits = values.astype("string").str.replace(r"\D", "", regex=True)
    ok = digits.str.len().between(1, 9).fillna(False).astype(bool)
    return digits.where(ok).str.zfill(9)
