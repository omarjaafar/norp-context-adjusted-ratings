"""Validator (CP3): the deterministic gate every finding must pass.

Pieces, each usable on its own:
- UnitPermuter + permutation_test: shuffle the exposure across the units it is
  assigned to (counties, within each tax year), re-estimate, and count how often
  a shuffled estimate is at least as large as the real one. Organizations in one
  county share that county's need, so shuffling organizations one by one would
  understate the noise; shuffling whole counties does not.
- benjamini_hochberg: false discovery rate control over a family of p-values.
- passes_effect_floor: a significant gap must also be large enough to matter,
  measured in outcome standard deviations.
- held_out_year_replication: fit on the other years, re-estimate on the held-out
  year, and require the same sign and significance there.
- decide_verdict: supported / weak / unsupported / reversed, from H1 (the gap
  exists) and H2 (the gap survives the controls), as docs/ratings_theory.md
  defines them.

Nothing here asks an LLM anything, and nothing an LLM writes can change a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from theorylab.executor import MethodResult, to_float

VERDICTS = ("supported", "weak", "unsupported", "reversed")


# ---------------------------------------------------------------- permutation

class UnitPermuter:
    """Draws permutations of one column, shuffled across units within strata.

    unit=None shuffles rows. With unit="county_fips" and within="tax_year", each
    county-year keeps one exposure value, and the values are shuffled among the
    counties of the same tax year, so every draw has the same number of exposed
    counties per year as the data. Raises if the column varies inside a unit.
    """

    def __init__(self, df: pd.DataFrame, column: str, unit: str | None = None,
                 within: str | None = None):
        values = to_float(df[column]).to_numpy()
        if np.isnan(values).any():
            raise ValueError(f"{column} has missing values; select the population first")
        for key in (unit, within):
            if key is not None and df[key].isna().any():
                raise ValueError(f"{key} has missing values; select the population first")
        self.column = column
        self._values = values
        if unit is None:
            self._codes = np.arange(len(df))
            level = values.copy()
            strata = (pd.factorize(df[within], sort=True)[0] if within
                      else np.zeros(len(df), dtype="int64"))
        else:
            keys = ([within] if within else []) + [unit]
            frame = df[keys].copy()
            frame["_v"] = values
            grouped = frame.groupby(keys, sort=True)
            self._codes = grouped.ngroup().to_numpy()
            agg = grouped["_v"].agg(["first", "nunique"])
            varying = int((agg["nunique"] > 1).sum())
            if varying:
                raise ValueError(f"{column} varies inside {varying} {unit} units; it cannot be "
                                 f"permuted at that level")
            level = agg["first"].to_numpy(dtype="float64")
            strata = (pd.factorize(agg.index.get_level_values(0), sort=True)[0] if within
                      else np.zeros(len(agg), dtype="int64"))
        self._level = level
        self._strata = [np.flatnonzero(strata == s) for s in np.unique(strata)]
        self.n_units = int(len(level))

    def observed(self) -> np.ndarray:
        return self._values.copy()

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        v = self._level.copy()
        for pos in self._strata:
            v[pos] = rng.permutation(v[pos])
        return v[self._codes]


def permutation_test(statistic, permuter: UnitPermuter, n_permutations: int = 999, seed: int = 0,
                     alternative: str = "two-sided") -> dict:
    """p-value of statistic(exposure) against its permutation distribution.

    statistic: function of the exposure vector (row-aligned) returning a float.
    p = (1 + number of draws at least as extreme) / (n_permutations + 1), so it is
    never 0. alternative: "two-sided" compares absolute values, "less"/"greater"
    one tail.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    observed = float(statistic(permuter.observed()))
    rng = np.random.default_rng(seed)
    null = np.array([statistic(permuter.draw(rng)) for _ in range(n_permutations)], dtype="float64")
    slack = 1e-12 * max(abs(observed), 1.0)
    if alternative == "two-sided":
        extreme = np.abs(null) >= abs(observed) - slack
    elif alternative == "less":
        extreme = null <= observed + slack
    elif alternative == "greater":
        extreme = null >= observed - slack
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return {
        "observed": observed,
        "p_value": float((1 + int(extreme.sum())) / (n_permutations + 1)),
        "n_permutations": int(n_permutations),
        "units_permuted": permuter.n_units,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if n_permutations > 1 else float("nan"),
        "seed": int(seed),
        "alternative": alternative,
    }


def difference_in_means(y: np.ndarray, exposure: np.ndarray) -> float:
    """mean(y | exposure = 1) - mean(y | exposure = 0)."""
    exposed = exposure == 1
    if not exposed.any() or exposed.all():
        return float("nan")
    return float(y[exposed].mean() - y[~exposed].mean())


# ---------------------------------------------------------------- multiple testing, size

def benjamini_hochberg(p_values, alpha: float = 0.05):
    """Benjamini-Hochberg adjusted p-values (q-values) and which hypotheses are rejected.

    Missing p-values stay missing and are never rejected; they do not count
    toward the family size.
    """
    p = np.asarray(p_values, dtype="float64")
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m:
        pv = p[ok]
        order = np.argsort(pv, kind="stable")
        ranked = pv[order] * m / np.arange(1, m + 1)
        adjusted = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
        out = np.empty(m)
        out[order] = adjusted
        q[ok] = out
    reject = np.where(np.isfinite(q), q <= alpha, False)
    return q, reject


def passes_effect_floor(standardized_effect: float, floor: float) -> bool:
    return bool(np.isfinite(standardized_effect) and abs(standardized_effect) >= floor)


# ---------------------------------------------------------------- replication

def held_out_year_replication(df: pd.DataFrame, fit, holdout_year: int, *, time_col: str = "tax_year",
                              alpha: float = 0.05, expected_sign: int | None = None) -> dict:
    """Fit on every year but holdout_year, then on holdout_year alone.

    fit: function frame -> MethodResult. Replicated means the held-out estimate
    has the training sign (and the expected sign, when given) and p < alpha.
    """
    years = to_float(df[time_col])
    train, test = df[years != holdout_year], df[years == holdout_year]
    if train.empty or test.empty:
        raise ValueError(f"held-out year {holdout_year} leaves an empty training or test set")
    r_train: MethodResult = fit(train)
    r_test: MethodResult = fit(test)
    s_train, s_test = int(np.sign(r_train.effect)), int(np.sign(r_test.effect))
    same_sign = s_train == s_test and s_test != 0 and (expected_sign is None or s_test == expected_sign)
    significant = bool(np.isfinite(r_test.p_value) and r_test.p_value < alpha)
    return {
        "holdout_year": int(holdout_year),
        "train_years": sorted(int(y) for y in years[years != holdout_year].unique()),
        "train": r_train.to_dict(),
        "test": r_test.to_dict(),
        "same_sign": bool(same_sign),
        "significant_in_holdout": significant,
        "replicated": bool(same_sign and significant),
    }


# ---------------------------------------------------------------- verdict

@dataclass
class GateInput:
    """What the verdict rule needs to know about one hypothesis."""
    effect: float
    standardized_effect: float
    q_value: float

    def holds(self, sign: int, alpha: float, floor: float) -> bool:
        return bool(np.isfinite(self.q_value) and self.q_value < alpha
                    and passes_effect_floor(self.standardized_effect, floor)
                    and int(np.sign(self.effect)) == sign)


def decide_verdict(h1: GateInput, h2: GateInput, *, expected_sign: int, alpha: float, floor: float,
                   replicated: bool) -> dict:
    """The four-verdict rule.

    reversed:    H1 holds with the opposite sign (e.g. higher ratings in high-need counties).
    unsupported: no H1 gap that is significant (after FDR) and above the effect floor.
    supported:   H1 holds, H2 holds with the same sign, and H2 replicates in the held-out year.
    weak:        H1 holds, but H2 shrinks below the floor, loses significance, flips, or
                 does not replicate.
    """
    if expected_sign not in (-1, 1):
        raise ValueError("expected_sign must be -1 or 1")
    reasons = []
    if h1.holds(-expected_sign, alpha, floor):
        verdict = "reversed"
        reasons.append("H1 gap is significant and above the floor, in the opposite direction")
    elif not h1.holds(expected_sign, alpha, floor):
        verdict = "unsupported"
        if not (np.isfinite(h1.q_value) and h1.q_value < alpha):
            reasons.append(f"H1 not significant after FDR (q = {h1.q_value:.4g})")
        if not passes_effect_floor(h1.standardized_effect, floor):
            reasons.append(f"H1 gap {h1.standardized_effect:+.3f} SD is below the {floor} SD floor")
    else:
        h2_holds = h2.holds(expected_sign, alpha, floor)
        if h2_holds and replicated:
            verdict = "supported"
            reasons.append("H1 and H2 hold and H2 replicates in the held-out year")
        else:
            verdict = "weak"
            reasons.append("H1 holds")
            if not h2_holds:
                reasons.append(f"H2 fails the gate (effect {h2.standardized_effect:+.3f} SD, "
                               f"q = {h2.q_value:.4g})")
            if not replicated:
                reasons.append("H2 does not replicate in the held-out year")
    return {"verdict": verdict, "reasons": reasons}
