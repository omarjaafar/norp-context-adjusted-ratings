"""Validator (CP3): the deterministic gate every finding must pass.

Pieces, each usable on its own:
- UnitPermuter + permutation_test: shuffle the exposure across the units it is
  assigned to (counties, within each tax year), re-estimate, and count how often
  a shuffled estimate is at least as large as the real one. Organizations in one
  county share that county's need, so shuffling organizations one by one would
  understate the noise; shuffling whole counties does not. With keep_paths, each
  county takes another county's whole history, because need persists over years
  and a year-by-year shuffle would understate the noise the same way.
- benjamini_hochberg: false discovery rate control over a family of p-values.
- passes_effect_floor: a significant gap must also be large enough to matter,
  measured in outcome standard deviations.
- held_out_group_replication: split the states into folds, fit on the other
  states, re-estimate on each held-out fold, and require the expected sign in
  every fold and a significant pooled held-out estimate. This is the replication
  that feeds the verdict.
- held_out_year_replication: fit on the other years, re-estimate on the held-out
  year. Reported for information only: need barely moves between years, so a
  held-out year re-tests the same organizations in the same counties.
- decide_verdict: supported / weak / unsupported / reversed, from H1 (the gap
  exists) and H2 (the gap survives the controls), as docs/ratings_theory.md
  defines them.

Nothing here asks an LLM anything, and nothing an LLM writes can change a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from theorylab.executor import MethodResult, to_float

VERDICTS = ("supported", "weak", "unsupported", "reversed")


# ---------------------------------------------------------------- permutation

class UnitPermuter:
    """Draws permutations of one column, shuffled across units within strata.

    unit=None shuffles rows. With unit="county_fips" and within="tax_year", each
    county-year keeps one exposure value, and the values are shuffled among the
    counties of the same tax year, so every draw has the same number of exposed
    counties per year as the data. Raises if the column varies inside a unit.

    keep_paths=True (needs unit and within) shuffles whole histories instead: each
    county takes another county's exposure in every year, among counties present
    in the same years. The exposed count per year is still exact, and an exposure
    that persists over years (most counties are high-need in all years or none)
    stays persistent in every draw. Shuffling each year separately breaks that
    persistence, and when organizations and counties carry over between years
    the resulting null is too narrow (see docs/validator.md).
    """

    def __init__(self, df: pd.DataFrame, column: str, unit: str | None = None,
                 within: str | None = None, keep_paths: bool = False):
        values = to_float(df[column]).to_numpy()
        if np.isnan(values).any():
            raise ValueError(f"{column} has missing values; select the population first")
        for key in (unit, within):
            if key is not None and df[key].isna().any():
                raise ValueError(f"{key} has missing values; select the population first")
        if keep_paths and (unit is None or within is None):
            raise ValueError("keep_paths needs both unit and within")
        self.column = column
        self.keep_paths = bool(keep_paths)
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
        if self.keep_paths:
            self._init_paths(agg.index)

    def _init_paths(self, index: pd.MultiIndex) -> None:
        """Lookup tables for drawing whole unit histories (keep_paths=True)."""
        w = pd.factorize(index.get_level_values(0), sort=True)[0]
        u, units = pd.factorize(index.get_level_values(1), sort=True)
        pos = np.full((len(units), int(w.max()) + 1), -1, dtype="int64")
        pos[u, w] = np.arange(len(w))
        patterns = pd.factorize(pd.Series([row.tobytes() for row in pos >= 0]))[0]
        self._u, self._w, self._pos = u, w, pos
        self._unit_strata = [np.flatnonzero(patterns == p) for p in np.unique(patterns)]
        self.n_units = int(len(units))

    def observed(self) -> np.ndarray:
        return self._values.copy()

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        if self.keep_paths:
            source = np.arange(self._pos.shape[0])
            for units in self._unit_strata:
                source[units] = rng.permutation(units)
            return self._level[self._pos[source[self._u], self._w]][self._codes]
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


def group_folds(groups: pd.Series, n_folds: int, seed: int) -> list:
    """Assign the distinct values of `groups` to n_folds folds at random.

    The values are sorted, shuffled with the seed, and dealt into folds of near-equal
    size (in groups, not rows), so a seed always gives the same folds.
    """
    values = np.array(sorted(groups.dropna().unique()), dtype=object)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(values) < n_folds:
        raise ValueError(f"{len(values)} groups cannot be split into {n_folds} folds")
    shuffled = values[np.random.default_rng(seed).permutation(len(values))]
    return [sorted(fold.tolist()) for fold in np.array_split(shuffled, n_folds)]


def held_out_group_replication(df: pd.DataFrame, fit, *, group: str = "state", n_folds: int = 5,
                               seed: int = 0, alpha: float = 0.05,
                               expected_sign: int | None = None) -> dict:
    """K-fold replication over groups (states): fit on the other groups, test on each fold.

    fit: function frame -> MethodResult. Replicated means
      1. in every fold the held-out estimate has the sign of its training fit (and the
         expected sign, when given), and
      2. the inverse-variance-pooled held-out estimate has p < alpha.
    A gap carried by a few states fails (1) in the folds without them; a gap that is
    the right sign everywhere but too noisy to tell from zero fails (2). Folds share no
    states, so no counties, which makes the held-out estimates independent and the
    pooling valid. Requiring significance in every fold would ask each fold to have
    the power of the whole sample; that is not the test. Cochran's Q across folds is
    reported for information and does not change `replicated`.
    """
    if df[group].isna().any():
        raise ValueError(f"{group} has missing values; select the population first")
    folds = group_folds(df[group], n_folds, seed)
    rows = []
    for i, held_out in enumerate(folds):
        in_test = df[group].isin(held_out)
        r_train: MethodResult = fit(df[~in_test])
        r_test: MethodResult = fit(df[in_test])
        s_train, s_test = int(np.sign(r_train.effect)), int(np.sign(r_test.effect))
        rows.append({
            "fold": i,
            "held_out": held_out,
            "train": r_train.to_dict(),
            "test": r_test.to_dict(),
            "same_sign": bool(s_test != 0 and s_test == s_train
                              and (expected_sign is None or s_test == expected_sign)),
        })

    effects = np.array([r["test"]["effect"] for r in rows], dtype="float64")
    ses = np.array([r["test"]["std_error"] for r in rows], dtype="float64")
    usable = np.isfinite(effects) & np.isfinite(ses) & (ses > 0)
    if usable.all():
        w = 1.0 / ses ** 2
        pooled = float((w * effects).sum() / w.sum())
        pooled_se = float(1.0 / np.sqrt(w.sum()))
        pooled_p = float(2 * stats.norm.sf(abs(pooled / pooled_se)))
        q_stat = float((w * (effects - pooled) ** 2).sum())
        q_p = float(stats.chi2.sf(q_stat, n_folds - 1))
        i2 = float(max(0.0, (q_stat - (n_folds - 1)) / q_stat)) if q_stat > 0 else 0.0
    else:  # a fold without a usable estimate cannot count as replicated
        pooled = pooled_se = pooled_p = q_stat = q_p = i2 = float("nan")
    sign_ok = all(r["same_sign"] for r in rows)
    pooled_sign_ok = expected_sign is None or int(np.sign(pooled)) == expected_sign
    significant = bool(np.isfinite(pooled_p) and pooled_p < alpha and pooled_sign_ok)
    return {
        "by": group,
        "n_folds": int(n_folds),
        "seed": int(seed),
        "rule": "expected sign in every held-out fold and inverse-variance-pooled held-out p < alpha",
        "folds": rows,
        "folds_with_expected_sign": int(sum(r["same_sign"] for r in rows)),
        "pooled_effect": pooled,
        "pooled_std_error": pooled_se,
        "pooled_p_value": pooled_p,
        "heterogeneity": {"cochran_q": q_stat, "p_value": q_p, "i_squared": i2},
        "same_sign_in_every_fold": bool(sign_ok),
        "significant_pooled": significant,
        "replicated": bool(sign_ok and significant),
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
    supported:   H1 holds, H2 holds with the same sign, and H2 replicates in the held-out states.
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
            reasons.append("H1 and H2 hold and H2 replicates in the held-out states")
        else:
            verdict = "weak"
            reasons.append("H1 holds")
            if not h2_holds:
                reasons.append(f"H2 fails the gate (effect {h2.standardized_effect:+.3f} SD, "
                               f"q = {h2.q_value:.4g})")
            if not replicated:
                reasons.append("H2 does not replicate in the held-out states")
    return {"verdict": verdict, "reasons": reasons}
