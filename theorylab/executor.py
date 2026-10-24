"""Executor method library (CP3): correlation, fixed effects, multilevel, lagged outcomes.

Every number here is computed by NumPy, SciPy or statsmodels. The LLM agents of
CP4 choose which method to run on which panel columns; they never compute a
statistic, and they never see anything but the MethodResult this module returns.

Conventions shared by every method:
- Inputs are the organization-year panel (theorylab/panel.py) or any frame with
  the same column names. Booleans read back from the panel CSV ("True"/"False")
  are converted by to_float().
- Rows with a missing outcome, exposure, control, absorbed factor or cluster are
  dropped, and n_obs reports how many rows were used.
- `effect` is the coefficient on the exposure (Pearson/Spearman r for
  correlation). `standardized_effect` divides it by the outcome's standard
  deviation in the estimation sample, so the Validator's effect-size floor is in
  outcome SD units for every method.
- Fixed effects are absorbed by alternating projections (demeaning), so a
  2.4M-row panel never builds a dummy matrix. Standard errors are clustered
  (CR1, the same small-sample factor statsmodels uses) when `cluster` is given.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning

METHODS = ("correlation", "fixed_effects", "multilevel", "two_stage_group_means", "lagged_outcome")


# ---------------------------------------------------------------- helpers

def to_float(s: pd.Series) -> pd.Series:
    """Numeric view of a panel column. Booleans and "True"/"False" text become 1.0/0.0."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("float64")
    text = s.astype("string").str.strip().str.lower()
    text = text.replace({"true": "1", "false": "0"})
    return pd.to_numeric(text, errors="coerce").astype("float64")


def to_jsonable(obj):
    """Plain Python for json.dumps: numpy scalars unwrapped, NaN/inf/NA become None."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


@dataclass
class MethodResult:
    method: str
    outcome: str
    exposure: str
    effect: float
    std_error: float
    p_value: float
    n_obs: int
    standardized_effect: float = float("nan")
    n_groups: int | None = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))


def _complete_rows(df: pd.DataFrame, numeric: list, keys: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(numeric columns as float64, key columns) over rows where all of them are present."""
    numeric = list(dict.fromkeys(numeric))
    keys = [k for k in dict.fromkeys(keys) if k]
    missing = [c for c in numeric + keys if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in the panel: {missing}")
    num = pd.DataFrame({c: to_float(df[c]) for c in numeric}, index=df.index)
    ok = np.isfinite(num.to_numpy()).all(axis=1)
    for k in keys:
        ok &= df[k].notna().to_numpy()
    if not ok.any():
        raise ValueError("no rows with every required column present")
    return num.loc[ok], df.loc[ok, keys]


def demean(values: np.ndarray, codes: list, tol: float = 1e-10, max_iter: int = 10_000):
    """Residuals of `values` on the indicator sets in `codes` (alternating projections).

    values: (n,) or (n, k) array. codes: one integer array (0..L-1) per absorbed
    factor. With one factor this is exact after one sweep; with several it
    repeats sweeps until no group mean left to remove exceeds tol (relative to
    the data's scale). With no factor, subtracts the column means (an intercept).
    Returns (residuals, sweeps).
    """
    X = np.array(values, dtype="float64", copy=True)
    squeeze = X.ndim == 1
    if squeeze:
        X = X[:, None]
    if not codes:
        X -= X.mean(axis=0)
        return (X[:, 0] if squeeze else X), 0
    counts = [np.bincount(c).astype("float64") for c in codes]
    scale = max(float(np.abs(X).max()) if X.size else 0.0, 1.0)
    for sweep in range(1, max_iter + 1):
        largest = 0.0
        for c, n in zip(codes, counts):
            for j in range(X.shape[1]):
                means = np.bincount(c, weights=X[:, j], minlength=len(n)) / n
                X[:, j] -= means[c]
                largest = max(largest, float(np.abs(means).max()))
        if len(codes) == 1 or largest < tol * scale:
            break
    else:
        raise RuntimeError(f"demeaning did not converge in {max_iter} sweeps")
    return (X[:, 0] if squeeze else X), sweep


def factor_codes(keys: pd.DataFrame, names) -> list:
    return [pd.factorize(keys[a], sort=True)[0] for a in names]


def _absorbed_parameters(codes: list) -> int:
    """Parameters the absorbed factors stand for, intercept included (no nesting assumed)."""
    if not codes:
        return 1
    return int(sum(int(c.max()) + 1 for c in codes) - (len(codes) - 1))


def _ols_cov(yd: np.ndarray, Xd: np.ndarray, k_absorbed: int, cluster_codes: np.ndarray | None):
    """OLS on already-demeaned data. Returns beta, covariance, dof, n_clusters."""
    n, k = Xd.shape
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ (Xd.T @ yd)
    resid = yd - Xd @ beta
    k_total = k + k_absorbed
    if cluster_codes is not None:
        G = int(cluster_codes.max()) + 1
        if G < 2:
            raise ValueError("clustered standard errors need at least two clusters")
        scores = np.column_stack([np.bincount(cluster_codes, weights=Xd[:, j] * resid, minlength=G)
                                  for j in range(k)])
        meat = scores.T @ scores
        factor = (G / (G - 1)) * ((n - 1) / (n - k_total))
        cov = factor * XtX_inv @ meat @ XtX_inv
        return beta, cov, G - 1, G
    dof = n - k_total
    if dof <= 0:
        raise ValueError("not enough rows for the number of parameters")
    sigma2 = float(resid @ resid) / dof
    return beta, sigma2 * XtX_inv, dof, None


def _t_pvalue(estimate: float, se: float, dof: float) -> float:
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    return float(2 * stats.t.sf(abs(estimate / se), dof))


# ---------------------------------------------------------------- methods

def correlation(df: pd.DataFrame, outcome: str, exposure: str, method: str = "pearson") -> MethodResult:
    """Pearson or Spearman correlation between exposure and outcome, with its p-value."""
    num, _ = _complete_rows(df, [outcome, exposure], [])
    x, y = num[exposure].to_numpy(), num[outcome].to_numpy()
    n = len(x)
    if n < 3:
        raise ValueError("correlation needs at least 3 rows")
    if method == "pearson":
        res = stats.pearsonr(x, y)
    elif method == "spearman":
        res = stats.spearmanr(x, y)
    else:
        raise ValueError(f"unknown correlation method {method!r}")
    r, p = float(res.statistic), float(res.pvalue)
    se = float(np.sqrt(max(1 - r * r, 0.0) / (n - 2)))
    return MethodResult("correlation", outcome, exposure, effect=r, std_error=se, p_value=p,
                        n_obs=n, standardized_effect=r, details={"kind": method})


def fixed_effects(df: pd.DataFrame, outcome: str, exposure: str, controls=(), absorb=(),
                  cluster: str | None = None, tol: float = 1e-10) -> MethodResult:
    """OLS of outcome on exposure + numeric controls, absorbing categorical fixed effects.

    absorb: columns whose levels get their own intercept (e.g. tax_year, category,
    state). cluster: column for clustered standard errors (e.g. county_fips, the
    level at which need is assigned). The coefficient is identical to OLS with a
    dummy per level; the standard errors match statsmodels' cluster covariance.
    """
    controls, absorb = list(controls), list(absorb)
    num, keys = _complete_rows(df, [outcome, exposure, *controls], [*absorb, cluster])
    codes = factor_codes(keys, absorb)
    regressors = [exposure, *controls]
    Z, sweeps = demean(num[[outcome, *regressors]].to_numpy(), codes, tol=tol)
    yd, Xd = Z[:, 0], Z[:, 1:]
    if float(Xd[:, 0] @ Xd[:, 0]) <= 1e-12 * max(len(Xd), 1):
        raise ValueError(f"{exposure} has no variation left after absorbing {absorb}")
    cluster_codes = pd.factorize(keys[cluster], sort=True)[0] if cluster else None
    beta, cov, dof, n_clusters = _ols_cov(yd, Xd, _absorbed_parameters(codes), cluster_codes)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    sd_y = float(num[outcome].std(ddof=1))
    return MethodResult(
        "fixed_effects", outcome, exposure,
        effect=float(beta[0]), std_error=float(se[0]), p_value=_t_pvalue(beta[0], se[0], dof),
        n_obs=int(len(num)), standardized_effect=float(beta[0]) / sd_y if sd_y > 0 else float("nan"),
        n_groups=n_clusters,
        details={
            "controls": {c: {"coef": float(b), "std_error": float(s), "p_value": _t_pvalue(b, s, dof)}
                         for c, b, s in zip(controls, beta[1:], se[1:])},
            "absorbed_levels": {a: int(c.max()) + 1 for a, c in zip(absorb, codes)},
            "demeaning_sweeps": int(sweeps),
            "covariance": "cluster (CR1)" if cluster else "classical",
            "cluster": cluster,
            "dof": int(dof),
            "outcome_sd": sd_y,
        })


MIXEDLM_OPTIMIZERS = ("lbfgs", "powell", "cg", "nm")


def _fit_mixedlm(model, reml: bool, method: str, maxiter: int) -> dict:
    """One MixedLM fit with one optimizer; never raises. valid = finite SE on x and no ConvergenceWarning."""
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = model.fit(reml=reml, method=[method], maxiter=maxiter)
        except Exception as exc:  # an optimizer that crashes is one failed attempt
            return {"method": method, "seconds": round(time.perf_counter() - start, 2), "res": None,
                    "valid": False, "warnings": [], "error": f"{type(exc).__name__}: {exc}"}
    msgs = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    se = float(res.bse["x"]) if "x" in res.bse.index else float("nan")
    valid = bool(getattr(res, "converged", False)) and np.isfinite(se) and se > 0 and not msgs
    return {"method": method, "seconds": round(time.perf_counter() - start, 2), "res": res,
            "valid": valid, "warnings": msgs, "error": None}


def multilevel(df: pd.DataFrame, outcome: str, exposure: str, controls=(), categorical=(),
               group: str = "county_fips", reml: bool = True, methods=MIXEDLM_OPTIMIZERS,
               maxiter: int = 300) -> MethodResult:
    """Random-intercept model (statsmodels MixedLM): outcome ~ exposure + controls + C(categorical), (1 | group).

    Optimizers are tried one at a time, in order, until one gives a valid fit.
    On the 2022 full-990 panel lbfgs stops on the boundary (county variance 0,
    singular Hessian) and returns an exposure coefficient of ~1e-13 with no SE,
    while powell, cg and nm agree with each other; statsmodels still flags the
    lbfgs fit as converged, so its own flag is not enough. If no optimizer is
    valid, the attempt with the highest log-likelihood and a finite SE is
    reported with converged = False.

    details records the fit time in seconds (all attempts), the optimizer used,
    every attempt, the random-intercept variance and the residual variance,
    which Assumption 5 is about.
    """
    controls, categorical = list(controls), list(categorical)
    num, keys = _complete_rows(df, [outcome, exposure, *controls], [group, *categorical])
    data = pd.DataFrame({"y": num[outcome].to_numpy(), "x": num[exposure].to_numpy()})
    terms = ["x"]
    for i, c in enumerate(controls):
        data[f"c{i}"] = num[c].to_numpy()
        terms.append(f"c{i}")
    for i, c in enumerate(categorical):
        data[f"f{i}"] = keys[c].astype(str).to_numpy()
        terms.append(f"C(f{i})")
    groups = keys[group].astype(str).to_numpy()
    model = smf.mixedlm("y ~ " + " + ".join(terms), data, groups=groups)
    attempts = []
    for method in methods:
        attempts.append(_fit_mixedlm(model, reml, method, maxiter))
        if attempts[-1]["valid"]:
            break
    chosen = next((a for a in attempts if a["valid"]), None)
    if chosen is None:
        usable = [a for a in attempts if a["res"] is not None and np.isfinite(float(a["res"].bse["x"]))]
        if not usable:
            raise RuntimeError("no MixedLM optimizer produced a usable fit: "
                               + "; ".join(f"{a['method']}: {a['error'] or a['warnings']}" for a in attempts))
        chosen = max(usable, key=lambda a: float(a["res"].llf))
    res = chosen["res"]
    seconds = sum(a["seconds"] for a in attempts)
    convergence_warnings = chosen["warnings"]
    sizes = pd.Series(groups).value_counts()
    sd_y = float(data["y"].std(ddof=1))
    effect, se, p = float(res.params["x"]), float(res.bse["x"]), float(res.pvalues["x"])
    return MethodResult(
        "multilevel", outcome, exposure, effect=effect, std_error=se, p_value=p,
        n_obs=int(len(data)), standardized_effect=effect / sd_y if sd_y > 0 else float("nan"),
        n_groups=int(len(sizes)),
        details={
            "fit_seconds": round(seconds, 2),
            "converged": bool(chosen["valid"]),
            "convergence_warnings": convergence_warnings,
            "optimizer": chosen["method"],
            "attempts": [{"method": a["method"], "seconds": a["seconds"], "valid": a["valid"],
                          "warnings": a["warnings"], "error": a["error"],
                          "effect": float(a["res"].params["x"]) if a["res"] is not None else None}
                         for a in attempts],
            "group": group,
            "random_intercept_variance": float(np.asarray(res.cov_re)[0, 0]),
            "residual_variance": float(res.scale),
            "group_size": {"median": float(sizes.median()), "min": int(sizes.min()),
                           "max": int(sizes.max()), "groups_with_one_row": int((sizes == 1).sum())},
            "controls": {c: float(res.params[f"c{i}"]) for i, c in enumerate(controls)},
            "categorical": categorical,
            "reml": reml,
            "optimizers_tried": [a["method"] for a in attempts],
            "outcome_sd": sd_y,
        })


def two_stage_group_means(df: pd.DataFrame, outcome: str, exposure: str, controls=(), absorb=(),
                          groups=("county_fips",)) -> MethodResult:
    """Assumption 5 fallback: partial out controls, then regress group means on group means.

    Stage 1 removes the absorbed fixed effects and numeric controls from both the
    outcome and the exposure (Frisch-Waugh-Lovell). Stage 2 is a weighted
    regression of each group's mean residual outcome on its mean residual
    exposure, weighted by group size, with HC1 standard errors. One row per
    group, so it stays fast however many organizations a county has.
    """
    controls, absorb, groups = list(controls), list(absorb), list(groups)
    num, keys = _complete_rows(df, [outcome, exposure, *controls], [*absorb, *groups])
    resid = Residualizer.from_frames(num, keys, controls, absorb)
    frame = keys[groups].copy()
    frame["_y"] = resid(num[outcome].to_numpy())
    frame["_x"] = resid(num[exposure].to_numpy())
    g = frame.groupby(groups, sort=True).agg(y=("_y", "mean"), x=("_x", "mean"), n=("_y", "size"))
    if len(g) < 3:
        raise ValueError("two-stage model needs at least 3 groups")
    fit = sm.WLS(g["y"].to_numpy(), sm.add_constant(g["x"].to_numpy()), weights=g["n"].to_numpy()).fit(
        cov_type="HC1")
    effect, se, p = float(fit.params[1]), float(fit.bse[1]), float(fit.pvalues[1])
    sd_y = float(num[outcome].std(ddof=1))
    return MethodResult("two_stage_group_means", outcome, exposure, effect=effect, std_error=se,
                        p_value=p, n_obs=int(len(num)),
                        standardized_effect=effect / sd_y if sd_y > 0 else float("nan"),
                        n_groups=int(len(g)),
                        details={"groups": groups, "controls": controls, "absorb": absorb,
                                 "covariance": "HC1", "outcome_sd": sd_y})


def lagged_outcome(df: pd.DataFrame, outcome: str, exposure: str, controls=(), entity: str = "ein",
                   time_col: str = "tax_year", lag: int = 1, include_lagged_outcome: bool = True,
                   absorb=("tax_year",), cluster: str | None = None) -> MethodResult:
    """Outcome in year t on exposure in year t - lag (and the outcome in t - lag).

    Each organization-year is matched to the same organization's filing exactly
    `lag` tax years earlier; rows without one are dropped (logged in details).
    The estimate is fixed_effects() on the matched rows, so the same absorbed
    factors and clustering apply.
    """
    controls = list(controls)
    cols = list(dict.fromkeys([entity, time_col, outcome, exposure, *controls, *absorb,
                               *([cluster] if cluster else [])]))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in the panel: {missing}")
    d = df[cols].copy()
    d[time_col] = to_float(d[time_col])
    x_lag, y_lag = f"{exposure}_lag{lag}", f"{outcome}_lag{lag}"
    prev = d[[entity, time_col, exposure, outcome]].rename(columns={exposure: x_lag, outcome: y_lag})
    prev[time_col] = prev[time_col] + lag
    matched = d.merge(prev, on=[entity, time_col], how="inner", validate="one_to_one")
    lag_controls = ([y_lag] if include_lagged_outcome else []) + controls
    result = fixed_effects(matched, outcome, x_lag, controls=lag_controls, absorb=absorb, cluster=cluster)
    result.method = "lagged_outcome"
    result.exposure = exposure
    result.details.update({"lag": lag, "entity": entity, "rows_in": int(len(d)),
                           "rows_with_lagged_filing": int(len(matched)),
                           "lagged_outcome_included": include_lagged_outcome})
    return result


# ---------------------------------------------------------------- partialling out

class Residualizer:
    """Removes absorbed fixed effects and numeric controls from any column (Frisch-Waugh-Lovell).

    Built once per population; calling it on a vector returns that vector's
    residual. The Validator uses it to re-estimate a fixed-effects coefficient
    under each permutation of the exposure without refitting the controls.
    """

    def __init__(self, codes: list, controls: np.ndarray | None, tol: float = 1e-10):
        self.codes = codes
        self.tol = tol
        if controls is not None and controls.size:
            self.C, _ = demean(controls, codes, tol=tol)
            self.CtC_inv = np.linalg.pinv(self.C.T @ self.C)
        else:
            self.C = None
            self.CtC_inv = None

    @classmethod
    def from_frames(cls, num: pd.DataFrame, keys: pd.DataFrame, controls, absorb, tol: float = 1e-10):
        C = num[list(controls)].to_numpy() if controls else None
        return cls(factor_codes(keys, absorb), C, tol=tol)

    @classmethod
    def from_panel(cls, df: pd.DataFrame, controls=(), absorb=(), tol: float = 1e-10):
        """For a frame that is already complete in controls and absorb (the Validator's population)."""
        controls, absorb = list(controls), list(absorb)
        C = np.column_stack([to_float(df[c]).to_numpy() for c in controls]) if controls else None
        if C is not None and not np.isfinite(C).all():
            raise ValueError("controls have missing values; select the population first")
        if any(df[a].isna().any() for a in absorb):
            raise ValueError("absorbed factors have missing values; select the population first")
        return cls([pd.factorize(df[a], sort=True)[0] for a in absorb], C, tol=tol)

    def __call__(self, v) -> np.ndarray:
        r, _ = demean(np.asarray(v, dtype="float64"), self.codes, tol=self.tol)
        if self.C is not None:
            r = r - self.C @ (self.CtC_inv @ (self.C.T @ r))
        return r


def fwl_coefficient(y_resid: np.ndarray, x_resid: np.ndarray) -> float:
    """Slope of y on x after both have been residualized on the same controls."""
    denom = float(x_resid @ x_resid)
    return float(x_resid @ y_resid) / denom if denom > 0 else float("nan")
