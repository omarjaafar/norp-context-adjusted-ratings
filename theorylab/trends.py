"""H1 and H2 for each tax year, and a test of whether the gap shifts over time (CP2 feedback).

    population (theory_run.select_population, the same rows as the pooled run)
      -> continuous exposure: standardized once on the pooled population, as in run_spec,
         so every year's coefficient and the trend are per pooled SD of the exposure
      -> for each tax year:
           H1  difference in means (binary) or Pearson r (continuous), counties shuffled
           H2  fixed effects (spec controls; tax_year dropped from absorb), clustered by
               county, permutation on the same coefficient
         Benjamini-Hochberg over the five H1 and, separately, the five H2 p-values
      -> trend: fixed effects of the outcome on high_need and high_need x (tax_year - center),
         spec controls and absorbed factors, SEs clustered by county
      -> cross-check: inverse-variance weighted regression of the five H2 estimates on year

The trend p-value is the clustered-SE t-test, not a permutation. Shuffling the
exposure across counties tests "no gap at all"; it is not a valid null for "the
gap is the same every year", so it would answer the wrong question.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm

from theorylab.executor import Residualizer, fixed_effects, to_float, to_jsonable
from theorylab.spec import spec_columns
from theorylab.theory_run import add_derived, select_population
from theorylab.validator import (GateInput, UnitPermuter, benjamini_hochberg, difference_in_means,
                                 gate_statistic, pearson_r, permutation_test)


def _continuous(spec: dict) -> bool:
    return spec.get("exposure_type", "binary") == "continuous"


def _year_tests(pop: pd.DataFrame, spec: dict, n_perm: int, seed: int) -> dict:
    outcome, exposure, unit = spec["outcome"], spec["exposure"], spec["exposure_unit"]
    h2 = next(h for h in spec["hypotheses"] if h["id"] == "H2")
    controls = list(h2.get("controls", []))
    absorb = [a for a in h2.get("absorb", []) if a != "tax_year"]
    cluster = h2.get("cluster")
    y = to_float(pop[outcome]).to_numpy()
    x = to_float(pop[exposure]).to_numpy()
    sd_y = float(np.std(y, ddof=1))
    permuter = UnitPermuter(pop, exposure, unit=unit)

    continuous = _continuous(spec)
    h1_stat = (lambda xv: pearson_r(y, xv)) if continuous else (lambda xv: difference_in_means(y, xv))
    h1 = h1_stat(x)
    # same statistic as the pooled gate: studentized slope when the policy says so (V3)
    studentized = bool((spec["validator"].get("permutation") or {}).get("studentized", False))
    unit_codes = pd.factorize(pop[unit], sort=True)[0]
    if studentized:
        h1_stat = gate_statistic(y - y.mean(), None, unit_codes, True)
    h1_perm = permutation_test(h1_stat, permuter, n_perm, seed)
    fit = fixed_effects(pop, outcome, exposure, controls=controls, absorb=absorb, cluster=cluster)
    resid = Residualizer.from_panel(pop, controls, absorb)
    y_resid = resid(y)
    clusters = pd.factorize(pop[cluster], sort=True)[0] if cluster else None
    h2_perm = permutation_test(gate_statistic(y_resid, resid, clusters, studentized),
                               permuter, n_perm, seed + 1)
    if continuous:  # a correlation is already standardized
        h1_out = {"method": "correlation", "effect": h1, "standardized_effect": h1}
    else:
        h1_out = {"method": "difference_in_means", "effect": h1,
                  "standardized_effect": h1 / sd_y if sd_y > 0 else float("nan"),
                  "mean_exposed": float(y[x == 1].mean()), "mean_unexposed": float(y[x == 0].mean())}
    return {
        "rows": int(len(pop)),
        "units": int(pop[unit].nunique()),
        "exposed_units": None if continuous else int(pop.loc[x == 1, unit].nunique()),
        "outcome_sd": sd_y,
        "H1": {**h1_out, "p_value": h1_perm["p_value"]},
        "H2": {"effect": fit.effect, "standardized_effect": fit.standardized_effect,
               "std_error": fit.std_error, "clustered_p_value": fit.p_value,
               "p_value": h2_perm["p_value"], "absorb": absorb},
    }


def gap_trend(pop: pd.DataFrame, spec: dict, center: float | None = None) -> dict:
    """Fixed effects with exposure x (tax_year - center); the interaction is the change in gap per year."""
    outcome, exposure = spec["outcome"], spec["exposure"]
    h2 = next(h for h in spec["hypotheses"] if h["id"] == "H2")
    years = to_float(pop["tax_year"])
    center = float(np.mean(np.unique(years))) if center is None else float(center)
    frame = pop.copy()
    frame["_exposure_x_year"] = to_float(pop[exposure]) * (years - center)
    fit = fixed_effects(frame, outcome, "_exposure_x_year",
                        controls=[exposure, *h2.get("controls", [])],
                        absorb=h2.get("absorb", []), cluster=h2.get("cluster"))
    gap_at_center = fit.details["controls"][exposure]
    return {
        "model": f"{outcome} ~ {exposure} + {exposure} x (tax_year - {center:g}) + controls | absorb",
        "center_year": center,
        "gap_change_per_year": fit.effect,
        "gap_change_per_year_sd": fit.standardized_effect,
        "std_error": fit.std_error,
        "p_value": fit.p_value,
        "gap_at_center_year": gap_at_center["coef"],
        "gap_at_center_year_p_value": gap_at_center["p_value"],
        "n_obs": fit.n_obs,
        "clusters": fit.n_groups,
        "covariance": fit.details["covariance"],
    }


def meta_trend(by_year: dict) -> dict:
    """Weighted least squares of the yearly H2 estimates on year, weights 1 / SE^2."""
    years = np.array([float(y) for y in by_year])
    est = np.array([r["H2"]["effect"] for r in by_year.values()], dtype="float64")
    se = np.array([r["H2"]["std_error"] for r in by_year.values()], dtype="float64")
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    if ok.sum() < 3:
        return {"error": "fewer than 3 years with a finite H2 estimate"}
    fit = sm.WLS(est[ok], sm.add_constant(years[ok] - years[ok].mean()), weights=1 / se[ok] ** 2).fit()
    return {"slope_per_year": float(fit.params[1]), "std_error": float(fit.bse[1]),
            "p_value": float(fit.pvalues[1]), "years": int(ok.sum())}


def run_by_year(panel: pd.DataFrame, spec: dict, *, n_permutations: int | None = None) -> dict:
    """Per-year H1/H2 with the spec's gate settings, plus the gap trend. Returns a JSON-ready report."""
    v = spec["validator"]
    alpha, floor = float(v["alpha"]), float(v["effect_floor_sd"])
    n_perm = int(n_permutations or v["n_permutations"])
    seed, sign = int(v["seed"]), int(spec["expected_sign"])
    df = add_derived(panel, spec_columns(spec))
    pop, pop_log = select_population(df, spec)
    if _continuous(spec):
        xs = to_float(pop[spec["exposure"]])
        pop = pop.copy()
        pop[spec["exposure"]] = (xs - xs.mean()) / xs.std(ddof=1)
    years = to_float(pop["tax_year"])

    by_year = {}
    for i, year in enumerate(sorted(years.unique())):
        part = pop[years == year].reset_index(drop=True)
        by_year[str(int(year))] = _year_tests(part, spec, n_perm, seed + 10 * i)

    for h in ("H1", "H2"):
        q, _ = benjamini_hochberg([r[h]["p_value"] for r in by_year.values()], alpha)
        for r, qv in zip(by_year.values(), q):
            r[h]["q_value"] = float(qv)
            r[h]["holds"] = GateInput(r[h]["effect"], r[h]["standardized_effect"], float(qv)).holds(
                sign, alpha, floor)

    return to_jsonable({
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "theory_id": spec["theory_id"],
        "exposure_type": "continuous" if _continuous(spec) else "binary",
        "settings": {"alpha": alpha, "effect_floor_sd": floor, "n_permutations": n_perm, "seed": seed,
                     "expected_sign": sign, "permutation_unit": spec["exposure_unit"],
                     "fdr_families": ["H1 over years", "H2 over years"],
                     "trend_inference": "county-clustered SE (CR1); see module docstring"},
        "population": pop_log,
        "by_year": by_year,
        "years_H1_holds": [y for y, r in by_year.items() if r["H1"]["holds"]],
        "years_H2_holds": [y for y, r in by_year.items() if r["H2"]["holds"]],
        "trend": gap_trend(pop, spec),
        "trend_meta_regression": meta_trend(by_year),
    })
