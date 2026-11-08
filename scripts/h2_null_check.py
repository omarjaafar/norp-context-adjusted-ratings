"""CP4 V3 (CP3 feedback F5): why H2's permutation null is narrower than its clustered SE.

In ratings_fairness.json, H2's permutation null has SD 0.029 while the
county-clustered SE of the same coefficient is 0.049. Under the null both measure
the spread of the coefficient, so one of them is off. This script measures the
candidate explanations on the real panel, for three H2 models:

  ratings      the CP3/CP4 ratings theory (binary high_need), where the gap was seen
  t5           unemployment -> liabilities/assets (continuous), where it was not
  sweep_cell   liabilities/assets x housing burden (continuous, state FE): the sweep
               cell with permutation q 0.003 but clustered p 0.11

For each model:
  1. Calibration. The clustered SE is recomputed inside every permutation draw.
     If the SE is right, its average over draws equals the SD of the permuted
     coefficients. The ratio of the SE at the observed exposure to the average
     permuted SE then says whether the real exposure sits on counties whose
     residuals are noisier or more alike than random counties' (heteroskedasticity
     or spatial clustering), which exchangeability ignores.
  2. Spatial correlation: null SD with histories shuffled only within a state,
     and the SE clustered by state.
  3. Heteroskedasticity: residual variance per row in exposed vs other counties,
     and each county's share of the clustered variance.
  4. Leverage: drop the 10 counties with the most rows, recompute null SD and SE.
  5. Residualizer: the permuted exposure is residualized on the same controls and
     absorbed effects as the fit (coefficient equality, orthogonality).
  6. Studentized permutation: the p-value of t = coefficient / SE against the
     permuted t's, with the CR1 SE and with the county jackknife SE the gate now
     uses (validator.ClusteredSlopeT), next to the plain coefficient permutation p.

Writes data/output/h2_null_check.json.

Usage:
    python scripts/h2_null_check.py [--n-permutations 499] [--cases ratings t5 sweep_cell]
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.executor import Residualizer, fixed_effects, to_float, to_jsonable  # noqa: E402
from theorylab.spec import lock_gate, spec_columns  # noqa: E402
from theorylab.sweep import RATING_SIGNS, cell_spec  # noqa: E402
from theorylab.theory_run import add_derived, load_panel, select_population  # noqa: E402
from theorylab.validator import ClusteredSlopeT, UnitPermuter  # noqa: E402

OUT = REPO_ROOT / "data" / "output" / "h2_null_check.json"


def case_specs() -> dict:
    ratings = json.loads((REPO_ROOT / "specs" / "ratings_fairness.json").read_text())
    t5 = json.loads((REPO_ROOT / "data" / "output" / "theory_runs" / "cp4"
                     / "unemployment_leverage_link.spec.json").read_text())
    cell = cell_spec("liabilities_to_assets", "avg_housing_burden",
                     RATING_SIGNS["liabilities_to_assets"], "fixed")
    return {"ratings": lock_gate(ratings)[0], "t5": lock_gate(t5)[0], "sweep_cell": cell}


def cluster_se(xr: np.ndarray, yr: np.ndarray, codes: np.ndarray, k: int) -> tuple[float, float]:
    """FWL slope and its CR1 cluster-robust SE (same small-sample factors as executor._ols_cov)."""
    sxx = float(xr @ xr)
    b = float(xr @ yr) / sxx
    e = yr - b * xr
    s = np.bincount(codes, weights=xr * e)
    g, n = int(codes.max()) + 1, len(yr)
    factor = g / (g - 1) * (n - 1) / (n - k)
    return b, float(np.sqrt(factor * (s @ s)) / sxx)


def prepare(panel: pd.DataFrame, spec: dict):
    df = add_derived(panel, spec_columns(spec))
    pop, _ = select_population(df, spec)
    exposure, outcome = spec["exposure"], spec["outcome"]
    if spec.get("exposure_type") == "continuous":
        xs = to_float(pop[exposure])
        pop = pop.copy()
        pop[exposure] = (xs - xs.mean()) / xs.std(ddof=1)
    h2 = next(h for h in spec["hypotheses"] if h["id"] == "H2")
    return pop, h2


def null_draws(permuter, resid, yr, county, state, k, n, seed, b_obs=None, se_obs=None,
               xr_obs=None) -> dict:
    """Permuted coefficients with their clustered SEs; p-values of the plain and studentized tests."""
    rng = np.random.default_rng(seed)
    jk = ClusteredSlopeT(yr, county, se="jackknife")
    b, se_c, se_s, se_j = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        xr = resid(permuter.draw(rng))
        b[i], se_c[i] = cluster_se(xr, yr, county, k)
        se_s[i] = cluster_se(xr, yr, state, k)[1]
        se_j[i] = jk.standard_error(xr, b[i])
    out = {}
    if b_obs is not None:
        jk_obs = jk.standard_error(xr_obs, b_obs)
        t_obs, t = b_obs / se_obs, b / se_c
        tj_obs, tj = b_obs / jk_obs, b / se_j
        out = {"p_permutation_coefficient": float((1 + np.sum(np.abs(b) >= abs(b_obs))) / (n + 1)),
               "p_permutation_studentized": float((1 + np.sum(np.abs(t) >= abs(t_obs))) / (n + 1)),
               "p_permutation_studentized_jackknife (gate)":
                   float((1 + np.sum(np.abs(tj) >= abs(tj_obs))) / (n + 1)),
               "null_sd_of_t": float(t.std(ddof=1)),
               "se_jackknife_observed": jk_obs,
               "mean_permuted_se_jackknife": float(se_j.mean())}
    return {**out, "n_permutations": n, "units_permuted": permuter.n_units,
            "null_sd": float(b.std(ddof=1)), "null_mean": float(b.mean()),
            "mean_permuted_se_county": float(se_c.mean()),
            "mean_permuted_se_state": float(se_s.mean()),
            "null_sd_over_mean_se_county": float(b.std(ddof=1) / se_c.mean()),
            "share_of_draws_with_county_p_below_0_05":
                float(np.mean(np.abs(b / se_c) > 1.959964))}


def run_case(name: str, spec: dict, panel: pd.DataFrame, n_perm: int, seed: int) -> dict:
    t0 = time.perf_counter()
    pop, h2 = prepare(panel, spec)
    outcome, exposure = spec["outcome"], spec["exposure"]
    unit, within = spec["exposure_unit"], spec.get("exposure_varies_by")
    controls, absorb = h2.get("controls", []), h2.get("absorb", [])
    fit = fixed_effects(pop, outcome, exposure, controls=controls, absorb=absorb, cluster=unit)
    resid = Residualizer.from_panel(pop, controls, absorb)
    y = to_float(pop[outcome]).to_numpy()
    x = to_float(pop[exposure]).to_numpy()
    yr, xr = resid(y), resid(x)
    county = pd.factorize(pop[unit], sort=True)[0]
    state = pd.factorize(pop["state"], sort=True)[0]
    k = 1 + len(controls) + sum(int(pd.factorize(pop[a])[0].max()) for a in absorb)
    b_obs, se_county = cluster_se(xr, yr, county, k)
    se_state = cluster_se(xr, yr, state, k)[1]

    # 5. Residualizer: same coefficient as the fit; a permuted x is residualized the same way
    rng = np.random.default_rng(seed)
    gate = UnitPermuter(pop, exposure, unit=unit, within=within, keep_paths=True)
    xp = resid(gate.draw(rng))
    C = np.column_stack([to_float(pop[c]).to_numpy() for c in controls]) if controls else None
    max_group_mean = max(float(pd.Series(xp).groupby(pop[a].to_numpy()).mean().abs().max())
                         for a in absorb) if absorb else 0.0
    residualizer = {
        "fwl_coefficient": b_obs, "fixed_effects_coefficient": fit.effect,
        "coefficients_equal": bool(np.isclose(b_obs, fit.effect, rtol=1e-8, atol=1e-12)),
        "cr1_se_here": se_county, "cr1_se_fixed_effects": fit.std_error,
        "se_equal": bool(np.isclose(se_county, fit.std_error, rtol=1e-6)),
        "permuted_x_max_abs_mean_within_absorbed_levels": max_group_mean,
        "permuted_x_max_abs_corr_with_controls":
            0.0 if C is None else float(max(abs(np.corrcoef(xp, c)[0, 1]) for c in C.T)),
    }

    # 1 + 2. Calibration, under the gate's shuffle and within state
    schemes = {
        "whole_county_histories (gate)": gate,
        "whole_county_histories_within_state": UnitPermuter(pop, exposure, unit=unit, within=within,
                                                            keep_paths=True, group="state"),
    }
    nulls = {label: null_draws(p, resid, yr, county, state, k, n_perm, seed + 1, b_obs, se_county, xr)
             for label, p in schemes.items()}

    # 3. Heteroskedasticity: per-row residual variance and share of the cluster variance
    e = yr - b_obs * xr
    s = np.bincount(county, weights=xr * e)
    rows = np.bincount(county)
    contrib = s ** 2 / (s ** 2).sum()
    unit_x = pd.Series(x).groupby(county).first().to_numpy()
    hetero = {"rows": int(len(pop)), "counties": int(len(rows)),
              "top10_counties_share_of_cluster_variance": float(np.sort(contrib)[-10:].sum()),
              "top10_counties_share_of_rows": float(np.sort(rows)[-10:].sum() / rows.sum())}
    if spec.get("exposure_type", "binary") == "binary":
        exposed_units = unit_x == 1
        exposed_rows = x == 1
        hetero.update({
            "residual_var_exposed_rows": float(e[exposed_rows].var()),
            "residual_var_other_rows": float(e[~exposed_rows].var()),
            "exposed_share_of_counties": float(exposed_units.mean()),
            "exposed_share_of_rows": float(exposed_rows.mean()),
            "exposed_counties_share_of_cluster_variance": float(contrib[exposed_units].sum()),
            # how much the county residual sums correlate within a county: ICC-like
            "mean_rows_per_county_exposed": float(rows[exposed_units].mean()),
            "mean_rows_per_county_other": float(rows[~exposed_units].mean()),
        })
    else:
        big = np.abs(unit_x) > 1
        hetero.update({"residual_var_rows_exposure_beyond_1sd": float(e[np.abs(x) > 1].var()),
                       "residual_var_rows_exposure_within_1sd": float(e[np.abs(x) <= 1].var()),
                       "counties_beyond_1sd_share_of_cluster_variance": float(contrib[big].sum())})

    # 4. Leverage: drop the 10 counties with the most rows
    top = pd.Series(rows, index=pd.factorize(pop[unit], sort=True)[1]).nlargest(10)
    sub = pop[~pop[unit].isin(top.index)].reset_index(drop=True)
    r_sub = Residualizer.from_panel(sub, controls, absorb)
    yr_sub = r_sub(to_float(sub[outcome]).to_numpy())
    c_sub = pd.factorize(sub[unit], sort=True)[0]
    s_sub = pd.factorize(sub["state"], sort=True)[0]
    xr_sub = r_sub(to_float(sub[exposure]).to_numpy())
    b_sub, se_sub = cluster_se(xr_sub, yr_sub, c_sub, k)
    p_sub = UnitPermuter(sub, exposure, unit=unit, within=within, keep_paths=True)
    leverage = {"dropped": top.index.tolist(), "rows_dropped": int(top.sum()),
                "coefficient": b_sub, "se_county": se_sub,
                **null_draws(p_sub, r_sub, yr_sub, c_sub, s_sub, k, max(199, n_perm // 2), seed + 2,
                             b_sub, se_sub, xr_sub)}

    gate_null = nulls["whole_county_histories (gate)"]
    return to_jsonable({
        "spec": spec["theory_id"], "outcome": outcome, "exposure": exposure,
        "exposure_type": spec.get("exposure_type", "binary"), "controls": controls,
        "absorb": absorb,
        "observed": {"coefficient": b_obs, "se_county": se_county, "se_state": se_state,
                     "p_cluster_county_normal": float(2 * stats.norm.sf(abs(b_obs / se_county))),
                     "null_sd_gate": gate_null["null_sd"],
                     "se_county_over_null_sd": se_county / gate_null["null_sd"],
                     "se_county_over_mean_permuted_se":
                         se_county / gate_null["mean_permuted_se_county"]},
        "nulls": nulls, "heteroskedasticity": hetero, "leverage": leverage,
        "residualizer": residualizer, "seconds": round(time.perf_counter() - t0, 1),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-permutations", type=int, default=499)
    ap.add_argument("--cases", nargs="+", default=["ratings", "t5", "sweep_cell"])
    ap.add_argument("--seed", type=int, default=6235)
    args = ap.parse_args()

    specs = {k: v for k, v in case_specs().items() if k in args.cases}
    cols = set().union(*(spec_columns(s) for s in specs.values())) | {"state"}
    panel = load_panel(columns=cols)
    out = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "question": "H2 permutation null SD 0.029 vs county-clustered SE 0.049 (ratings_fairness.json)",
           "cases": {}}
    for name, spec in specs.items():
        out["cases"][name] = run_case(name, spec, panel, args.n_permutations, args.seed)
        print(name, json.dumps(out["cases"][name]["observed"]), flush=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
