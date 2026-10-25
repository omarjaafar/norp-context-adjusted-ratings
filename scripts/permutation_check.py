"""Checkpoint 3: the numbers behind docs/validator.md section 2 (permutation review, U7).

    python scripts/permutation_check.py                  # 199 permutations

On the ratings theory's population:
- how persistent high_need is over tax years (counties present in all five years),
- the share of high_need's variance the H2 controls explain (whether permute-X needs
  Freedman-Lane),
- H1 and H2 permutation nulls with the year-by-year shuffle (TA code) and with whole
  county histories (keep_paths), next to H2's county-clustered SE.

Needs data/derived/org_year_panel.csv.gz (python scripts/build_panel.py).
Writes data/output/permutation_check.json.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.controls import population  # noqa: E402
from theorylab.executor import Residualizer, fixed_effects, fwl_coefficient, to_float  # noqa: E402
from theorylab.spec import load_spec, spec_columns  # noqa: E402
from theorylab.theory_run import PANEL_PATH, available_columns, load_panel, write_report  # noqa: E402
from theorylab.validator import UnitPermuter, difference_in_means, permutation_test  # noqa: E402

OUT = REPO_ROOT / "data" / "output" / "permutation_check.json"
NEED_TABLE = REPO_ROOT / "data" / "derived" / "county_need.csv"


def persistence(need_table: Path = NEED_TABLE) -> dict:
    need = pd.read_csv(need_table, dtype={"county_fips": str})
    wide = need.pivot(index="county_fips", columns="acs_year", values="high_need")
    full = wide.dropna().astype(int)
    years_high = full.sum(axis=1)
    n_years = full.shape[1]
    return {
        "source": "data/derived/county_need.csv",
        "counties_in_all_years": int(len(full)),
        "years": [int(y) for y in full.columns],
        "counties_by_years_high_need": {str(k): int(v) for k, v in
                                        years_high.value_counts().sort_index().items()},
        "share_never_switch": float(((years_high == 0) | (years_high == n_years)).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO_ROOT / "specs" / "ratings_fairness.json"))
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--n-permutations", type=int, default=199)
    args = ap.parse_args()

    spec = load_spec(args.spec, available_columns())
    pop, pop_log = population(load_panel(args.panel, columns=spec_columns(spec) | {"ein"}), spec)
    outcome, exposure = spec["outcome"], spec["exposure"]
    unit, within = spec["exposure_unit"], spec.get("exposure_varies_by")
    h2 = next(h for h in spec["hypotheses"] if h["id"] == "H2")
    controls, absorb = h2.get("controls", []), h2.get("absorb", [])
    seed = int(spec["validator"]["seed"])

    y = to_float(pop[outcome]).to_numpy()
    x = to_float(pop[exposure]).to_numpy()
    resid = Residualizer.from_panel(pop, controls, absorb)
    y_resid, x_resid = resid(y), resid(x)
    x_centered = x - x.mean()
    r2 = 1.0 - float(x_resid @ x_resid) / float(x_centered @ x_centered)

    schemes = {}
    for name, keep_paths in (("year_by_year", False), ("whole_history", True)):
        permuter = UnitPermuter(pop, exposure, unit=unit, within=within, keep_paths=keep_paths)
        schemes[name] = {
            "keep_paths": keep_paths,
            "units_permuted": permuter.n_units,
            "H1": permutation_test(lambda v: difference_in_means(y, v), permuter,
                                   args.n_permutations, seed),
            "H2": permutation_test(lambda v: fwl_coefficient(y_resid, resid(v)), permuter,
                                   args.n_permutations, seed + 1),
        }
    fit = fixed_effects(pop, outcome, exposure, controls=controls, absorb=absorb,
                        cluster=h2.get("cluster"))
    ratio = schemes["whole_history"]["H1"]["null_sd"] / schemes["year_by_year"]["H1"]["null_sd"]

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "theory_id": spec["theory_id"],
        "n_permutations": args.n_permutations,
        "population": pop_log,
        "persistence": persistence(),
        "exposure_r2_on_h2_controls": r2,
        "schemes": schemes,
        "h1_null_sd_ratio_history_to_yearly": ratio,
        "H2_clustered": {"effect": fit.effect, "std_error": fit.std_error, "p_value": fit.p_value},
    }
    write_report(report, OUT)
    p = report["persistence"]
    print(f"counties never switching high_need: {p['share_never_switch']:.1%} "
          f"{p['counties_by_years_high_need']}")
    print(f"R2 of {exposure} on the H2 controls: {r2:.4f}")
    for name, s in schemes.items():
        print(f"{name:13s} H1 null SD {s['H1']['null_sd']:.4f} p {s['H1']['p_value']:.3f} | "
              f"H2 null SD {s['H2']['null_sd']:.4f} p {s['H2']['p_value']:.3f}")
    print(f"H2 clustered SE {fit.std_error:.4f} p {fit.p_value:.3f}; H1 null SD ratio {ratio:.2f}")
    print(f"wrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
