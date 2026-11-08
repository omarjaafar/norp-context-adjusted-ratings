"""CP4: how much do extreme ratios drive the SD behind each theory's effect floor?

The gate's floor is 0.1 SD of the outcome. For a ratio with a tiny denominator a few
organizations can set that SD, and then the floor and H1's Pearson r say little. For each
CP4 theory, on its own population: SD against a robust SD (IQR / 1.349), the share of the
variance held by the top 0.1% of rows, and Pearson against Spearman.

Descriptive only: it changes no verdict. Spearman is row-level with no permutation, so
read its size, not its significance. Writes data/output/outcome_tails.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.spec import spec_columns  # noqa: E402
from theorylab.theory_run import add_derived, load_panel, select_population  # noqa: E402

RUN_DIR = REPO_ROOT / "data" / "output" / "theory_runs" / "cp4"


def tails(pop: pd.DataFrame, outcome: str, exposure: str) -> dict:
    y, x = pop[outcome].astype(float), pop[exposure].astype(float)
    q = y.quantile([0.001, 0.25, 0.5, 0.75, 0.999])
    sd, robust = float(y.std()), float((q[0.75] - q[0.25]) / 1.349)
    dev2 = (y - y.mean()) ** 2
    return {
        "rows": int(len(y)), "sd": sd, "robust_sd": robust, "sd_over_robust_sd": sd / robust,
        "floor_in_outcome_units": 0.1 * sd,
        "p0.1": float(q[0.001]), "median": float(q[0.5]), "p99.9": float(q[0.999]),
        "min": float(y.min()), "max": float(y.max()),
        "variance_share_top_0.1pct_rows": float(np.sort(dev2.to_numpy())[-max(1, len(y) // 1000):].sum()
                                                / dev2.sum()),
        "pearson": float(x.corr(y)), "spearman": float(x.rank().corr(y.rank())),
    }


def main() -> int:
    specs = {}
    for path in sorted(RUN_DIR.glob("*.spec.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        specs[spec["theory_id"]] = spec
    cols = set().union(*(spec_columns(s) for s in specs.values()))
    panel = load_panel(columns=sorted(cols))
    out = {}
    for tid, spec in specs.items():
        pop, _ = select_population(add_derived(panel, spec_columns(spec)), spec)
        out[tid] = {"outcome": spec["outcome"], "exposure": spec["exposure"],
                    **tails(pop, spec["outcome"], spec["exposure"])}
    path = REPO_ROOT / "data" / "output" / "outcome_tails.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
