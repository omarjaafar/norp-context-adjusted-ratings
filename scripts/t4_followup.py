"""CP5: is T4's +0.03 SD (violent crime -> admin share) a small real effect or a proxy?

Reruns the T4 spec under the locked gate with extra controls in H2, one at a time, on the
same population: poverty rate, median income, county population (a size proxy), the ACS
need index, and all four together. A proxy would shrink toward zero when its stand-in is
controlled; a real effect keeps its size. Also reports the CHR 2014/2016 crime snapshot
against a second crime measure when one is present in the panel (columns named
`crime_*` other than chr_violent_crime_rate); with none, it says so and stops there.

The second source has to be acquired by hand (it is not in the repo): add it to
catalog/catalog.yaml and the need table, then rerun. Writes data/output/t4_followup.json.
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402
from theorylab.theory_run import PANEL_PATH, available_columns, load_panel, run_spec  # noqa: E402
from theorylab.spec import spec_columns  # noqa: E402

SPEC = REPO_ROOT / "data" / "output" / "theory_runs" / "cp4" / "violence_admin_overhead_share.spec.json"
EXTRA = {"poverty_rate": ["poverty_rate"], "median_income": ["median_household_income"],
         "population": ["population"], "need_index": ["need_index"],
         "all_four": ["poverty_rate", "median_household_income", "population", "need_index"]}


def main() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    second = sorted(c for c in available_columns() if c.startswith("crime_") and c != "chr_violent_crime_rate")
    variants = {"as_run": spec}
    for name, cols in EXTRA.items():
        s = copy.deepcopy(spec)
        h2 = next(h for h in s["hypotheses"] if h["id"] == "H2")
        h2["controls"] = sorted(set(h2.get("controls", [])) | set(cols))
        s["theory_id"] = f"{spec['theory_id']}_plus_{name}"
        variants[name] = s
    cols = set().union(*(spec_columns(s) for s in variants.values())) | {"ein"}
    panel = load_panel(PANEL_PATH, columns=sorted(cols))
    out = {"second_crime_columns_in_panel": second, "variants": {}}
    for name, s in variants.items():
        r = run_spec(panel, s, include_robustness=False)
        out["variants"][name] = {"verdict_within_theory": r["verdict"],
                                 "H1_std": r["H1"]["standardized_effect"],
                                 "H2_std": r["H2"]["fit"]["standardized_effect"],
                                 "H2_p": r["H2"]["permutation"]["p_value"],
                                 "folds_with_expected_sign": r["replication"].get("folds_with_expected_sign")}
    base = out["variants"]["as_run"]["H2_std"]
    for v in out["variants"].values():
        v["share_of_as_run_effect"] = v["H2_std"] / base if base else None
    (REPO_ROOT / "data" / "output" / "t4_followup.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if not second:
        print("no second crime measure in the panel: the source check still needs the data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
