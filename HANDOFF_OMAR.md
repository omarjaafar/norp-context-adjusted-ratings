# CP4 handoff: Vien → Omar (Sun 11/8)

All my items (V1–V6) are pushed. What's left is yours: rerun, paste, submit. Every step below has the exact command. Times are measured on my laptop (M-series Mac). **Delete this file in your last commit before submitting.**

## TL;DR

1. `git pull && pip install -r requirements.txt && python -m pytest -q` → **161 passed**
2. Rebuild the panel (CHR columns are in the need table now): ~5 min
3. **Rerun all 5 theories**: the gate statistic changed, so every earlier p-value is stale. ~20–25 min
4. `python scripts/review_theory_runs.py` → paste its 5 lines into my report (V6)
5. Rerun the sweep (main + ta_after only): ~35–40 min
6. README, assemble the report, push, submit

## What changed that affects you

| Change | Commit | What it means for you |
|---|---|---|
| **5 CHR columns** in the need table and panel: `chr_poor_fair_health`, `chr_uninsured`, `chr_some_college`, `chr_hs_completion`, `chr_violent_crime_rate` (plan names) | `1437cb9` | T2–T4 are runnable after a panel rebuild. `main_grid` picks them up in the sweep automatically. Descriptions are in `catalog/panel_columns.yaml`. |
| **Gate permutes t = slope / county jackknife SE** (`GATE_POLICY["permutation"]["studentized"] = True`), for H1 and H2 | `7ab7337` | The old raw-coefficient null was too narrow when the exposure sits on a few big counties (`docs/validator.md` 2.5). Example: the liabilities × housing-burden sweep cell goes from p 0.001 to 0.133. **T1, T5 and the whole sweep must be rerun.** Effects are unchanged; only p/q change. `permutation.p_value` is now the studentized p; the old raw-coefficient p is still reported under `permutation.coefficient`. |
| `apply_family` takes alpha from the reports, not a hard-coded 0.05 | `eccdeda` | No action. Every report from `run_spec` already has `validator.alpha`. |
| **I edited your `theorylab/trends.py`** (6 lines): the per-year H1/H2 permutation uses the same studentized statistic | `78ccd99` | So per-year results match the pooled gate. Tests pass. Revert it if you disagree, but then the per-year and pooled p's come from different statistics. |
| `scripts/review_theory_runs.py` | `78ccd99` | Checks that every verdict follows from its numbers, that the run used the current gate, and that the spec matches the gold answer. It produces the V6 lines. |

## Data caveats for the theory write-up (D8 in `docs/data_decisions.md`)

- **T4 (crime):** CHR froze violent crime. The same 2014/2016 UCR values appear in every release, and the measure is absent from 2023 on. It is a static county attribute: T4 compares counties, not years. 91.4% of county rows have it.
- **T2 (poor/fair health):** the level drops from tax year 2020 (median 0.201 → 0.153; county correlation 0.97). H2 has tax-year effects, so it is fine. The pooled H1 correlation is not protected; mention it if T2's H1 looks odd.
- `chr_hs_completion` is missing for 2018 (the measure starts in the 2021 release). No theory uses it.
- Connecticut planning regions get CHR values only in 2022 (health, uninsured); Puerto Rico has none.

## Steps

### 1. Pull, install, test

```bash
git pull
pip install -r requirements.txt
python -m pytest -q          # expect 161 passed
```

### 2. Rebuild the panel and do the hash check (V1, second round)

```bash
python scripts/build_panel.py                    # ~4.5 min; overwrites data/output/panel_join_log.json
python scripts/check_panel_hash.py --ref 78ccd99 \
    --out data/output/panel_rebuild_check_omar.json \
    --compare data/output/panel_rebuild_check.json
```

- My CHR panel has `sha256_csv` **`3dfa5eb19c30a65b1635b5fe460216ecbf0467a93715458ab0c8fac6be893bec`** (2,390,744 rows, 53 columns). `--ref 78ccd99` compares your build against it. `match.sha256_csv: true` closes F6.
- If it is `false`, look at `compared_with.columns_differing` in your output file. It names the columns that differ between our machines (the 5 `chr_*` columns show up under `only_here`, which is expected because my round-1 file predates them). Put that list in the report's F6 finding, replacing the sentence "This is still open".
- If `git status` shows `panel_join_log.json` changed only in `generated_at`/`size_mb`, don't commit it. If the hash changed, commit it with your check file.
- Commit `data/output/panel_rebuild_check_omar.json` (yours).

### 3. Rerun the five theories (O6)

```bash
export ANTHROPIC_API_KEY=...      # T2–T4 need live Planner/Scout calls; T1/T5 replay from the cache
python scripts/run_cp4.py         # 999 permutations + per-year; one cell-sized run is ~1 min on my machine
```

- All five should now be runnable (`benchmark.runnable` sees the `chr_*` columns in the join log).
- Then rerun A1/A2 so they cover all five theories: `python scripts/measure_assumptions_1_2.py`.
- Commit the new runs, `summary.json`, the new LLM cache files and `assumptions_1_2.json` (yours).

### 4. Review (V6), 1 min

```bash
python scripts/review_theory_runs.py
```

- It must end with `all verdicts follow: True; all on the current gate: True`. If not, the line says which theory and why.
- Paste the 5 printed lines into `project_one/CP4_REPORT_VIEN.md` in place of `FINAL_V6_LINES`.
- Commit `data/output/theory_runs/cp4/review.json`.

### 5. Rerun the sweep (O7)

```bash
python scripts/run_sweep.py --part main --part ta_after
```

- `main` is now 4 ratios × 7 measures (need_index, housing burden, 5 CHR) = **28 cells, 56 hypotheses**, at 999 permutations. One cell took **52 s** on my machine, so about 25 min in all. Your last main run took 17 min per cell; if your machine is that slow, it is 8 h. In that case use the drop-list option: run the TA's 8 cells (`MAIN_NEEDS` only) and say so.
- At 999 permutations a family must have fewer than 100 hypotheses for anything to be able to survive (`docs/validator.md` 2.6). 56 is fine.
- `ta_after` (40 cells, 199 permutations) uses the fixed gate, so it changes too: ~10–15 min. **`ta_before` uses the TA's own validator and does not change**, so don't rerun it. `summarize()` reads the existing file.
- Commit `data/output/sweep/*.json` and `data/output/sweep.json`.

### 6. README and report

- README (W15): the CP4 commands are steps 2–5 above, plus mine: `python scripts/h2_null_check.py`, `python scripts/run_controls.py --family`, `python scripts/drilldown_check.py`, `python scripts/build_county_need.py --chr-only --chr-dir data/raw/chr`.
- My report sections are in **`project_one/CP4_REPORT_VIEN.md`**: match items, evidence (path @ commit), findings, changes with reasons, V6, AI usage. Paste them in as they are, after filling `FINAL_V6_LINES` and, if step 2 tells you, the F6 sentence.
- **Changes table rows from me** (already in my file): the ownership split, drill-down narrowed to one level (F4), the studentized gate (F5), the family alpha (W11), the CHR columns, HS completion for 2018, the `trends.py` edit.
- **Environment for the report:** mine is macOS arm64, Python 3.12.7, pandas 3.1.0, numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0. Yours is in `panel_rebuild_check_omar.json` → `machine`.

### 7. Submit

```bash
git rm HANDOFF_OMAR.md
git commit -m "..." && git push origin main
```

(Only `origin` exists in my clone; push to `gatech` too if you have it.)

## Numbers you may want in the summary

- **H2 null/SE (F5):** 10 counties hold 13% of rows but 89–95% of the clustered variance. High-need counties cover 8% of rows, while random sets of the same size cover ~23%. That is why the raw null was narrow. Gate p: ratings H2 0.656, T5 0.514.
- **Family controls:** 0 of 8 placebo theories survive; the planted 0.25 SD theory stays supported (`controls_family.json`).
- **Drill-down:** finds Georgia only when planted in Georgia, and Education only when planted in Education. Neither one-stratum effect survives nationally, so neither would be drilled in a real run. The 30-county rule skips 15 of 51 states (`drilldown_check.json`).
