# CP4 report sections: Vien Tran (county data, gate statistics, drill-down)

Commits: `37014de` (V1), `1437cb9` (V2), `7ab7337` (V3), `eccdeda` (V4), `72306a7` (V5), `78ccd99` (V6 review tool; per-year statistic). All on `main`, pushed to origin.
Environment: macOS 26.6 arm64, Python 3.12.7, pandas 3.1.0, numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0. 161 tests pass (136 before).

## Match items (CP3 feedback → what was done)

| # | Feedback | Done | Evidence |
|---|---|---|---|
| F1 | Health, education and crime county indicators, joined to the panel | Five County Health Rankings columns (`chr_poor_fair_health`, `chr_uninsured`, `chr_some_college`, `chr_hs_completion`, `chr_violent_crime_rate`) in the need table and the panel. Vintage rule D8. | `data/output/county_need_report.json` → `chr` @ `1437cb9`; `docs/data_decisions.md` D8 |
| F4 | Log drill-down narrowed to one level | Changes row below; drill-down tested on planted strata | `data/output/drilldown_check.json` @ `72306a7` |
| F5 | H2 permutation null (SD 0.028) narrower than clustered SE (0.049) | Measured the cause; the gate now permutes the studentized slope (county jackknife SE) | `data/output/h2_null_check.json` @ `7ab7337`; `docs/validator.md` 2.5 |
| F6 | Rebuild the panel on a second machine, commit the hash check | `scripts/check_panel_hash.py`; rebuilt at `b8de60d`; hashes differed; Omar's rebuild found the cause (CRLF on Windows) and now matches | `data/output/panel_rebuild_check.json` @ `37014de`; `data/output/panel_rebuild_check_omar.json` |
| W11 | Family alpha hard-coded; always-true test | Alpha from specs; three real tests; family controls on the real panel | `data/output/controls_family.json` @ `eccdeda`; `docs/validator.md` 2.6 |
| W12 | Drill-down never runs on real data | Planted one-state, one-sector and national effects | `data/output/drilldown_check.json` @ `72306a7` |
| W13 | H2 null/SE gap untouched | = F5 | as F5 |

## Evidence rows

| Claim | Number | Path @ commit |
|---|---|---|
| Panel rebuild matches reference hash | First round **no**: `ed5c05d7…` here vs `7d0e2bda…` (Omar, 2026-10-24), rows, columns and join stats equal. Second round **yes**: `3dfa5eb1…` on both machines after the line-ending fix | `data/output/panel_rebuild_check.json` @ `37014de`; `data/output/panel_rebuild_check_omar.json` |
| Panel build deterministic on one machine | Builds of 2026-10-25 and 2026-11-08 identical cell for cell | same file, `cell_diff` |
| Panel with CHR columns | 2,390,744 rows, 53 columns, sha256_csv `3dfa5eb1…` | `data/output/panel_join_log.json` @ `1437cb9` |
| CHR coverage | ~97.3% of need rows per year for health/education; crime 91.4%; hs_completion 0% in 2018 | `county_need_report.json` → `chr.match_rate_by_year` @ `1437cb9` |
| CHR contract | exact county rows per release: 3,142 (2020–2023), 3,143 (2024), 3,152 (2025); 0 values out of range | `county_need_report.json` → `chr.contracts`, `chr.range_violations` |
| CHR crime is frozen | v043 identical for all 2,951 counties across releases 2019–2022; absent 2023+ | D8 (measured on the raw files) |
| H2 null vs SE | ratings: null SD 0.0288, county SE 0.0485, state SE 0.0532; mean permuted SE 0.0277 | `data/output/h2_null_check.json` @ `7ab7337` |
| Large-county leverage | top 10 counties: 13% of rows, 89% (ratings) / 95% (sweep cell) of clustered variance | same |
| Gate p under the new statistic | ratings H2 0.656 (was 0.436), T5 0.514 (0.439), sweep cell 0.133 (**0.001**) | same |
| Family false positives | 0 of 8 placebo theories; planted 0.25 SD theory supported (q 0.045) | `data/output/controls_family.json` @ `eccdeda` |
| Drill-down finds the stratum | GA planted → GA only survives (q 0.036) of 36 states; Education planted → Education only (q 0.026) of 26 sectors; national 0.25 SD → expected sign in 34/36 states, 26/26 sectors, 3/3 terciles | `data/output/drilldown_check.json` @ `72306a7` |
| States skipped by the 30-county rule | 15 of 51 (219,642 of 1,303,953 rows, 16.8%) | same |

## Findings

1. **The panel hash does not reproduce across our two machines (F6).** Same rows, same columns, every join statistic identical, and no panel code changed between the reference build (`517a92b`) and `b8de60d`. My own two builds two weeks apart are identical cell for cell, so the build is deterministic within one environment. The difference is between environments (Omar's 2026-10-24 build predates his pandas/numpy reinstall). **Closed by Omar's rebuild:** the 48 shared column hashes are equal on Windows and macOS, and the whole-file gap was line endings (pandas writes CRLF on Windows). With LF forced in `build_panel.py` (`aa1319b`), both machines give sha256_csv `3dfa5eb1…` (`data/output/panel_rebuild_check_omar.json`, `match.sha256_csv: true`).
2. **CHR crime data is frozen.** CHR carries the same 2014/2016 UCR violent-crime values in every release from 2019 to 2022 and drops the measure from 2023. `chr_violent_crime_rate` is therefore a static county attribute, so T4 can only compare counties, not years.
3. **Poor or fair health has a level break** between tax years 2019 and 2020 (median 0.201 → 0.153; county correlation 0.97). T2 must compare within year (tax-year effects are in H2; a pooled H1 correlation is not protected).
4. **Connecticut gets CHR values only where CHR publishes planning regions** (2025 release → tax 2022, health and uninsured): 9 rows. Puerto Rico has no CHR data.
5. **The CP3 gate's H2 p-values were too small whenever the exposure sits on a few large counties.** The cause is the count of exposed *counties* being held fixed while the count of exposed *rows* is not: high-need counties cover 8% of rows, random sets 23%. Spatial correlation and row-level heteroskedasticity were ruled out by measurement. The studentized permutation fixes the ratings and sweep cases; under the old gate the sweep cell's p = 0.001 was an artefact. **Every CP4 theory and sweep result must be rerun under the new gate.**
6. **The permutation count limits the family size**: a theory can survive BH only if the family has fewer than 2·alpha·(n+1) hypotheses (100 at 999 permutations), and a drill-down over k strata needs n > k/alpha (36 states → more than 720 permutations).
7. **The one-level drill-down finds the right stratum, but only for effects that survive nationally.** A 0.3 SD effect confined to Georgia or to Education is unsupported nationally (−0.040 / −0.065 SD), so the survivor-only rule never drills it. The check had to force the search open. Drill-down explains where a national effect lives; it is not a search for local effects.

## Changes (with reasons)

| Change | From | To | Reason |
|---|---|---|---|
| Ownership | Vien: orchestrator (CP3 report) | Vien: county data module (F1) and gate statistics; Omar: orchestrator | The TA built the orchestrator; its CP4 gaps sit in Omar's agent code. The feedback items F1/F5/F6 are data and gate work. |
| Drill-down (F4) | Automatic: any depth, any dimension | One level: one stratifier per run (state, sector or size), survivors only, strata in one BH family | Each extra level multiplies the family; with 36 testable states one level already needs 720+ permutations to let any stratum survive (finding 6). Deeper drill-down would be data mining without a family that can be controlled. |
| Gate statistic (F5) | Permute the raw coefficient | Permute t = coefficient / county jackknife SE, for H1 and H2 (`permutation.studentized`) | Measured in `h2_null_check.json`: raw-coefficient null too narrow when the exposure sits on large counties; jackknife beat CR1 in a size simulation (0.11–0.12 vs 0.14–0.18 vs 0.28–0.33 for the raw coefficient). |
| Family alpha (W11) | Hard-coded 0.05 | From the specs; mixed alphas rejected | The gate is locked in the spec policy; the family step must use the same value. |
| Need table | ACS + NORP only | + 5 CHR columns, D8 vintage rule | CP3 feedback F1 |
| HS completion | (plan: 2018–2022) | missing for 2018 | v168 starts in the 2021 release; using it for 2018 would look ahead (D4/D8) |
| Per-year tests (`trends.py`, Omar's module, agreed) | Raw-coefficient permutation | Same studentized statistic as the pooled gate | So the per-year results and the pooled verdict are judged the same way (F3 + F5) |

## V6: review of the theory runs

`scripts/review_theory_runs.py` recomputes each run's family q-values from its stored permutation p-values and its verdict from those numbers, checks that the run used the current gate, and checks the spec against the gold answer (`data/output/theory_runs/cp4/review.json`).

- **ratings_fairness** (`distressed_county_rating_penalty`): unsupported. Follows from the numbers. H1 -0.022 SD (family q 0.725), H2 -0.012 SD (q 0.78), replicated False (0/5 folds); years passing H1 none, trend p 0.14.
- **education_fundraising** (`education_fundraising_cost`): unsupported. Follows from the numbers. H1 -0.000 SD (family q 0.966), H2 -0.001 SD (q 0.78), replicated False (2/5 folds); years passing H1 none, trend p 0.23.
- **health_margin** (`health_burden_thin_margins`): unsupported. Follows from the numbers. H1 -0.000 SD (family q 0.78), H2 +0.006 SD (q 0.00667), replicated False (0/5 folds); years passing H1 none, trend p 0.65.
- **unemployment_vulnerability** (`unemployment_leverage`): unsupported. Follows from the numbers. H1 +0.001 SD (family q 0.078), H2 +0.001 SD (q 0.078), replicated False (4/5 folds); years passing H1 none, trend p 0.17.
- **crime_overhead** (`violence_admin_overhead_share`): unsupported. Follows from the numbers. H1 +0.027 SD (family q 0.005), H2 +0.035 SD (q 0.005), replicated True (5/5 folds); years passing H1 none, trend p 0.04.

Before the rerun (`85563d7`, raw-coefficient gate), T1 and T5 were both unsupported and both verdicts followed from their numbers. One mismatch in that commit: the message says T1 had "1/5 state folds right sign", while the JSON has `folds_with_expected_sign: 0`.

## AI usage

Built on the TA's `ai-suggestions/cp4` code as imported unchanged in `7d9ee8a`; only the work above is ours. Claude Code (Claude Opus 5.5) was used for code, the measurement scripts and drafting this text. It ran every command and measured every number on the real panel; I chose the CHR vintage rule and the gate statistic from the measurements. The CHR data years come from CHR's 2025 "Select Measures and Data Sources" table, read for 2025 only; the 2020–2024 lags are assumed equal (D8 says so).
