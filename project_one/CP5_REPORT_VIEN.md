# CP5 report sections: Vien Tran (outcome rule, theories, controls, reruns)

Commits after the TA import `f825052`: `970907b` (V1), `3c76453` (V3), `0eaed99` (V7 review tool), `6e3116a` (V2), `3059286` (V5), `747f8ad` (V6), `3792229` (report, handoff, V4 script), `53860c5` (V4).
Environment: Arch Linux (kernel 7.2), x86-64, 22 cores, 30 GB RAM, Python 3.12.16 (uv venv), packages exactly as pinned in `requirements.txt` (pandas 3.1.1, numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0). 192 tests pass (185 at `466f21a`). The panel was rebuilt from a clean clone on Linux and matched the CP4 reference hash `3dfa5eb1…` before V3 changed it.

## Match items (CP4 feedback → what was done)

| # | Feedback | Done | Evidence |
|---|---|---|---|
| F1 | Name the ~10 literature theories with source, outcome, exposure, sign | 10 literature entries, each with a full citation (`source`) and what the source argues versus our inference (`claim`), checked against the paper or its abstract on 2026-11-19. File frozen before any Planner run | `benchmark/cp5_theories.json` @ `3c76453` |
| F2 | Crime is one 2014/2016 snapshot; find a multi-year source | Source chosen (Kaplan, UCR Return A, openICPSR 100707). It needs an account, so it is a documented roadblock; L9 swapped for a spare before the freeze. T4 proxy check run anyway | `docs/data_decisions.md` D9 @ `3059286`; `data/output/t4_followup.json` |
| F3 | Fair share of the CP5 analysis | 9 commits after `f825052` (V1–V7, report, V4 results); the outcome rule, theories, controls, reruns and crime are mine | `git log --author=Vien f825052..` |
| F4 | Plant a positive control in a heavy-tailed ratio | Two positive controls on `operating_margin` and `liabilities_to_assets`, planted in the raw outcome before the rule. Missed without the rule, recovered with it | `data/output/controls_heavy_tail.json` @ `6e3116a` |
| X3 | Control planted after winsorizing | Planted raw at 0.25 × the post-rule SD; the rule is applied once afterwards (the TA code applied it twice) | `theorylab/benchmark_run.py` `prepare_population` @ `6e3116a` |
| X4 | The rule clips bounded shares; some outcomes stay above 3 | D-a (bounded outcomes exempt), D-b (pooled clip), D-c (one clip, logged). Measured on all 10 outcomes | `data/output/outcome_rule_check.json` @ `970907b`; `docs/validator.md` 2.8 |
| X7 | One placebo draw per negative control | 8-placebo families on both heavy-tailed outcomes under the rule: 0 false positives | `data/output/controls_family_heavy_tail.json` @ `747f8ad` |
| X8, X9 | Sources do not support their claims; theories repeat CP4 | Sources replaced or dropped; new sector-side exposure `nonprofit_density` (L6) | `benchmark/cp5_theories.json`; `docs/data_decisions.md` D9 |

## Evidence rows

| Claim | Number | Path @ commit |
|---|---|---|
| Spread (SD / robust SD) before → after the rule, 990 filers pooled | `revenue_growth` 265,883 → 4.03; `equity_ratio` 68,191 → 4.95; `operating_margin` 12,697 → 3.68; `fundraising_efficiency` 1,268 → 3.96; `liabilities_to_assets` 780 → 2.08 | `data/output/outcome_rule_check.json` @ `970907b` |
| Outcomes still above 3 after one clip (D-c) | 4 of 5 clipped outcomes (3.68–4.95); logged as `still_above_threshold_after` | same |
| Bounded shares exempt (D-a) | fundraising share: spread 4.94, 63.2% zeros. The TA rule would clip it at 0.313 (13,400 rows); now untouched. Program 1.38, admin 1.58, concentration 0.70, rating 0.90: untouched | same, `without_exemption` |
| Panel with `nonprofit_density` | 2,390,744 rows, 54 columns, sha256_csv `334951b1…`; county median 6.9 (2018) → 9.0 (2022) full-990 filers per 10,000 residents | `data/output/panel_join_log.json` @ `3c76453` |
| Heavy-tail control, margin × unemployment, 0.25 SD | rule on: **supported**, H1 −0.253 SD; rule off: unsupported, −0.001 SD (raw SD 2,502 vs 0.72 after the rule); no effect: unsupported (−0.016 SD, under the floor) | `data/output/controls_heavy_tail.json` @ `6e3116a` |
| Heavy-tail control, leverage × poverty, 0.25 SD | rule on: **supported**, +0.274 SD; rule off: unsupported, +0.002 SD (raw SD 154 vs 0.47); no effect: unsupported (+0.038 SD, under the floor) | same |
| Placebo families under the rule | 0 of 8 placebos not unsupported on either outcome (largest placebo H1 0.006 and 0.010 SD); planted control supported after the family correction on both (q 0.045) | `data/output/controls_family_heavy_tail.json` @ `747f8ad` |
| T4 is not a proxy | H2 +0.035 SD as run; +0.032 SD with poverty, income, population and need index together (90% kept); p 0.001 and 5/5 folds in every variant; under the 0.1 SD floor | `data/output/t4_followup.json` @ `3059286` |
| T1, T4 do not move under the rule | T1 rating H1 −0.022 / H2 −0.012 SD, T4 admin share +0.027 / +0.035 SD: identical to CP4 (rule not applied, SD 2.15 and 0.202 unchanged) | `data/output/theory_runs/cp5_reruns/review.json` @ `53860c5` |
| T2, T3, T5 under the rule (CP4 → CP5) | outcome SD 2,510 → 0.72 (T2), 43.9 → 0.139 (T3), 154 → 0.475 (T5). H1 / H2: T2 −0.000 / +0.006 → −0.013 / +0.048 SD; T3 −0.000 / −0.001 → **+0.065 / +0.055** SD (expected −); T5 +0.001 / +0.001 → +0.045 / +0.037 SD (5/5 folds). All five unsupported; every verdict follows from its numbers (family of 10 hypotheses) | same |
| Rerun time | 9.0 min wall for all five at 999 permutations with per-year results (5 processes, BLAS on one thread); the same job multi-threaded had not finished after 5 h 51 min | `cp5_reruns/summary.json` → `seconds_per_theory`, `seconds_wall` |

## Findings

1. **The rule fixes the disasters but not completely, and we do not iterate (D-c).** One 1%/99% clip takes the five unbounded ratios from spreads of 780–265,883 down to 2.1–5.0. Four stay above the 3.0 threshold, because more than 1% of filers on each side are far from the middle (for example, organizations with almost no revenue). Clipping again would keep moving the clip points until the ratio fell under a number that was chosen to diagnose tails, not to clip them. So the gate sees an outcome whose SD is still about 4 times its robust SD. That is a large improvement on the 13,000× of CP4, but the floor is still somewhat harder to reach for those outcomes than for the rating.
2. **The TA rule would have clipped the fundraising share.** With 63% of filers at zero, the IQR is small and the spread is 4.94, so the ratio test fires on a share that is bounded in [0, 1] by construction. It would have cut it at 0.313, contradicting the TA doc's claim that expense shares pass through. One TA positive control used that outcome. The exemption is read from the catalog (`bounded: [0, 1]`), not from a list in code.
3. **The heavy-tail positive controls show the rule is needed, not just harmless.** The same 0.25 SD effect, planted in the raw ratio, is **invisible without the rule**: −0.001 and +0.002 SD, because the raw SD is 2,502 and 154. With the rule it is **recovered at full size** (−0.253, +0.274 SD), and the unplanted versions stay unsupported. CP4's "unsupported" for T2, T3 and T5 therefore said nothing about those theories. This is the evidence for CP4 feedback F4.
4. **The rule does not create false positives.** Across 16 placebo theories on the two heavy-tailed outcomes, the largest placebo effect is 0.010 SD and none survives the family. The real, unplanted relations do reach significance (p 0.005) but stay well under the floor (−0.016 and +0.038 SD), so the floor, not the p-value, is what keeps them unsupported.
5. **T4's crime effect is small and real, not a proxy.** Controlling for poverty, income, county population and the need index together keeps 90% of it (+0.035 → +0.032 SD), with p 0.001 and all 5 folds in the expected direction every time. It is about a third of the floor. Without a multi-year crime source we cannot say whether it changes over time.
6. **CP4's LLM replay no longer works, and cannot be restored by reverting one file.** The cache is keyed on the exact prompt. The TA import (`f825052`) added `outcome_rule` to `specs/ratings_fairness.json`, which is the example spec in every Planner prompt, and that alone makes every CP4 key miss (bisected: replay works at `f661856`, fails at `f825052`). V3's new column changes the Scout's column list as well. The CP4 specs are committed, so V4 reruns them directly (`scripts/rerun_specs.py`, no LLM call). The README quickstart must not rely on `run_cp4.py --replay`.
7. **Under the rule, CP4's three uninformative theories become measured small effects, and one points the wrong way (V4).** T1 and T4 do not move at all, so the rule leaves well-behaved outcomes alone. For T2, T3 and T5 the outcome SD falls by a factor of 200 to 3,500, and effects that read as 0.000 SD in CP4 are now measurable and significant (family q ≤ 0.014). They are still under the 0.1 SD floor, so all three stay unsupported, but "unsupported" now means "a real effect, smaller than we count", not "the test could not see". **T5** (unemployment → leverage, +) is in the expected direction in all 5 held-out folds at about half the floor. **T3** (education → fundraising cost, expected −) is **+0.065 SD: more educated counties have more expensive fundraising per dollar**, significant and opposite to the theory, but under the floor, so the gate reports unsupported, not reversed. **T2** is split: H1 −0.013 SD in the expected direction, H2 +0.048 SD against it once size, sector, year and state are absorbed. The CP4 verdicts are kept (2.8); these are reported next to them.
8. **Thread contention made the gate about 30 times slower on a many-core machine.** OpenBLAS spreads every small matrix product over all cores. One T1 run at 199 permutations used 60 CPU-minutes multi-threaded and 2 single-threaded. `rerun_specs.py` now pins BLAS to one thread and runs theories in parallel (`--jobs`). V2 and V6 ran before this fix; their numbers are unaffected, only their run times (35 and 60 min).

## Changes (with reasons)

| Change | Reason | When |
|---|---|---|
| Bounded [0, 1] outcomes exempt from the heavy-tail rule (D-a) | The ratio test fires on a share with many zeros; its SD is already meaningful | Decided 2026-11-19, before any rerun |
| Clip points pooled over years (D-b) | One scale across years for per-year and trend tests | same |
| One clip, ratio after logged (D-c) | Iterating would chase a threshold chosen for diagnosis | same |
| Positive controls planted raw at 0.25 × post-rule SD | The gate measures in post-rule SD; the raw SD plants a huge effect, the robust SD one under the floor; planting after the rule never tests the rule | same |
| Rule applied once, in `run_spec` only | The TA `prepare_population` applied it too, so every benchmark entry was clipped twice | found in V2 |
| Two TA positive controls replaced by heavy-tailed ones | CP4 feedback F4; all five TA controls were on well-behaved outcomes | before the freeze |
| Literature sources replaced; L6 added; L9 swapped for a spare; L10 sourced but flagged | Audit X8/X9; F1; crime roadblock (F2) | before the freeze |
| New column `nonprofit_density` | L6 needs a sector-side exposure; the only theory not about local need | V3 |
| Multi-year crime: roadblock, not added | The source needs an openICPSR account; county-level public alternatives do not exist | V5 |
| CP4 theories rerun from committed specs, not LLM replay | The prompts changed (finding 6) | V4 |

## Literature sources (V3): what each one supports

| id | Source | Supports the sign directly? |
|---|---|---|
| `ratings_fairness` | Gregory & Howard (2009, SSIR); Lecy & Searing (2015, NVSQ) | No: they support "overhead-based scores mismeasure"; the need link is the CP3 hypothesis |
| `lit_poverty_reserves` | Pfeffer & Salancik (1978); measure Tuckman & Chang (1991) | Partly: resource dependence applied to county poverty |
| `lit_unemployment_margin` | Salamon, Geller & Spence (2009) | Direction only: a survey of a downturn, not a county comparison |
| `lit_income_fundraising` | Wolpert (1993) | Partly: richer places give more; cheaper fundraising is our inference |
| `lit_income_concentration` | Grønbjerg & Paarlberg (2001); measure Carroll & Stater (2009) | Weakest: supply-side resources, concentration is our inference |
| `lit_density_fundraising` | Thornton (2008, JLEO) | **Yes**: more competition raises the share of donations spent on promotion |
| `lit_highneed_growth` | Salamon, Geller & Spence (2009) | Direction only |
| `lit_uninsured_programs` | Weisbrod (1988) | Weak: theory of where nonprofits exist, not how they spend |
| `lit_education_fundraising` | Wolpert (1993); Grønbjerg & Paarlberg (2001) | Partly (spare that replaced L9) |
| `lit_housing_debt` | Calabrese (2011, PBF) | Weak: pecking order; the housing-cost link is our inference |

Dropped: Greenlee & Trussel (2000) (about vulnerability prediction, not donor wealth); "Hager and others" (not a citation); McConville & Cordery (2021) (performance-reporting methods); the unsourced food-desert and high-school-reserves theories.

## V7: review of the benchmark run

To be filled after Omar's benchmark run (O4): `python scripts/review_theory_runs.py --dir data/output/theory_runs/cp5` prints one line per entry and checks that every verdict follows from its numbers and that every entry was judged in its own family. Add one line of findings per literature theory under it. See `project_one/CP5_HANDOFF_OMAR.md`.

## AI usage

Built on the TA's `ai-suggestions/cp5` code as imported at `f825052` (audited in our plan, section 3). My work used Claude Code (Claude Opus 5.5) as a coding assistant. It implemented V1–V7 from our written plan, which fixed the decisions (D-a to D-c, the planting scale, the theory list) before any code or run. It also looked up and summarized the literature sources: abstracts and bibliographic records for Thornton (2008), Lecy & Searing, Gregory & Howard, Tuckman & Chang, Weisbrod and Calabrese. Every number above comes from a committed script and output file. I reviewed the code, the source checks and the results before committing. The assistant found the double application of the rule in `prepare_population` and bisected the broken LLM replay to `f825052`.
