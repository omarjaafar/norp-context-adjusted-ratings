# The Validator gate

The Validator decides whether a theory's finding stands. It is deterministic Python (`theorylab/validator.py`, called from `theorylab/theory_run.py::run_spec`). No LLM computes a number or changes a verdict. This page describes how the gate works, the review of its permutation scheme (CP3 audit items U7 and U8), and what changed in CP3.

## 1. The gate in one page

| Step | What it does | Setting (`specs/ratings_fairness.json`) |
|---|---|---|
| Population | Full-990 filers with a county and need data, complete in every column the spec uses | `population` |
| H1 | Difference in mean outcome, exposed vs. other rows | `hypotheses[H1]` |
| H2 | Fixed-effects coefficient on the exposure, with numeric controls, absorbed factors and SEs clustered by county | `hypotheses[H2]` |
| p-values | Permutation test for H1 and H2: the exposure is reassigned across counties, and the **studentized** statistic (slope / county jackknife SE, CP4 section 2.5) is recomputed | `n_permutations`, `seed`, `permutation` |
| Multiple testing | Benjamini–Hochberg over {H1, H2} | `alpha` |
| Effect floor | \|effect\| ≥ 0.1 SD of the outcome | `effect_floor_sd` |
| Replication | H2 is refit with states held out, in 5 folds | `holdout` |
| Verdict | `decide_verdict()` | — |

**Verdict rule.** A hypothesis *holds* when its q < alpha, its effect clears the floor, and its sign is the expected one.

| Verdict | Condition |
|---|---|
| reversed | H1 holds with the opposite sign |
| unsupported | H1 does not hold |
| supported | H1 holds, H2 holds, and H2 replicates in held-out states |
| weak | H1 holds, but H2 fails or does not replicate |

**Permutation unit.** Need is assigned to counties, and organizations in one county share it. So the permutation moves whole counties, never single organizations. With `"permutation": {"keep_paths": true}` (the setting for the ratings theory), each county takes another county's **whole history** of exposure over the tax years. Counties are swapped only with counties present in the same years, so the number of exposed counties per year is exactly the real one. Section 2 explains why.

**Held-out-state replication.** States are dealt into 5 folds at random (the seed is `validator.seed + 2`). For each fold, H2 is fit on the other states and on the held-out fold. H2 *replicates* when:

1. in every fold, the held-out estimate has the expected sign (and the sign of its own training fit), and
2. the inverse-variance-pooled held-out estimate has p < alpha.

Rule 1 catches a gap that a few states carry. Rule 2 catches a gap that has the right sign everywhere but cannot be told apart from zero. The folds share no states, and so no counties, which makes the five held-out estimates independent and the pooling valid. We did not require significance in every fold: each fold has about a fifth of the data, so that rule would reject real effects for lack of power. Cochran's Q across folds is reported for information. The old held-out-year test is still reported as `replication_year`, but it no longer feeds the verdict. Need barely moves between years, so a held-out year re-tests the same organizations in the same counties (CP2 feedback G1).

## 2. Review of the permutation scheme

### 2.1 The permutation unit was wrong: whole county histories, not one year at a time

The TA code shuffled `high_need` across counties **separately in each tax year**. On the real data, `high_need` is very persistent:

| Years high-need, of 5 (counties in all 5 years) | Counties |
|---|---|
| 0 | 2,207 |
| 1–4 | 408 |
| 5 | 602 |

87% of counties never switch. A year-by-year shuffle gives a county a new random exposure every year. But the outcome is persistent too: the same organizations file every year, and county effects carry over. So in the real data, a county's five years of ratings move together with one persistent exposure. In a year-by-year shuffle they are spread across five independent exposures, which averages the county effect out. The resulting null is too narrow, and the p-values are too small.

Measured on the real panel (1,303,953 rows, 3,087 counties, 199 permutations). Every number in sections 2.1 and 2.2 comes from `scripts/permutation_check.py` → `data/output/permutation_check.json`.

| | Year-by-year shuffle (TA) | Whole-history shuffle (CP3) |
|---|---|---|
| H1 null SD | 0.0169 | 0.0310 |
| H1 p | **0.010** | **0.135** |
| H2 null SD | 0.0153 | 0.0282 |
| H2 p | 0.095 | 0.440 |

The null is about 1.8 times wider, close to the √5 ≈ 2.2 expected when five persistent years act as one. The ratings verdict does not change: it is unsupported either way, because H1 is far below the floor. But the claim "the gap is significant under the county permutation (q = 0.02)" from the first audit run does not hold. The test `test_year_by_year_shuffle_understates_the_null_when_exposure_persists` reproduces this effect on a synthetic panel.

`keep_paths` is an option of `UnitPermuter`, off by default, and on in `specs/ratings_fairness.json`. Any theory whose exposure persists over years should turn it on. The per-year runs in `theorylab/trends.py` are unaffected, because each tests one year.

**Resolved in CP4 (section 2.5).** Even with whole histories, the H2 permutation null SD (0.028) was smaller than H2's county-clustered SE (0.049). Section 2.5 measures why and changes the statistic the gate permutes.

### 2.2 Permute-X versus Freedman–Lane (U7)

H2's permutation shuffles the exposure X, keeps the controls Z fixed, and refits by Frisch–Waugh–Lovell. This is the Draper–Stoneman scheme. Its known weakness: when X is correlated with Z, shuffling X breaks that correlation, so the null describes an X unrelated to the controls. Freedman–Lane avoids this by permuting the residuals of y on Z instead.

For the ratings theory, the question does not matter. The controls (log revenue, tax-year and category fixed effects) explain **0.3%** of the variance of `high_need` (R² = 0.0030 on the real population). With almost no X–Z correlation to break, the two schemes target the same null.

We did not add a Freedman–Lane option in CP3. It is first on the drop list, and the measurement shows it is not needed here. It does matter for CP4 theories whose exposure is strongly tied to a control, for example an exposure that is mostly organization size. Two points for that version:

- The exchangeable units are counties, which differ in size, so residuals cannot be swapped between counties row by row. A cluster version has to move whole-county residual blocks (sign-flipping blocks is the usual way), and that needs its own synthetic test.
- The check that decides whether it is needed is cheap: report the R² of the exposure on the controls next to every H2. We suggest the Executor do this from CP4 on.

### 2.3 What the BH family must be once several theories are tested (U8)

Today the family is {H1, H2} of one theory. That is right for a single theory and wrong as soon as several are tested together: 12 theories at alpha 0.05 with a separate BH each would let through far more than 5% false discoveries across the benchmark. The rule from CP4 on:

- **The family is every primary hypothesis decided in one run.** That means H1 and H2 of every theory in the run, and every cell of the sweep (Omar's O6: 4 ratios × 2 need measures = 8). One BH correction runs over all of them, and each verdict uses its q from that family.
- **Not in the family:** robustness models and anchor variants, which never change a verdict; and the negative and positive controls, which test the gate itself rather than a theory.
- **Drill-down** (state, sector, size) gets its own family per surviving theory, counting every subgroup tried, not only those reported.
- **Dependence.** BH holds under positive dependence (PRDS). Tests that share an outcome and use overlapping exposures are positively dependent in practice, so BH is fine there. If the family mixes outcomes whose tests could be negatively related, use Benjamini–Yekutieli (`multipletests(method="fdr_by")`) instead.

`run_spec` takes one spec at a time, so a multi-theory run has to collect the p-values first and apply BH afterwards. That is a CP4 change to the runner, not to `benjamini_hochberg()`.

### 2.4 An asymmetry in the verdict rule

"Supported" needs H1, H2 and replication. "Reversed" needs only H1, with the opposite sign. So a gap in the unexpected direction is easier to declare than one in the expected direction, and it is never checked against the controls or held-out states. In the synthetic controls test, a small panel (10 organizations per county) produced a chance "reversed" verdict (seed 2) for exactly this reason. That panel was enlarged; on the real panel the noise is far below the floor, and none of the 20 negative-control runs was reversed. For CP4 we suggest that "reversed" should also require H2 and replication with the opposite sign, and otherwise be "weak". We have not changed it in CP3, so the CP3 verdicts follow the rule as written above.

### 2.5 The H2 null/SE gap: permute the studentized slope (CP4 V3, CP3 feedback F5)

**The gap.** For the ratings theory, H2's permutation null had SD 0.029, and the county-clustered SE of the same coefficient was 0.049. Under the null both measure the spread of the coefficient, so one of them was wrong. The sweep showed it matters. On liabilities/assets × housing burden, the permutation q was 0.003 and the clustered p was 0.11.

**What we measured.** `scripts/h2_null_check.py` → `data/output/h2_null_check.json`, on the real panel, 999 permutations per scheme, for three H2 models: ratings (binary `high_need`), T5 (unemployment → liabilities/assets) and that sweep cell (both continuous).

| | Ratings | T5 | Sweep cell |
|---|---|---|---|
| Clustered SE at the real exposure (county) | 0.0485 | 0.0977 | 0.2887 |
| Clustered SE, by state | 0.0532 | 0.1098 | 0.2890 |
| Permutation null SD, whole histories (CP3 gate) | 0.0288 | 0.1002 | 0.1174 |
| Mean clustered SE over the permuted draws | 0.0277 | 0.0886 | 0.1057 |
| SD of the permuted t (clustered SE) | 1.08 | 1.06 | 1.05 |
| Null SD, histories shuffled within state | 0.0218 | 0.1206 | 0.1530 |
| Real SE / mean permuted SE | 1.75 | 1.10 | 2.73 |
| Top 10 counties: share of rows / of clustered variance | 13% / 89% | 13% / 73% | 13% / 95% |
| p: coefficient permutation (CP3 gate) | 0.436 | 0.439 | **0.001** |
| p: clustered t, normal | 0.613 | 0.449 | 0.113 |
| p: permuted t, county jackknife SE (CP4 gate) | **0.656** | **0.514** | **0.133** |

The four candidate causes, one at a time:

1. **Spatial correlation across counties: no.** Clustering by state barely moves the SE (0.053 vs 0.049; 0.289 vs 0.289). Shuffling only within states makes the ratings null *narrower* (0.022), and it widens the continuous nulls only a little (0.15 against an SE of 0.29).
2. **Row-level heteroskedasticity: no.** Residual variance is the same in exposed and other rows (4.61 vs 4.52 for ratings).
3. **A few very large counties: yes.** The clustered SE is calibrated for random exposures: averaged over the permuted draws it equals the null SD (0.028 vs 0.029, 0.106 vs 0.117), the permuted t has SD 1.05–1.08, and the clustered test rejects 5–7% of the null draws. At the **real** exposure the SE is 1.75 times (ratings) and 2.73 times (sweep cell) its typical permuted value. Ten counties (Los Angeles, New York, Cook, DC, Harris, King, San Diego, Maricopa, Orange, Suffolk MA) hold 13% of the rows but 89–95% of the clustered variance. The real exposure puts weight on some of them, and random exposures mostly do not. High-need counties are small (146 rows on average, against 504), so the real exposure covers 8% of rows, while a random set of the same number of counties covers about 23%. The permutation keeps the number of exposed counties, not the number of exposed rows, so a raw-coefficient null describes a design with about three times as many exposed rows. It is too narrow. With the 10 counties dropped, the coefficient permutation calls ratings H2 significant (p = 0.014) and the clustered SE does not change (0.047). That p is the narrow null at work, not a finding.
4. **The Residualizer: correct.** The permuted exposure is residualized on the same controls and absorbed effects as the fit. The FWL coefficient equals the fixed-effects coefficient to 1e-8, the recomputed CR1 SE equals the fit's, and a permuted, residualized exposure has mean 5e-15 inside every absorbed level and correlation 7e-18 with log revenue.

**Decision: permute the studentized slope.** From CP4 on, the permutation test for H1 and H2 recomputes t = slope / SE in every draw and compares the real t with the permuted t's (`validator.ClusteredSlopeT`, switched on by `"permutation": {"studentized": true}` in `spec.GATE_POLICY` and `specs/ratings_fairness.json`). The reported effects are unchanged. The coefficient's own permutation null is still reported under `permutation.coefficient`, for comparison.

- Permuting t is exact when counties really are exchangeable, as a coefficient permutation is, and it stays valid when they are not (Chung & Romano 2013; DiCiccio & Romano 2017). Each draw is scaled by its own SE, so a draw that happens to put the exposure on a big noisy county is judged against that county's noise.
- **The SE is the county jackknife, not the CR1 sandwich.** With a few dominant counties the sandwich understates the variance, because a high-leverage county's own residuals are shrunk by the fit. The leave-one-county-out slope does not have that problem. In a simulation with no effect and the real panel's pattern (skewed county sizes; the exposed set is the smallest counties plus a few of the largest; 300 or 1,000 counties; 150 runs; 5% level), the share of runs that rejected was:

  | | Coefficient permutation | Permuted t, CR1 | Clustered t, normal | Permuted t, jackknife |
  |---|---|---|---|---|
  | 300 counties, 3 big exposed | 0.28 | 0.18 | 0.25 | **0.12** |
  | 1,000 counties, 8 big exposed | 0.33 | 0.14 | 0.16 | **0.11** |
  | no big county exposed | 0.00 | 0.02–0.03 | 0.02–0.05 | **0.02–0.03** |

  No method reaches 5% when the very largest counties are exposed; the jackknife comes closest. The test `test_studentized_permutation_cuts_false_positives_when_big_counties_are_exposed` keeps this result (35% vs 12% on 60 runs).
- **Alternatives we did not take.** Shuffling within state does not fix the gap (point 1). Taking the larger of the permutation p and the clustered p uses the CR1 SE, which the simulation shows is too small under leverage. A wild cluster bootstrap would answer the same question with a different resampling. The studentized permutation reuses the county-history shuffle we already validated in CP3 (section 2.1), and adds two bincounts per draw.
- **What it changes.** Nothing is significant either way for ratings H2 or T5. The sweep cell goes from p = 0.001 to 0.133, in line with its clustered p of 0.113. With the 10 largest counties dropped, ratings H2 goes from a spurious p = 0.014 to 0.306. Every CP4 theory and sweep result must be rerun under the new gate.
- **Approximation.** The jackknife drops a county from the already-residualized vectors and does not refit the controls and absorbed effects without it. Those effects are estimated from all 3,000+ counties, so one county moves them very little.

### 2.6 The gate for many theories (CP4 V4, audit W11)

- `gate.apply_family` takes alpha from the reports (`validator.alpha`), not a hard-coded 0.05, and refuses a family whose reports were run with different alphas. The alpha used is written to each report as `validator.family_alpha`.
- The test that always passed is replaced by three that can fail: a family of 1 true + 9 null theories (20 hypotheses), where the true theory survives at p = 0.004 (q = 0.04) and not at p = 0.006 (q = 0.06); a theory supported alone (q = 0.01) that is unsupported in a family of 40 hypotheses (q = 0.20); and alpha taken from the specs.
- **The permutation count limits the family size.** The smallest permutation p is 1/(n + 1). A theory whose H1 and H2 rank first and second gets q = p × m / 2, so it can survive only if m < 2 × alpha × (n + 1). At 999 permutations that is m < 100 hypotheses (50 theories); at 199, m < 20. A drill-down family of k strata needs k < alpha × (n + 1): 36 states need more than 720 permutations, so the drill-down must run at the gate's 999.
- **Family controls on the real panel** (`scripts/run_controls.py --family` → `data/output/controls_family.json`, 199 permutations, the studentized gate): 8 placebo theories (each county takes another county's whole exposure history) and 1 theory with 0.25 SD planted in high-need counties, in one family of 18 hypotheses. **0 of 8 placebos survive** (family q 0.225 to 0.99, every effect within ±0.025 SD). **The planted theory stays supported** (H1 −0.271 SD, H2 −0.261 SD, family q 0.045, replicated). Its q is at the 199-permutation floor (0.005 × 18 / 2); at 999 it would be about 0.009.

## 3. Evidence that the gate works (V2)

`scripts/run_controls.py` → `data/output/controls.json`, on the real population, with 199 permutations and no robustness models:

| Control | Expected | Result |
|---|---|---|
| Negative: exposure replaced by another county's whole history, 20 seeds | unsupported × 20 | **20 / 20 unsupported**; none significant, none above the floor, none replicated |
| Positive: −0.25 SD planted in high-need counties | supported | **supported** (H1 −0.271 SD, H2 −0.261 SD, replicated in 5/5 state folds) |
| Positive: −0.05 SD planted | unsupported (below floor) | **unsupported** (H1 −0.072 SD: significant at q = 0.005, but below the 0.1 SD floor) |

The negative-control placebos keep the real exposure's persistence over years, so a test that ignored persistence would show up here as too many significant H1 gaps. With the whole-history shuffle, none were significant.
