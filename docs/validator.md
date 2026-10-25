# The Validator gate

The Validator decides whether a theory's finding stands. It is deterministic Python (`theorylab/validator.py`, called from `theorylab/theory_run.py::run_spec`). No LLM computes a number or changes a verdict. This page describes how the gate works, the review of its permutation scheme (CP3 audit items U7 and U8), and what changed in CP3.

## 1. The gate in one page

| Step | What it does | Setting (`specs/ratings_fairness.json`) |
|---|---|---|
| Population | Full-990 filers with a county and need data, complete in every column the spec uses | `population` |
| H1 | Difference in mean outcome, exposed vs. other rows | `hypotheses[H1]` |
| H2 | Fixed-effects coefficient on the exposure, with numeric controls, absorbed factors and SEs clustered by county | `hypotheses[H2]` |
| p-values | Permutation test for H1 and H2: the exposure is reassigned across counties, and the same statistic is recomputed | `n_permutations`, `seed`, `permutation` |
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

Measured on the real panel (1,303,953 rows, 3,087 counties, 199 permutations):

| | Year-by-year shuffle (TA) | Whole-history shuffle (CP3) |
|---|---|---|
| H1 null SD | 0.0169 | 0.0310 |
| H1 p | **0.010** | **0.135** |
| H2 null SD | 0.0153 | 0.0282 |
| H2 p | 0.095 | 0.440 |

The null is about 1.8 times wider, close to the √5 ≈ 2.2 expected when five persistent years act as one. The ratings verdict does not change: it is unsupported either way, because H1 is far below the floor. But the claim "the gap is significant under the county permutation (q = 0.02)" from the first audit run does not hold. The test `test_year_by_year_shuffle_understates_the_null_when_exposure_persists` reproduces this effect on a synthetic panel.

`keep_paths` is an option of `UnitPermuter`, off by default, and on in `specs/ratings_fairness.json`. Any theory whose exposure persists over years should turn it on. The per-year runs in `theorylab/trends.py` are unaffected, because each tests one year.

**Still open.** Even with whole histories, the H2 permutation null SD (0.028) is smaller than H2's county-clustered SE (0.049). Both say H2 is not significant (p = 0.44 and p = 0.61), so no conclusion depends on which one is used. A plausible cause is that a few very large counties carry much of the weight, which a permutation null and a sandwich SE treat differently. We note this for CP4 rather than resolve it now.

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

## 3. Evidence that the gate works (V2)

`scripts/run_controls.py` → `data/output/controls.json`, on the real population, with 199 permutations and no robustness models:

| Control | Expected | Result |
|---|---|---|
| Negative: exposure replaced by another county's whole history, 20 seeds | unsupported × 20 | **20 / 20 unsupported**; none significant, none above the floor, none replicated |
| Positive: −0.25 SD planted in high-need counties | supported | **supported** (H1 −0.271 SD, H2 −0.261 SD, replicated in 5/5 state folds) |
| Positive: −0.05 SD planted | unsupported (below floor) | **unsupported** (H1 −0.072 SD: significant at q = 0.005, but below the 0.1 SD floor) |

The negative-control placebos keep the real exposure's persistence over years, so a test that ignored persistence would show up here as too many significant H1 gaps. With the whole-history shuffle, none were significant.
