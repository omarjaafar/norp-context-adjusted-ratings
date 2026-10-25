# The ratings-fairness theory

This is the starting theory for the pipeline, and the CP1 feedback asked us to define it. This page says what the "standard rating" is, how it is scored, and what the theory claims in a form the pipeline can test.

**The theory.** Standard financial-ratio ratings unfairly penalize nonprofits in high-need counties.

## 1. The standard rating

The standard rating is a score from 0 to 10 built from four financial ratios. It looks only at an organization's own Form 990 numbers. It never looks at where the organization is or whom it serves. That blindness to context is the thing the theory puts on trial.

Code: `theorylab/ratios.py::standard_rating`. `add_ratios` adds it to the panel as `standard_rating`, with `rating_components` giving how many ratios went into it.

### The four ratios

| Ratio | Formula | Form 990 source | Better when |
|---|---|---|---|
| Program expense ratio | program expenses ÷ total expenses | Part IX | higher |
| Administrative expense ratio | management and general expenses ÷ total expenses | Part IX | lower |
| Fundraising efficiency | fundraising expenses ÷ contributions | Part IX, Part I | lower |
| Liabilities to assets | total liabilities ÷ total assets | Part X | lower |

These are the ratios that charity raters have built their financial scores on: three measures of how much spending reaches programs, and one measure of solvency. All four were already in the ratio library.

### How each ratio is scored

Each ratio gets a score from 0 to 10 against two fixed anchors. At or past the "zero" anchor it scores 0. At or past the "full" anchor it scores 10. In between, the score is a straight line.

| Ratio | Scores 0 at | Scores 10 at | Where the zero anchor comes from |
|---|---|---|---|
| Program expense ratio | 65% or less | 85% or more | BBB Wise Giving Alliance Standard 8: spend at least 65% of expenses on programs |
| Administrative expense ratio | 35% or more | 15% or less | What is left when programs get only 65% |
| Fundraising efficiency | $0.35 or more per dollar raised | $0.10 or less | BBB Wise Giving Alliance Standard 9: spend no more than 35% of related contributions on fundraising |
| Liabilities to assets | 1.0 or more | 0 | Liabilities equal to assets means no net assets |

For example, a program expense ratio of 75% is halfway between 65% and 85%, so it scores 5.

**Why fixed thresholds and not percentiles.** Raters use absolute cut-offs: a charity is judged against a line, not against its neighbors. Percentile scores would also change whenever the set of organizations in the panel changes, so the same organization could get a different rating for reasons that have nothing to do with it.

**The "full marks" anchors are our choice.** The zero anchors come from published standards. The full anchors (85%, 15%, $0.10, 0) are our judgment of what raters treat as strong. They are parameters (`RATING_ANCHORS`), and `standard_rating` accepts other values, so CP3 can check whether a result survives different anchors.

### How the scores combine

The rating is the plain average of the ratio scores, with equal weights.

- **At least three of the four ratios must be defined.** Otherwise the rating is missing.
- **The usual missing ratio is fundraising efficiency.** It is undefined for an organization that reports no contributions, because there is nothing to divide by. Those organizations are rated on the other three.
- **`rating_components`** records whether a rating rests on three ratios or four, so an analysis can restrict to four.

### Why we build it ourselves

- **We do not have a rater's scores.** No rater's historical scores are in our data sources.
- **Rater scores cover a narrow, non-random set.** Raters score mostly large public charities. Our question is about organizations in high-need counties, many of which a rater never scores.
- **A rater's score is partly a black box.** Published methods change over time and include judgment. If our pipeline found a penalty, we could not say which part of the score caused it.
- **Ours is reproducible.** Every rating traces to four Form 990 fields and four pairs of anchors, and it can be computed for every full Form 990 filer in every year.

The cost is that our rating is a stand-in. A result about it is a result about ratings of this kind, not about any named rater.

## 2. What the rating looks like on our data

Numbers are from `data/output/ratio_audit.json`, built by `scripts/ratio_audit.py` from the committed NCCS extracts. The scope is full Form 990 filers; Form 990-EZ has no Part IX or Part X.

| Tax year | Full 990 filings | Rated | On 4 ratios | On 3 ratios | Median rating |
|---|---|---|---|---|---|
| 2018 | 263,339 | 87.2% | 191,277 | 38,331 | 8.77 |
| 2019 | 280,154 | 87.3% | 204,360 | 40,103 | 8.71 |
| 2020 | 315,274 | 86.9% | 231,593 | 42,319 | 8.75 |
| 2021 | 330,682 | 86.9% | 244,434 | 43,001 | 8.90 |
| 2022 | 341,854 | 87.1% | 251,611 | 46,073 | 8.90 |

Three things to know before using it:

- **Most organizations score high.** The median is near 8.8 and at least one in ten scores a perfect 10. In 2022, 54% of scored organizations get full marks on the program ratio. Real rater scales behave the same way, but it means the rating separates weak organizations better than it separates strong ones.
- **About 13% are unrated.** Most of these did not fill in the functional expense columns of Part IX, which only 501(c)(3) and 501(c)(4) organizations must complete. In 2022, 11.4% of full 990 filings report expenses but no usable split.
- **The TA formulas would have mis-scored some of them.** 16,815 filings in 2022 enter zeros in all three functional columns. The TA code read that as a 0% program ratio, the worst possible score. We now treat it as missing. This and the other formula checks are described at the top of `theorylab/ratios.py`.

## 3. What the theory claims, in testable form

Each term in the theory maps to a column in the panel.

| Term | Meaning in our data |
|---|---|
| Standard rating | `standard_rating`, as defined above |
| High-need county | `high_need`: the top quarter of counties on the ACS need index in that year (`docs/data_decisions.md`, D4) |
| Penalize | Organizations in high-need counties have lower ratings |
| Unfairly | The gap remains after accounting for differences a rating may fairly reflect, such as size and sector |

That gives two hypotheses for the Planner to work from:

- **H1, the gap exists.** The mean standard rating is lower in high-need counties than elsewhere.
- **H2, the gap is not explained away.** The gap in H1 remains after controlling for organization size (revenue), sector (the NORP category) and tax year.

How the Validator's four verdicts would read:

| Verdict | What we would see |
|---|---|
| Supported | H1 and H2 both hold |
| Weak | H1 holds, but the gap shrinks to near zero or loses significance under H2 |
| Unsupported | No gap in H1 |
| Reversed | Organizations in high-need counties have higher ratings |

The statistical tests and thresholds belong to the Executor and Validator in CP3. This section fixes only what is being compared.

**The verdict does not depend on the rating anchors (CP3).** `scripts/anchor_sensitivity.py` reruns the theory with each "full marks" anchor moved one at a time, plus a rating that requires all four ratios (10 variants, 199 permutations each, whole county histories shuffled, 5-fold held-out states). Every variant is **unsupported**, the same as the base rating. The H1 gap runs from −0.011 to −0.030 SD, a third or less of the 0.1 SD floor. Only one variant (fundraising full marks at 20 cents per dollar) reaches q < 0.05 on H1, and none replicates across held-out states. Source: `data/output/anchor_sensitivity.json`.

## 4. Limits

- **Need changes slowly.** The need index is a five-year average, so the theory is tested across counties, not within a county over time (D4).
- **Connecticut and need.** Connecticut rows for 2018–2021 use rebuilt planning-region estimates (D5).
- **Full Form 990 filers only.** Small organizations that file Form 990-EZ are not rated, and they may be more common in high-need counties. A finding applies to full filers.
- **The anchors are ours.** Section 1 explains which are borrowed and which are judgment.
