# Assumption 1 on unseen theories (CP5 O3): failure analysis and H2 controls

Source: `data/output/assumption_1_unseen.json` (first pass, live, `claude-sonnet-5-5`,
2026-11-19, committed in `ba10cf7` together with its LLM cache). No prompt or code was changed
before that run, and none has been changed since. `python scripts/verify_outputs.py` recomputes
the rate from the rows and replays every row from the cache with no API call.

## Result

| | Texts | Schema-valid | Agree with gold |
|---|---|---|---|
| Literature | 20 | 20 | 19 |
| Positive controls | 10 | 10 | 10 |
| Negative controls | 10 | 10 | 10 |
| **All** | **40** | **40** | **39 (97.5%)** |

A1 needs 85%. Cost $1.04 for 102 calls (Scout + Planner for 40 texts = 80, so 22 were
second Planner attempts: a schema rejection or a Scout/Planner disagreement sends the Planner
back once with the errors). The retries are part of the first pass; the number is what the
pipeline returns, not what a single call returns. No final spec asked to loosen the gate.

**Why 22 of 40 texts needed a second attempt (read from the retry prompts in the cache).** All
22 were rejected for the same reason: the Scout mapped a continuous exposure, and the Planner's
first spec said `exposure_type: binary` with a difference in means. The other two of the 22
also set `exposure_varies_by` wrongly. The Planner prompt shows one example spec, and it is
for a binary exposure (`specs/ratings_fairness.json`); the instruction for continuous
exposures is one sentence after it. On first attempt the Planner copied the example. The
Scout agreement check (CP4) caught every case, and every retry fixed it, so A1's pass rate
does not show this, but without that check 22 of 40 specs would have run the wrong test.
Fix for CP6: a second, continuous example spec in the prompt. Not changed in CP5, because
changing the prompt changes every cache key (the reason CP4's replay broke).

## The one failure

| Entry | Variant | What the Planner did | Gold | Kind of error | Would a rule catch it? |
|---|---|---|---|---|---|
| `ratings_fairness` | paraphrase | exposure `need_index`, continuous | `high_need`, binary | operationalization (right construct, different variable and type) | No. The paraphrase ("the ones based in the worst-off local economies") drops the "most distressed quarter" in the original text, so a continuous need index is a fair reading. The schema cannot tell which of two valid exposures a sentence meant. |

Variable, sign, unit and outcome were right in all 40 texts. The miss is partly the
benchmark's: the paraphrase is vaguer than its gold answer. We count it anyway, because the
gold answers were frozen before the run.

## The H2 controls the Planner chose (CP4 open question)

The gold review does not check H2 controls, so these were read by hand. Every spec controls
for `log_revenue` (size) and absorbs tax year, category and state. On top of that the Planner
added 2.7 controls per spec on average. Three things stand out:

1. **An outcome used as a control (8 of 40 specs).** `revenue_concentration` is a catalog
   outcome (`role: outcome`) but appears as an H2 control in 8 specs, including the benchmark
   (text) specs for `lit_income_fundraising`, `lit_density_fundraising`,
   `lit_uninsured_programs`, `lit_education_fundraising` and `pos_poverty_programs`. It is
   measured on the same organization-year as the outcome, so it can be a consequence of the
   exposure (a bad control). A schema rule would catch this: H2 controls must not have
   `role: outcome`. **Not added in CP5**, because the rule would change the prompts after the
   first pass; logged for CP6.
2. **A likely mediator as a control.** `lit_income_concentration` controls for
   `nonprofit_density` in both variants. Its own source (Grønbjerg & Paarlberg 2001) argues
   richer places have more nonprofit resources, which is the path from income to more revenue
   sources, so this control can absorb the mechanism H2 is meant to keep. Density appears as a
   control in 15 of 40 specs.
3. **Controls are not stable across wording.** For 13 of 20 entries the text and the
   paraphrase got different H2 control sets, although both passed the gold review. H1 has no
   controls, so H1 is unaffected; H2 results depend on wording more than A1's pass rate shows.

None of these change a verdict on its own (the gate needs H1 and H2 above the floor), but they
are a reason to treat H2 as the Planner's choice, not a fixed test. Planned fix for CP6: a
schema rule against outcome-role controls, and a fixed default control set the Planner may
only add to with a stated reason.
