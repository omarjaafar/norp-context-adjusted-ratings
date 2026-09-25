# NORP Context-Adjusted Nonprofit Ratings

CS 4220/6235, Fall 2026, Group 17. Omar Jaafar and Vien Tran.

**Question:** Do standard financial-ratio ratings of nonprofits systematically penalize organizations that operate in high-need communities?

Common rating approaches score nonprofits on ratios like fundraising cost share, program revenue share, and reserves. The NORP research statement argues these ratios ignore context: a food bank in a high-poverty rural county faces a harder environment than one in a wealthy suburb. This project measures how much that matters.

## Plan (summary)

This is a **human-directed agentic hypothesis-testing pipeline**, following NORP's Fall 2026 direction. We supply the theory, agents plan the tests and drill into the results, and a deterministic statistical gate decides what counts.

```
hypothesis ─► Planner (LLM) ─► test specs (JSON, schema-checked)
                                   │
                                   ▼
              Executor (Python): naive ratios · multilevel model · value-added · rank shift
                                   │
                                   ▼
              Validator (Python): state-stratified permutation · BH-FDR · effect-size floor
                                   │
                    survivors ─► drill-down (state / sector / size) ─► back to Executor
                                   │
                                   ▼
              Findings report (LLM narrates Python numbers only; failures reported too)
```

1. **Naive rating.** Score each nonprofit on standard ratio metrics built from its IRS Form 990 Part I, which is the input a ratio-based rater would use.
2. **Context-adjusted rating.** Fit a multilevel model: organizations nested in counties, counties nested in states. It predicts each organization's expected performance from county conditions (poverty, income, unemployment, housing burden, food desert share). Value-added = observed − expected.
3. **Rank-shift test.** Test whether organizations in high-need counties move up systematically after adjustment. A null result is a valid finding.
4. **Planner and drill-down.** The Planner agent expands the hypothesis into variations: by sector, by ratio, by need dimension, and by mission type. It can only reference real columns, and it can never compute or override a statistic.
5. **Mission-type classification.** An LLM labels each 990 mission statement as *direct service* or *systemic/advocacy*. We validate the labels against a hand-labeled sample.

## Data

All inputs come from the NORP structured project (NORP Metabase, `norpp.cc.gatech.edu`):

| File | Rows | Role |
|---|---|---|
| `F9_P01_T00_SUMMARY_2022.csv` | 131,587 | IRS 990 / 990EZ / 990PF Part I summary financials and mission text |
| `NGOs_with_categories` (4 gzip parts) | 3,420,024 | EIN → county, state, NTEE category |
| `nccs_crosswalk_economic.csv` | 3,142 | County income, poverty, unemployment |
| `disadvantaged_communities.csv` | 72,742 | Tract-level housing burden, food desert, DAC status |
| `Poverty_Rates_2023.csv` | 2,998 | County poverty rate |
| `county_fips_lookup.csv` | 3,076 | County name → FIPS |

Raw data is not committed yet. Committing it (within GitHub's 100 MB file limit) is a Checkpoint 2 milestone.

## Checkpoint 1 feasibility evidence

`scripts/feasibility_check.py` produced [`data/output/feasibility_report.json`](data/output/feasibility_report.json). Key numbers:

- 127,477 of 131,027 unique 990 EINs (97.3%) match the NGO table and therefore a county.
- 57,116 of the matched organizations are full-990 filers. Nearly every Part I field is present for them, and 57,091 have mission text.
- Matched filers cover 3,032 distinct counties (median 12 filers per county, 1,719 counties with 10 or more), which is enough for a county-level random effect.
- 990EZ filers lack fundraising expense, contributions, and mission text in this extract, so the core analysis uses full-990 filers.

To reproduce:

```bash
pip install -r requirements.txt
python scripts/feasibility_check.py --data-dir <path to NORP data/raw>
```

## Layout

```
scripts/feasibility_check.py      # CP1: data coverage and join check
data/output/feasibility_report.json
```
