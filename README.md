# NORP Finance Theory Lab

CS 4220/6235, Fall 2026, Group 17. Omar Jaafar and Vien Tran.

A **human-directed theory-testing pipeline for nonprofit finance**, built for the NORP (Non-profit Organization Research Panel) structured project. A researcher states a theory in plain language. The pipeline finds the data, tests the theory, drills into what holds up, and reports what survives a deterministic statistical gate, including theories that fail or come back reversed.

**Starting theory:** do standard financial-ratio ratings unfairly penalize nonprofits in high-need counties? It is one of about 20 theories the pipeline will be evaluated on.

## Design

```
"theory" ─► Planner (LLM) ─► hypotheses as schema-checked JSON
                                   │
                                   ▼
            Data Scout (LLM) ─► fields from the data catalog (contracts enforced)
                                   │
                                   ▼
            Executor (Python) ─► panel + method library
                                 (correlation · fixed effects · multilevel · lagged outcomes)
                                   │
                                   ▼
            Validator (Python) ─► permutation · BH-FDR · effect-size floor ·
                                  confounder controls · held-out-state replication
                                   │  verdict: supported / weak / unsupported / reversed
                    survivors ─► drill-down (state / sector / size) ─► Executor
                                   │
                                   ▼
            Reporter ─► findings (LLM narrates Python-computed numbers only)
```

The LLM never computes a statistic and cannot override the Validator.

## Deliverables (end of semester)

1. **Theory Lab pipeline** as shown above, runnable as `theorylab test "<theory>"`.
2. **Theory benchmark:** about 20 theories: the ratings theory, theories from nonprofit finance research, positive controls (known answers), and negative controls (placebos). We report the plan validity rate, the control recovery rate, and the false-positive rate.
3. **Reproducibility:**
   - Offline replay from committed extracts and a hash-checked LLM cache.
   - A live re-acquisition path.
   - A verification script.

## Data

| Source | What | Where |
|---|---|---|
| NCCS 990 efile, Parts I / IX / X, 2018–2022 | Financials, functional expenses, balance sheet | [nccs.urban.org](https://nccs.urban.org/nccs/catalogs/catalog-efile.html) |
| NORP Metabase | NGO table (3.42M orgs with county), 2022 Part I extract, county economics, poverty, tract-level need | norpp.cc.gatech.edu |
| IRS | Business Master File | irs.gov |

## Checkpoint 1 evidence

- `scripts/feasibility_check.py` → [`data/output/feasibility_report.json`](data/output/feasibility_report.json)
  - 127,477 of 131,027 unique 2022 990 EINs (97.3%) join to a county through the NORP NGO table.
  - 57,116 of them are full-990 filers across 3,032 counties.
- `scripts/probe_nccs_efile.py` → [`data/output/nccs_efile_probe.json`](data/output/nccs_efile_probe.json)
  - NCCS Parts I, IX, and X exist for every year 2018–2022 with the columns we need.
  - Files are 230–324 MB per table-year, so CP2 commits column-subset extracts.

To reproduce:

```bash
pip install -r requirements.txt
python scripts/probe_nccs_efile.py                                  # needs internet only
python scripts/feasibility_check.py --data-dir <path to NORP data/raw>
```

## Checkpoint 2: data layer and catalog

> **Built on the TA's `ai-suggestions/cp2` code.** The commit "Import TA ai-suggestions/cp2 branch unchanged" adds it as provided. Our fixes and additions are in the commits after it.

| Piece | Code | Output |
|---|---|---|
| Data catalog with contracts (keys, fields, expected rows) | `catalog/catalog.yaml`, `theorylab/catalog.py` | — |
| Slim NCCS extracts, Parts I/IX/X 2018–2022, hash manifest (Assumption 3) | `scripts/build_nccs_extracts.py` | `data/extracts/nccs/*.csv.gz`, `data/extracts/nccs/MANIFEST.json` |
| EIN → county crosswalk incl. FL and CT, per-year match rate (Assumption 4) | `theorylab/crosswalk.py`, `scripts/build_crosswalk.py` | `data/derived/ein_county_crosswalk.csv.gz`, `data/output/crosswalk_report.json` |
| County need table (ACS 5-year 2018–2022, Connecticut planning regions rebuilt for 2018–2021, NORP tract-level housing burden and food desert share) | `theorylab/need.py`, `scripts/build_county_need.py` | `data/derived/county_need.csv`, `data/output/county_need_report.json` |
| Organization-year panel with join log | `theorylab/panel.py`, `scripts/build_panel.py` | `data/derived/org_year_panel.csv.gz` (git-ignored), `data/output/panel_join_log.json` |
| Financial ratio library, audited, with the standard rating (defined in `docs/ratings_theory.md`) | `theorylab/ratios.py`, `scripts/ratio_audit.py` | `data/output/ratio_audit.json` |
| Unit tests | `tests/` | — |

Build order:

```bash
pip install -r requirements.txt
pytest                                                              # unit tests, no data needed
python scripts/build_nccs_extracts.py                               # ~4 GB download into data/raw/nccs (ignored)
python scripts/build_crosswalk.py --norp-raw-dir <path to NORP data/raw>
python scripts/build_county_need.py --norp-raw-dir <path to NORP data/raw>   # Census API; needs CENSUS_API_KEY
python scripts/build_panel.py
python scripts/ratio_audit.py                                       # ratio and rating coverage from the committed extracts
```

Notes:

- Every statistic and ratio is computed by pandas. The catalog lists the only fields
  the Data Scout (CP4) may request.
- Every NCCS column the catalog uses is listed under `verified` and was confirmed in the
  2018–2022 headers by `scripts/probe_nccs_efile.py` (`data/output/nccs_efile_probe.json`).
  The extract script still stops and prints the real header if any column is missing.
- `expected_rows` for the NCCS tables are wide sanity bounds (100k–1M rows per file). The
  measured counts, 364,285 to 553,554 rows per file, are in `MANIFEST.json`.
- No raw data is needed to inspect results or rebuild the panel: the extracts, crosswalk
  and need table are committed, so `python scripts/build_panel.py` runs on a bare clone.
  `scripts/run_cp2.sh` rebuilds everything from the raw sources.
- One filing per organization-year: the latest `RETURN_TIME_STAMP` wins (D1 in
  `docs/data_decisions.md`).
- County names resolve against the Census 2020 county file, not the NORP
  `county_fips_lookup` table, which has no Florida rows. Connecticut legacy counties map to
  09001–09015 and planning regions to 09110–09190. The NORP NGO table names Connecticut
  organizations by planning region, which the ACS publishes only from 2022, so the need
  table rebuilds the planning regions for 2018–2021 from town-level ACS data (marked
  `acs_estimated`). See D5 in `docs/data_decisions.md`.

## Checkpoint 3: Executor and Validator

| Piece | Code | Output |
|---|---|---|
| Method library: correlation, fixed effects (absorbed, clustered SEs), multilevel (MixedLM), two-stage group means, lagged outcomes | `theorylab/executor.py` | — |
| Validator: county-level permutation tests, BH-FDR, effect-size floor, held-out-state replication (5 folds; held-out year reported for information), four-verdict rule; review of the permutation scheme and the BH family | `theorylab/validator.py`, `docs/validator.md`, `scripts/permutation_check.py` | `data/output/permutation_check.json` |
| Negative and positive controls through the gate on the real panel | `theorylab/controls.py`, `scripts/run_controls.py` | `data/output/controls.json` |
| Theory spec format and schema check (what the CP4 Planner will write) | `theorylab/spec.py`, `specs/ratings_fairness.json` | — |
| Ratings theory end to end from the hand-written spec (H1, H2, verdict) | `theorylab/theory_run.py`, `scripts/run_theory.py` | `data/output/theory_runs/ratings_fairness.json` |
| Assumption 5: MixedLM fit time and convergence on 2022 full-990 filers | `scripts/measure_assumption5.py` | `data/output/assumption_5.json` |
| Open checks from CP2: Part IX coverage by return type; rating anchor sensitivity | `theorylab/checks.py`, `theorylab/sensitivity.py`, `scripts/part9_coverage.py`, `scripts/anchor_sensitivity.py` | `data/output/part9_coverage.json`, `data/output/anchor_sensitivity.json` |
| Tests on synthetic data with known answers | `tests/test_executor.py`, `tests/test_validator.py`, `tests/test_theory_run.py`, `tests/test_controls.py` | — |

```bash
pip install -r requirements.txt          # adds scipy and statsmodels
bash scripts/run_cp3.sh                  # QUICK=1 for 199 permutations
```

How the gate reads (`theorylab/validator.py::decide_verdict`):

- **H1** (the gap exists): difference in mean rating, high-need vs other counties. Its
  p-value comes from giving each county another county's whole `high_need` history
  (the exposed count per year stays exact), so organizations in the same county are never
  treated as independent draws, and need's persistence over years is kept. See
  `docs/validator.md`.
- **H2** (the gap survives size, sector and year): fixed-effects regression of the rating
  on `high_need` + log revenue, absorbing tax year and NORP category, SEs clustered by
  county; its p-value comes from the same county-level permutation.
- H1 and H2 p-values go through Benjamini-Hochberg; each must also clear the effect-size
  floor (0.1 SD of the rating). H2 must replicate in held-out states: 5 folds of states,
  the expected sign in every fold and a significant pooled held-out estimate.
- **supported** = H1 and H2 hold and H2 replicates; **weak** = H1 holds only;
  **unsupported** = no H1 gap; **reversed** = the gap runs the other way.
- Multilevel and county-year-mean models are reported next to the verdict as robustness
  checks and never change it. Every theory run restricts to full Form 990 filers
  (`return_type == "990"`), which settles which rows the Part IX ratios come from.
