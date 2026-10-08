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
                                  confounder controls · held-out-year replication
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

| Piece | Code | Output |
|---|---|---|
| Data catalog with contracts (keys, fields, expected rows) | `catalog/catalog.yaml`, `theorylab/catalog.py` | — |
| Slim NCCS extracts, Parts I/IX/X 2018–2022, hash manifest (Assumption 3) | `scripts/build_nccs_extracts.py` | `data/extracts/nccs/*.csv.gz`, `data/extracts/nccs/MANIFEST.json` |
| EIN → county crosswalk incl. FL and CT, per-year match rate (Assumption 4) | `theorylab/crosswalk.py`, `scripts/build_crosswalk.py` | `data/derived/ein_county_crosswalk.csv.gz`, `data/output/crosswalk_report.json` |
| County need table (ACS 5-year 2018–2022) | `theorylab/need.py`, `scripts/build_county_need.py` | `data/derived/county_need.csv`, `data/output/county_need_report.json` |
| Organization-year panel with join log | `theorylab/panel.py`, `scripts/build_panel.py` | `data/derived/org_year_panel.csv.gz` (git-ignored), `data/output/panel_join_log.json` |
| Financial ratio library | `theorylab/ratios.py` | — |
| Unit tests | `tests/` | — |

Build order:

```bash
pip install -r requirements.txt
pytest                                                              # unit tests, no data needed
python scripts/build_nccs_extracts.py                               # ~4 GB download into data/raw/nccs (ignored)
python scripts/build_crosswalk.py --norp-raw-dir <path to NORP data/raw>
python scripts/build_county_need.py                                 # Census API; set CENSUS_API_KEY if rate-limited
python scripts/build_panel.py
```

Notes:

- Every statistic and ratio is computed by pandas. The catalog lists the only fields
  the Data Scout (CP4) may request.
- NCCS columns marked `verified` in the catalog were confirmed by the CP1 probe. The
  others use the same names as the NORP 2022 extract (`F9 01 Rev Contr Tot Cy` →
  `F9_01_REV_CONTR_TOT_CY`). The extract script stops and prints the real header if
  any are missing.
- `expected_rows` for the NCCS tables are sanity bounds. Tighten them to the counts in
  `MANIFEST.json` after the first extract run.
- County names resolve against the Census 2020 county file, not the NORP
  `county_fips_lookup` table, which has no Florida rows. Connecticut legacy counties map to
  09001–09015 and planning regions to 09110–09190. The 2022 ACS reports Connecticut by
  planning region, so 2022 need joins for legacy-county Connecticut EINs show up as misses
  in `panel_join_log.json` rather than being silently dropped.
