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

## Reproduce the reported results (README workflow)

Everything below was run from a clean clone of the submission commit in an empty folder, with
nothing installed beyond `requirements.txt`. The outputs quoted are from that run.

### Requirements for starting fresh

| Item | Value |
|---|---|
| Python | 3.12 (checked: 3.12.10 on Windows, 3.12.16 on Linux). The pins below are the versions every reported result was produced with |
| Packages | `pip install -r requirements.txt`: pandas 3.1.1, numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0, pyyaml 6.0.3, pytest 9.2.0, anthropic 2.5.0 (live LLM calls only) |
| OS | Checked on Windows 11 (10.0.26200) for the clean-clone run below. The panel hash also matched on macOS (CP4) and Arch Linux (CP5). Plain Python, no OS-specific calls |
| Hardware | Checked on an Intel i7-11800H laptop (8 cores, 16 GB RAM). The panel build peaks at ⟦TBD⟧ GB of memory |
| Disk | About 190 MB for the clone (every committed file is under 20 MB), plus about 400 MB for the rebuilt panel |
| Accounts and keys | **None for the quickstart or any replay.** `ANTHROPIC_API_KEY` only for live Planner/Scout calls; `CENSUS_API_KEY` only to rebuild the need table from the Census API. Both names are in `.env.example`; never commit a key |
| Git LFS | Not used: no committed file is over 50 MB |
| Quickstart runtime | ⟦TBD⟧ |

Install, from a fresh clone:

```bash
git clone https://github.gatech.edu/ojaafar3/norp-finance-theory-project.git norp && cd norp
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Quickstart

The claim it backs: **the ratings-fairness theory (T1) is unsupported under the CP5 gate, with a
high-need gap of −0.022 SD**, and every other committed number in the report is rechecked by
`verify_outputs.py`. No key, and no network after the install.

**Step 1. Unit tests**

```bash
python -m pytest -q
```

Expected last line: `⟦TBD⟧ passed in …s`. The tests use synthetic data and a fake LLM, so they
need neither the panel nor a key.

**Step 2. Recheck the committed results**

```bash
python scripts/verify_outputs.py
```

Each check recomputes something from committed files and compares it with what the report
claims, and prints `[PASS]`, `[FAIL]` or `[SKIP]`. Expected last line:

```
⟦TBD⟧ checks passed -> <path to the clone>/data/output/validation_report.json
```

The part before the arrow is the match. The exit code is 0 only if every check that ran passed.
The slowest check (`a1_unseen_replays`) replays all 40 A1 texts from the committed LLM cache and
fails if any of them would need a live call.

**Step 3. Rebuild the organization-year panel and check its hash**

```bash
python scripts/build_panel.py
python scripts/verify_outputs.py --panel
```

The panel (`data/derived/org_year_panel.csv.gz`, 2,390,744 organization-years × 54 columns,
about 400 MB) is too large to commit and is never downloaded: it is rebuilt from the committed
NCCS extracts, crosswalk and need table. `build_panel.py` rewrites
`data/output/panel_join_log.json`; only `generated_at` should change (`git diff` shows it). The
second command adds a `panel_hash` check that hashes the uncompressed CSV and compares it with
the log **as committed** (git `HEAD`). Expected:

```
[PASS] panel_hash: 2,390,744 rows, sha256_csv 334951b1...
```

Tolerance: none. The hash is of the exact CSV text, written with LF line endings on every OS.

**Step 4. Rerun one theory from its committed spec**

```bash
python scripts/rerun_specs.py --only distressed_county_rating_penalty --to quickstart --permutations 199 --no-by-year
python scripts/review_theory_runs.py --dir data/output/theory_runs/quickstart
```

This reruns the Planner's committed CP4 spec for T1 through the current gate (outcome rule,
studentized permutation test, effect floor, held-out states) with no LLM call, and writes to
`data/output/theory_runs/quickstart/` (git-ignored). Expected output:

```
⟦TBD⟧
```

What must match: the verdict `unsupported`, `rule applied False` (the rating is not
heavy-tailed), and the effects **H1 −0.022 SD and H2 −0.012 SD**, the same as the committed full
run in `data/output/theory_runs/cp5_reruns/distressed_county_rating_penalty.json` to the printed
digits. Effects do not depend on the permutation count; p- and q-values do (199 here, 999 in the
committed run), so they will differ.

### Data provenance

| Data | Rows | Committed or fetched | Source | Used by |
|---|---|---|---|---|
| `data/extracts/nccs/*.csv.gz` (15 files, 7.9–18.6 MB) | Parts I, IX, X for 2018–2022; exact counts in `MANIFEST.json` | Committed, sha256 in `MANIFEST.json` (checked by `verify_outputs.py`) | Column subsets of the NCCS efile tables, `https://nccs-efile.s3.us-east-1.amazonaws.com/public/efile_v2_1/` | `build_panel.py` |
| `data/derived/ein_county_crosswalk.csv.gz` | 612,230 EINs | Committed | NORP `NGOs_with_categories` (Metabase, course login) and the Census 2020 county file | `build_panel.py` |
| `data/derived/county_need.csv` | 16,140 county-years | Committed | ACS 5-year 2018–2022 (Census API), NORP tract tables, County Health Rankings 2020–2025 (`data/extracts/chr/`) | `build_panel.py` |
| `data/derived/org_year_panel.csv.gz` | 2,390,744 | **Rebuilt in Step 3**, git-ignored; CSV sha256 in `panel_join_log.json` | All of the above | every theory run |
| `data/llm_cache/` | one JSON per LLM response | Committed, each with request and response sha256 | Claude `claude-sonnet-5-5` | every `--replay` |
| Raw NCCS CSVs and NORP raw tables | — | Not committed (4.2 GB; NORP needs the course login) | see the Checkpoint 2 section | only to rebuild the committed extracts (`scripts/run_cp2.sh`) |
| Multi-year crime (UCR Return A, openICPSR 100707) | — | **Not in the repo.** It needs an openICPSR account, so it was not added (D9 in `docs/data_decisions.md`) | — | — |

### Longer workflows

Each writes the committed file named, so `git diff` after a run shows what moved. Times are from
the Windows laptop above unless marked. Set `OMP_NUM_THREADS=4` when running several at once.

| Goal (report claim) | Command | Writes | Expected | Time |
|---|---|---|---|---|
| **CP5 benchmark: 20 entries, three metrics** | `python scripts/run_benchmark.py --replay --screen 199 --rerun-q 0.25 --by-year` | `data/output/benchmark_cp5.json`, `data/output/theory_runs/cp5/` | at today's code this reproduces the **post-fix** numbers below (plan validity 20/20, recovery 5/5, false positives 0/5); the committed first pass (17/20, 5/5, 0/3) ran before the population fix | about 42 min |
| Same, written next to the first pass | add `--tag postfix` | `data/output/benchmark_cp5_postfix.json`, `data/output/theory_runs/cp5_postfix/` | 20/20, 5/5, 0/5; the 17 first-pass effects unchanged | about 42 min |
| Review every benchmark verdict | `python scripts/review_theory_runs.py --dir data/output/theory_runs/cp5` | `.../cp5/review.json` | ends `all verdicts follow: True` | < 1 min |
| A1 on 40 unseen texts (replay) | `python scripts/measure_assumption1_benchmark.py --replay --out <scratch>.json` | the file you name | 39 of 40, rows identical to `assumption_1_unseen.json` | < 1 min |
| T1–T5 under the outcome rule | `python scripts/rerun_specs.py` | `data/output/theory_runs/cp5_reruns/` | 5 unsupported; T1 and T4 effects unchanged from CP4 | about 25 min (Linux, 22 cores) |
| Outcome rule on all 10 outcomes | `python scripts/outcome_rule_check.py` | `data/output/outcome_rule_check.json` | 5 clipped, 4 still above 3 | < 1 min |
| Heavy-tailed positive controls | `python scripts/controls_heavy_tail.py` | `data/output/controls_heavy_tail.json` | both supported with the rule, unsupported without | about 35 min (Linux) |
| Placebo families on heavy tails | `python scripts/controls_heavy_tail.py --family` | `data/output/controls_family_heavy_tail.json` | 0 of 16 placebos; planted control survives | about 60 min (Linux) |
| T4 proxy check | `python scripts/t4_followup.py` | `data/output/t4_followup.json` | 90% of the H2 effect kept with population and need index added | about 10 min (Linux) |
| Ratio × need sweep under the rule | `python scripts/run_sweep.py --part main` | `data/output/sweep/main.json`, `data/output/sweep.json` | 28/28 unsupported, largest effect 0.065 SD | about 2.2 h |

**Live LLM runs** (`ANTHROPIC_API_KEY` set, drop `--replay`). A live run calls Claude
`claude-sonnet-5-5` and writes new cache entries. The text will differ, so a re-run should still
satisfy: every spec schema-valid, the gate block identical to `spec.GATE_POLICY` (the Planner
cannot change it), and at least 34 of 40 texts agreeing with the gold answers (A1's 85% line).
Our first pass: 39 of 40, $1.04, 102 calls.

### If something fails

- **`CacheError: no cache entry … (replay-only mode)` from `run_cp4.py --replay` or
  `measure_assumptions_1_2.py --replay`.** Expected. The cache is keyed on the exact prompt, and
  since CP4 the Planner's example spec (`specs/ratings_fairness.json`) and the Scout's column list
  changed, so the CP4 keys no longer match. The CP4 outputs stay as committed. To reproduce CP4's
  theories, use `python scripts/rerun_specs.py`, which reruns their committed specs with no LLM
  call. A CP5 replay breaks the same way if either of those two files is edited.
- **`panel_hash` fails after Step 3.** Check `git status`: an edit to `theorylab/panel.py`,
  `need.py`, `ratios.py` or a committed input changes the panel. If `extracts_match_manifest`
  fails as well, the clone's files differ from the commit; re-clone. `python
  scripts/check_panel_hash.py --out <scratch>.json` hashes each column separately and names the
  ones that differ.
- **`MemoryError`, or the machine stalls in Step 3.** The build holds all five years in memory.
  Close other programs or use a machine with 16 GB.
- **A long run is much slower than the table.** BLAS threads compete when several runs share the
  CPU; set `OMP_NUM_THREADS=4`. A laptop that sleeps pauses the run.
- **`verify_outputs.py` shows `[SKIP] panel_hash`.** The panel is not built yet; run Step 3.
  Skipped checks are never counted as passed.

### How this README maps to the report

The quickstart is the T1 row of the Checkpoint 5 evidence table, and Step 2 rechecks the rest of
that table from committed files. Each longer workflow names the report claim it backs. The
report's "How to Reproduce" section points here.

## Data sources

| Source | What | Where |
|---|---|---|
| NCCS 990 efile, Parts I / IX / X, 2018–2022 | Financials, functional expenses, balance sheet | [nccs.urban.org](https://nccs.urban.org/nccs/catalogs/catalog-efile.html) |
| NORP Metabase | NGO table (3.42M orgs with county), 2022 Part I extract, county economics, poverty, tract-level need | norpp.cc.gatech.edu |
| IRS | Business Master File | irs.gov |

## Checkpoint history

Earlier checkpoints, as they were reported. Commands that no longer run as written are marked.

### Checkpoint 1 evidence

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

### Checkpoint 2: data layer and catalog

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
- `expected_rows` for the NCCS tables are the exact measured count per year (CP4), so
  `build_panel` stops on a changed source file. The counts are also in `MANIFEST.json`.
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

### Checkpoint 3: Executor and Validator

| Piece | Code | Output |
|---|---|---|
| Method library: correlation, fixed effects (absorbed, clustered SEs), multilevel (MixedLM), two-stage group means, lagged outcomes | `theorylab/executor.py` | — |
| Validator: county-level permutation tests, BH-FDR, effect-size floor, held-out-state replication (5 folds; held-out year reported for information), four-verdict rule; review of the permutation scheme and the BH family | `theorylab/validator.py`, `docs/validator.md`, `scripts/permutation_check.py` | `data/output/permutation_check.json` |
| Negative and positive controls through the gate on the real panel | `theorylab/controls.py`, `scripts/run_controls.py` | `data/output/controls.json` |
| Theory spec format and schema check (what the CP4 Planner will write) | `theorylab/spec.py`, `specs/ratings_fairness.json` | — |
| Ratings theory end to end from the hand-written spec (H1, H2, verdict) | `theorylab/theory_run.py`, `scripts/run_theory.py` | `data/output/theory_runs/ratings_fairness.json` |
| Ratings theory for each tax year, and whether the gap shifts over time | `theorylab/trends.py`, `scripts/run_by_year.py` | `data/output/theory_runs/ratings_fairness_by_year.json` |
| Assumption 5: MixedLM fit time and convergence on 2022 full-990 filers | `scripts/measure_assumption5.py` | `data/output/assumption_5.json` |
| Open checks from CP2: Part IX coverage by return type; rating anchor sensitivity | `theorylab/checks.py`, `theorylab/sensitivity.py`, `scripts/part9_coverage.py`, `scripts/anchor_sensitivity.py` | `data/output/part9_coverage.json`, `data/output/anchor_sensitivity.json` |
| Tests on synthetic data with known answers | `tests/test_executor.py`, `tests/test_validator.py`, `tests/test_theory_run.py`, `tests/test_controls.py`, `tests/test_trends.py` | — |

**Result: the ratings-fairness theory is unsupported.** On 1,303,953 full-990 organization-years the high-need gap is −0.022 SD (q = 0.268), a fifth of the effect floor. It fails the gate in every tax year and under all 10 anchor variants, and it does not replicate across held-out states. The same gate recovers a planted 0.25 SD gap and rejects 20 of 20 placebos (`data/output/controls.json`).

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

### Checkpoint 4: agents

> **Built on the TA's `ai-suggestions/cp4` code.** Commit `7d9ee8a` adds it as provided. Our fixes and additions are in the commits after it.

| Piece | Code | Output |
|---|---|---|
| Planner and Data Scout agents, locked gate, hash-checked LLM cache | `theorylab/agents.py`, `theorylab/llm.py`, `theorylab/spec.py` | `data/llm_cache/` |
| Orchestrator: theory → Scout → Planner → spec → gate → drill-down, per-year results and trend for every theory | `theorylab/orchestrator.py`, `theorylab/cli.py`, `theorylab/trends.py` | `data/output/theory_runs/cp4/` |
| Five named theories with gold answers | `benchmark/cp4_theories.json`, `theorylab/benchmark.py`, `scripts/run_cp4.py` | `summary.json`, `<theory_id>.json`, `<theory_id>.spec.json` |
| Mechanical review of every verdict | `scripts/review_theory_runs.py` | `data/output/theory_runs/cp4/review.json` |
| Assumptions 1 and 2, first measurement | `scripts/measure_assumptions_1_2.py` | `data/output/assumptions_1_2_first_pass.json`, `data/output/assumptions_1_2.json`, `data/output/run_times.json` |
| Outlier check on each theory's outcome | `scripts/outcome_tails.py` | `data/output/outcome_tails.json` |
| Ratio × need sweep with the TA before/after | `theorylab/sweep.py`, `scripts/run_sweep.py` | `data/output/sweep.json`, `data/output/sweep/` |
| County Health Rankings columns (health, education, crime) | `theorylab/need.py`, `scripts/build_county_need.py`, D8 in `docs/data_decisions.md` | `data/extracts/chr/`, `data/output/county_need_report.json` |
| Studentized permutation gate (the H2 null/SE gap) | `theorylab/validator.py`, `scripts/h2_null_check.py`, `docs/validator.md` 2.5 | `data/output/h2_null_check.json` |
| BH family across theories, family-level controls | `theorylab/gate.py`, `scripts/run_controls.py --family` | `data/output/controls_family.json` |
| One-level drill-down tested on planted strata | `theorylab/drilldown.py`, `scripts/drilldown_check.py` | `data/output/drilldown_check.json` |
| Panel rebuild checked across two machines | `scripts/check_panel_hash.py` | `data/output/panel_rebuild_check.json`, `data/output/panel_rebuild_check_omar.json` |

**Result: all five theories are unsupported, and three of those nulls say very little.** T1 (ratings) and T4 (crime × admin share) are real nulls on well-behaved outcomes. T4 is significant and replicates in all 5 state folds, but at +0.027 SD it is far below the floor. T2, T3 and T5 use ratios whose SD is set by a tiny share of rows (over 99.9% of the variance in the top 0.1% of rows, `outcome_tails.json`), so neither the floor nor Pearson r can detect anything on them yet. Fixing that is the first CP5 item.

**Sweep.** 4 rating ratios × 7 need measures, one BH family, 999 permutations: all 28 cells unsupported, largest effect 0.040 SD. 7 cells are significant after the correction, all on program or administration share; the 14 cells on heavy-tailed ratios show nothing, and the same limit applies to them. On the TA's 40-cell grid, 21 cells are significant as the TA wrote it and 12 on the fixed gate (`data/output/sweep.json`).

Assumption 1 was **8 of 10 at first** (7 usable), below the 85% needed; after two fixes to the schema check and the retry prompt, the same texts give 9 of 10. A real test needs new theories (CP5).

To reproduce (as reported at CP4; two replay lines are now broken, marked):

```bash
pip install -r requirements.txt
python -m pytest -q                      # expect 164 passed at the CP4 commit
python scripts/build_panel.py            # sha256_csv must equal panel.sha256_csv in the join log
python scripts/check_panel_hash.py --out data/output/panel_rebuild_check_mine.json
python scripts/measure_assumptions_1_2.py --replay   # NO LONGER REPLAYS since f825052 (see "If something fails"); output stays as committed
python scripts/run_cp4.py --replay       # NO LONGER REPLAYS; use: python scripts/rerun_specs.py --to <scratch dir>
python scripts/review_theory_runs.py     # must end: all verdicts follow: True; all on the current gate: True
python scripts/outcome_tails.py
python scripts/run_sweep.py --part main --part ta_after   # about 1 h 50 min on a laptop; ta_before is kept as committed
python scripts/h2_null_check.py
python scripts/run_controls.py --family
python scripts/drilldown_check.py
python scripts/build_county_need.py --chr-only --chr-dir data/raw/chr   # needs the raw CHR files
```

Without `--replay`, the agents need `ANTHROPIC_API_KEY` (model `claude-sonnet-5-5`, no fallback model). The gate settings come from `spec.GATE_POLICY`, and the Planner cannot change them.

### Checkpoint 5: benchmark

> **Built on the TA's `ai-suggestions/cp5` code.** Commit `f825052` adds it as provided. Our fixes and additions are in the commits after it.

| Piece | Code | Output |
|---|---|---|
| Heavy-tail outcome rule (bounded outcomes exempt, pooled 1/99 clip, one clip) | `theorylab/outcomes.py`, `docs/validator.md` 2.8, `scripts/outcome_rule_check.py` | `data/output/outcome_rule_check.json` |
| Positive controls planted raw in heavy-tailed ratios; placebo families | `theorylab/benchmark_run.py`, `scripts/controls_heavy_tail.py` | `data/output/controls_heavy_tail.json`, `data/output/controls_family_heavy_tail.json` |
| T1–T5 rerun under the rule from their committed specs | `scripts/rerun_specs.py` | `data/output/theory_runs/cp5_reruns/` |
| 10 literature theories with checked sources, 5 positive and 5 negative controls, gold answers frozen before any run | `benchmark/cp5_theories.json`, `nonprofit_density` in `theorylab/need.py` (D9) | — |
| Benchmark runner: two BH families, crashes count as invalid plans, recovery needs a gold-matching spec, per-year results | `theorylab/benchmark_run.py`, `scripts/run_benchmark.py`, `docs/validator.md` 2.9 | `data/output/benchmark_cp5.json` (first pass), `data/output/benchmark_cp5_postfix.json` |
| Review of every benchmark verdict | `scripts/review_theory_runs.py` | `data/output/theory_runs/cp5*/review.json` |
| A1 on 40 unseen texts, failure analysis | `scripts/measure_assumption1_benchmark.py`, `docs/assumption_1_unseen.md` | `data/output/assumption_1_unseen.json` |
| T4 proxy check | `scripts/t4_followup.py` | `data/output/t4_followup.json` |
| Sweep under the rule | `scripts/run_sweep.py --part main` | `data/output/sweep.json`, `data/output/sweep/` |
| Recheck of every reported headline | `scripts/verify_outputs.py` | `data/output/validation_report.json` |

**Result: the gate works on every control, and no literature theory is supported.**

- **Benchmark, first pass:** plan validity **17/20** (3 specs crashed on H2 controls with missing values), control recovery **5/5**, false positives **0/3**. After the fix that requires controls to be present (separate file): **20/20, 5/5, 0/5**.
- **Literature:** all 10 unsupported. Seven are significant after the family correction, all between 0.016 and 0.066 SD, under the 0.1 SD floor. Three point the way their theory says and replicate in all 5 held-out state folds (donor competition → fundraising share, unemployment → lower margins, housing burden → more debt). Three run the other way in every year (richer or more educated counties have more expensive fundraising and more concentrated revenue).
- **A1:** 39 of 40 unseen texts agree with gold on the first pass; 17 of the 20 benchmark specs were usable. **A2:** 38 min for the benchmark with the 199 screen, $0.51 of API calls.
- **Outcome rule:** a 0.25 SD effect planted in a raw heavy-tailed ratio is invisible without the rule (−0.001 SD) and recovered at full size with it (−0.253 SD); 0 of 16 placebos pass.
