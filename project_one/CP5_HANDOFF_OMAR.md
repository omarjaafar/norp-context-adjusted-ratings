# CP5 handoff: Vien → Omar (2026-11-19)

All my items (V1–V6) are done and committed, and the V7 review tool is ready. V7's last step needs your benchmark run. What's left is your O-items plus pasting the V7 review lines. This file lists them in the order to run them, with the commands. My report sections are in `project_one/CP5_REPORT_VIEN.md`.

## 1. What I committed (after `466f21a`)

| Commit | Item | Output |
|---|---|---|
| `0ab388e` | V1 outcome rule: D-a bounded shares exempt (read from `bounded: [0, 1]` in `panel_columns.yaml`), D-b pooled clip, D-c one clip + `still_above_threshold_after` | `data/output/outcome_rule_check.json` |
| `1dc763e` | V3 theory file **frozen** with checked sources; new column `nonprofit_density`; panel rebuilt | `benchmark/cp5_theories.json`, `panel_join_log.json` |
| `b287b40` | V7 tool: `review_theory_runs.py` reviews a benchmark folder in two families | - |
| `c2bed6f` | V2: `prepare_population` plants raw at 0.25 × post-rule SD; heavy-tail controls | `data/output/controls_heavy_tail.json` |
| `4e67642` | V5: crime roadblock (D9) + T4 proxy check | `data/output/t4_followup.json` |
| `c9c5a8e` | V6: placebo families on two heavy-tailed outcomes | `data/output/controls_family_heavy_tail.json` |
| `fd57bee` + next | V4: T1–T5 rerun under the rule from the committed CP4 specs (`scripts/rerun_specs.py`). T1/T4 unchanged; T2/T3/T5 now show significant effects of 0.04–0.065 SD, all under the floor; T3 points against its theory | `data/output/theory_runs/cp5_reruns/` (+ `review.json`) |

191 tests pass. `verify_outputs.py --panel` gives 6/6 on the rebuilt panel.

## 2. Things that changed under you: read before running anything

1. **Rebuild the panel.** It has a new column (`nonprofit_density`). New reference: **2,390,744 rows, 54 columns, sha256_csv `334951b1…`**. Run `python scripts/build_panel.py`, then `python scripts/check_panel_hash.py`, which must say `match: true`. Everything below needs it.
2. **CP4 LLM replay is broken, and has been since the TA import.** `run_cp4.py --replay` now fails with `CacheError: no cache entry …`. I bisected it:
   - Replay works at `f661856` and fails at `f825052`. The TA added `outcome_rule` to `specs/ratings_fairness.json`, which is the example spec pasted into every Planner prompt, so every cache key changed.
   - V3's new column also changes the Scout's column list, so reverting that one line would not fix it.
   - **Consequences:**
     - (a) The README quickstart must **not** use `run_cp4.py --replay`. Use `python scripts/rerun_specs.py`, which reruns the committed CP4 specs under the current gate with no LLM call. It's the V4 command.
     - (b) `check_llm_cache_intact` in `verify_outputs.py` still passes, because it checks hashes and not whether the prompts still hit. Its docstring ("so --replay reproduces the specs") is no longer true, so reword it.
     - (c) The new CP5 runs (O3, O4) write fresh cache entries. **Commit `data/llm_cache/`** afterwards, so `run_benchmark.py --replay` and `measure_assumption1_benchmark.py --replay` reproduce them. Don't touch `panel_columns.yaml` or `specs/ratings_fairness.json` after your live runs, or their replay breaks the same way.
3. **`prepare_population` changed** (your file, agreed in the plan). It now returns the raw outcome and plants positive controls before the rule. Before, the rule was applied twice per entry, by `prepare_population` and again by `run_spec`. Your O2 fixes are untouched. There's one new test.
4. **O5 is unaffected by D-a.** The sweep's four ratios are the program and admin shares (spread 1.4 and 1.6, never clipped under either version of the rule), `fundraising_efficiency` and `liabilities_to_assets` (clipped under both). Your running sweep is valid.
5. **Theory list changes vs the plan's section 5:**
   - **L9 (crime) is swapped** for the spare `lit_education_fundraising` (`chr_some_college` → `fundraising_expense_ratio`, −), because the crime data needs an openICPSR login (D9).
   - **L6 sign confirmed +** from Thornton (2008, JLEO).
   - **Positive controls:** `pos_income_fundraising_share` and `pos_poor_health_programs` are out; `pos_unemployment_margin` and `pos_poverty_leverage` are in.
   - Every literature entry now has `source` and `claim` fields.
6. **I couldn't push.** My clone has no credentials for github.gatech.edu, and there's no `gatech` remote. Vien pushes these commits from their own terminal. If you see nothing after `466f21a`, ask Vien.

7. **Set `OPENBLAS_NUM_THREADS=1` (and `OMP_NUM_THREADS=1`) before any long run, the benchmark especially.** On a many-core machine OpenBLAS spreads the gate's small matrix products over every core and the threads fight. My V4 rerun didn't finish in 5 h 51 min multi-threaded, and took **9 min** single-threaded with the 5 theories in parallel. One T1 run used 60 CPU-minutes against 2. `rerun_specs.py` sets this itself; `run_benchmark.py` and `run_sweep.py` don't. On Windows (PowerShell): `$env:OPENBLAS_NUM_THREADS=1; $env:OMP_NUM_THREADS=1`. Your laptop has fewer cores, so the gain may be smaller, but it won't hurt. Put it in the README's failure points.

## 3. What's left, in order

### Step 1: O3, A1 on the 40 unseen texts (needs `ANTHROPIC_API_KEY`)
```bash
git pull && python scripts/build_panel.py && python scripts/check_panel_hash.py
python -m pytest -q                                   # expect 191 passed
python scripts/measure_assumption1_benchmark.py       # live; first pass is the number
git add data/output/assumption_1_unseen.json data/llm_cache && git commit -m "A1 on unseen theories, first pass (O3)"
```
Then do the failure analysis: for each failure, what was wrong (variable, sign, type, unit) and whether a schema or prompt rule would catch it. Make no prompt changes before the first pass is committed. Read the H2 controls it chose and flag odd ones.

### Step 2: O4, the full benchmark + A2
```bash
python scripts/run_benchmark.py --screen 199 --rerun-q 0.25    # ~1-2 h; reuses the O3 cache where texts match
git add data/output/benchmark_cp5.json data/output/theory_runs/cp5 data/llm_cache
git commit -m "Run the CP5 benchmark: 20 entries, two families, screen 199 (O4)"
```
- **A2:** `seconds_per_entry` and `planning_seconds` are in the output. Sum them for "20 theories in under an hour?".
- **What to expect from the controls, from my V2/V6 runs:**
  - Both heavy-tailed positive controls are recovered at about 0.25 SD under the rule.
  - The real unplanted relations are tiny (−0.016 and +0.038 SD).
- **If a heavy-tailed positive control is *missed* in the benchmark,** first check `review_spec`: did the Planner pick the gold variable? Recovery requires a gold-matching spec (your X6 fix).
- **Paraphrase run:** only if time allows. It's first on the drop list.

### Step 3: V7, the benchmark review (5 minutes)
```bash
python scripts/review_theory_runs.py --dir data/output/theory_runs/cp5
git add data/output/theory_runs/cp5/review.json && git commit -m "Review the CP5 benchmark: verdicts follow, own families (V7)"
```
- **What it must print:** `all verdicts follow: True` and `every entry in its own family: True`.
- **Where the lines go:** paste the printed lines under "V7" in `project_one/CP5_REPORT_VIEN.md`, then write **one line of findings per literature theory** under them.
- **Read the `claim` field before writing each line.** Five sources only support the direction, or need our own inference step, so a "supported" there is weaker evidence than for L6.
- **"Unsupported" on a heavy-tailed outcome is now informative.** The V2 controls show a 0.25 SD effect would have been found.

### Step 4: O5 (your sweep). Already running; report before and after for the 14 affected cells.

### Step 5: O6, README + `verify_outputs.py`
- **Add CP5 checks to `verify_outputs.py`:**
  - `outcome_rule_check.json`: 4 outcomes still above 3; the fundraising share exempt.
  - `controls_heavy_tail.json`: `finding_holds` true for both.
  - `controls_family_heavy_tail.json`: 0 false positives, planted survives, both outcomes.
  - `cp5_reruns/review.json`: `all_follow`.
  - `benchmark_cp5.json`: the three metrics.
  - `theory_runs/cp5/review.json`: `all_follow` and `all_in_own_family`.
- **README quickstart (clean clone):**
  ```
  pip install -r requirements.txt → pytest (191) → build_panel.py (~5–10 min) → check_panel_hash.py
  → verify_outputs.py --panel
  → python scripts/rerun_specs.py --only distressed_county_rating_penalty --to quickstart --permutations 199 --no-by-year
  ```
  The last step is one replayed evidence row: T1 under the CP5 gate, no key. I tested it and got `distressed_county_rating_penalty unsupported`, `rule applied False`, in 2.0 min with BLAS on one thread (6.4 min multi-threaded; see section 2.7). Time it on yours and paste the verdict line into the README. The script refuses `--to cp4`, but `--to cp5_reruns` would overwrite the committed V4 run, so keep `quickstart`, and git-ignore it or don't commit it.
- **Tested on:** Arch Linux x86-64, Python 3.12.16, the exact pins. The panel matched the CP4 hash `3dfa5eb1…` on Linux before V3. That makes a third OS after Windows and macOS.
- **Failure points to list:** the stale LLM cache (section 2.2); the panel hash after any change to `panel.py` or `need.py`; and, for runtime, BLAS thread contention (section 2.7; set `OPENBLAS_NUM_THREADS=1`).
- **Data provenance:** the multi-year crime source is not in the repo (needs an openICPSR account; D9).
- **Last:** run the clean-clone test in an empty folder after everything above is committed, and paste the actual outputs.

### Step 6: O7, the report, then submit
- **Changes table:** take my rows from `CP5_REPORT_VIEN.md` → "Changes".
- **Plus your own rows:** two BH families, run errors count as invalid plans, recovery needs gold, the README workflow, and the drop order (the plan's section 7).
- **AI usage:** built on `ai-suggestions/cp5` @ `f825052`. My AI usage paragraph is in my sections.
- **F3 (commit split):** 9 of my commits after `f825052`.

## 4. If you're short on time
Cut in the plan's section-7 order: the paraphrase run first, then drill-down on survivors. Never cut A1's first pass, the benchmark, or the clean-clone README test.
