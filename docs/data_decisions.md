# Data decisions

Choices we made in the data layer that affect results. Each entry gives the decision, the reason, and how to check or change it. Numbers come from committed outputs.

## D1. One filing per organization-year: latest timestamp wins (T4, Omar)

**Problem.** NCCS can hold more than one Part I/IX/X row for the same `(ein, tax_year)`: amended returns, re-filings, or short-period returns. The TA starter code kept "the last row in file order," which depends on how NCCS happened to sort the file.

**Decision.** `theorylab/panel.py::prepare_part` keeps the row with the latest `RETURN_TIME_STAMP`. An amended or corrected return is filed after the one it replaces. Rows with no parseable timestamp rank lowest, and exact ties fall back to file order.

**Evidence.**
- Each part's `duplicate_resolution` block in `data/output/panel_join_log.json` records:
  - how many duplicate groups there were
  - how many kept rows are marked amended (`RETURN_AMENDED_X`)
  - how many rows had no timestamp
  - how many groups share one timestamp, and of those, how many are **exact copies** (every column equal, compared row by row since CP3) and how many differ
- What it shows (Part I; Parts IX and X are alike):
  - 67,295 duplicate groups.
  - 37,138 are exact copies, so the choice does not matter for them.
  - 40 share a timestamp but differ in some column. These are the only groups where file order still decides.
  - 30,117 have distinct timestamps, which is where the timestamp rule matters.
- **Correction (CP2 feedback).** The CP2 report called the same-timestamp groups "exact copies" before any rows were compared. The row comparison confirms the claim for all but 40 groups in Part I, 24 in Part IX and 39 in Part X.
- Tests: `tests/test_panel.py::test_prepare_part_latest_timestamp_beats_file_order`, `test_prepare_part_separates_exact_copies_from_same_timestamp_rows_that_differ`.
- **Rebuild check.** `panel.sha256_csv` and `panel.rows` in the join log identify the panel. A rebuild from the committed inputs should reproduce the CSV hash. The gzip hash (`sha256_gz`) can differ across zlib versions.

## D2. Geography is each organization's current county, used for every year (T7, Omar)

**Problem.** The EIN → county crosswalk takes `STATE` and `COUNTY` from the NORP NGO table. That is a single snapshot of where each organization is registered now. The panel gives every tax year from 2018 to 2022 that same county, so an organization that moved between counties is assigned its later county for its earlier years.

**Decision.** For CP2, we accept the snapshot and document it as a known limitation.

Why it is acceptable for now:
- Most organizations don't change county within five years.
- The theories we test compare organizations against county conditions. A small share of misplaced organization-years adds noise; it doesn't create bias toward any one result.

**How to measure it later.** NCCS publishes a per-filing header table (`F9-P00-T00-HEADER-{year}`) with the filer's address for each return. Comparing its state and ZIP across years for the same EIN would give the share of organizations that moved. If that share turns out to be large, the crosswalk should switch to per-year addresses.

This is listed as a CP3 candidate, not a CP2 deliverable.

## D3. County reference is the Census 2020 county file, not NORP `county_fips_lookup` (TA starter code; kept)

**Reason.**
- `county_fips_lookup` has no Florida rows (found in Checkpoint 0).
- Connecticut has both legacy counties (09001–09015) and 2022+ planning regions (09110–09190).

The Census file resolves both. Per-state resolution shares are in `data/output/crosswalk_report.json` (`florida_resolved_share`, `connecticut_resolved_share`).

## D4. ACS 5-year vintage ↔ tax year alignment (T8, Vien)

**Problem.** ACS county estimates are 5-year averages. The "2018" vintage covers surveys from 2014 to 2018, and the "2022" vintage covers 2018 to 2022. The TA starter code joined vintage Y to tax year Y without saying so or saying why.

**Decision.** Tax year Y is joined to the ACS 5-year vintage that ends in Y. So tax year 2018 uses the 2014–2018 estimates, and tax year 2022 uses 2018–2022.

Why this alignment and not a centered one (vintage Y+2, whose middle year is Y):
- **No look-ahead.** The need measure for a tax year uses only surveys taken in or before that year. A centered window would describe a county in 2018 partly with 2020 data, which includes the pandemic.
- **It keeps Connecticut manageable.** Every vintage from 2022 on reports Connecticut by planning region. A centered window would put three of our five tax years on those vintages. See D5.
- **The choice barely moves the ranking of counties.** See the evidence below.

**What this means for analysis.**
- **Need is a slow, mostly cross-sectional measure.** Vintages one year apart share four of their five survey years. A county's need index correlates 0.98 to 0.99 with its own value one vintage later, and 0.94 between 2018 and 2022, the only two vintages in our range that share a single survey year.
- **Do not read year-to-year change in need as a signal.** Most of it is one survey year entering and one leaving the window. Models that use the change in need over time (the lagged models in our drop list) need non-overlapping vintages, and we have only one such pair.
- **Compare incomes within a year only.** Each vintage reports median household income in that vintage's dollars. The need index is safe because it standardizes within each year. The raw `median_household_income` column is not comparable across years without an inflation adjustment.
- **Unemployment moves the most.** It has the lowest correlation between 2018 and 2022 (0.77), so results that lean on unemployment alone are the most sensitive to the alignment.

**Evidence.**
- `vintage_correlation` in `data/output/county_need_report.json` gives the correlation of poverty rate, unemployment rate and need index between each pair of adjacent vintages and between 2018 and 2022, with the number of counties compared.
- Test: `tests/test_need.py::test_vintage_correlation_uses_counties_present_in_both_years`.

## D5. Connecticut planning regions rebuilt for 2018–2021 from town data (T6, Vien)

**Problem.** Our audit of the TA code said Connecticut drops out of 2022. It is the other way round, and larger:
- The NORP NGO table names Connecticut organizations by planning region, so the crosswalk gives all of them a planning-region code (09110–09190). `data/output/crosswalk_report.json` counts 41,728 EINs resolved this way (`method_counts.ct_planning_region`).
- The ACS publishes planning regions only from the 2022 vintage. For 2018–2021 it publishes the eight legacy counties (09001–09015).
- So Connecticut organizations matched the need table in 2022 and missed it in 2018, 2019, 2020 and 2021.

A county → planning-region lookup cannot fix this, because the nine regions cut across the eight counties.

**Decision.** `theorylab/need.py::ct_region_rows` rebuilds the nine planning regions for 2018–2021 from town-level ACS data. A planning region is a set of whole towns, and the ACS publishes all 169 Connecticut towns for every year.
- **Town → region mapping** comes from the 2022 ACS itself, which nests each town under its region. Nothing is typed in by hand.
- **Counts** (population, poverty universe, people below poverty, labor force, unemployed) are summed across a region's towns.
- **Median household income** cannot be summed. It is interpolated from the region's summed household-income brackets (ACS table B19001).
- **Rebuilt rows are marked** `acs_estimated = True`. Rows published by the ACS have `False`.
- **The need index for everyone else is unchanged.** Means, spreads and the top-quartile cut are computed from published rows only, and rebuilt rows are scored against them. Otherwise Connecticut would count twice in 2018–2021.

**Evidence.** In the `ct_planning_regions` block of `data/output/county_need_report.json`:
- All 169 towns map to a region in every year (`towns_unmapped_by_year` is empty).
- The same method applied to 2022 reproduces the nine published regions: every count matches exactly, and median income is off by 0.85% on average and 1.54% at most.
- 9 rows are added to each of 2018–2021 (`estimated_rows_by_year`).
- Tests: `test_towns_sum_to_planning_regions`, `test_grouped_median_interpolates_inside_the_bracket` and `test_estimated_rows_do_not_move_published_rows` in `tests/test_need.py`.

**Still open.**
- **Double counting.** For 2018–2021 the need table holds Connecticut twice, as legacy counties and as planning regions. Anything that sums or averages over counties should filter on `acs_estimated`.
- **No NORP need for Connecticut.** See D6.

## D6. NORP tract need is one undated snapshot, aggregated to county by population (T5, Vien)

**Problem.** The TA starter need table was ACS only. NORP's `disadvantaged_communities.csv` has housing burden and a food desert flag, but per census tract and with no year column, while the need table is per county per year.

**Decision.** `theorylab/need.py::aggregate_tract_need` takes the population-weighted county mean of each measure, and `add_norp_need` attaches the same county value to every ACS year.

- `avg_housing_burden` is the weighted mean of the tract `Avg Housing Burden` (a percent).
- `food_desert_share` is the weighted mean of the tract `Food Desert Pct`. That column is a 0/1 flag per tract, so the county value is the share of the county's population living in flagged tracts (0–1).
- A tract with zero population or a missing value does not count toward that measure.
- The file drops the leading zero of `County Fips` (`9001` for 09001); the loader restores it.
- The two columns are on different scales (a percent and a 0–1 share), so standardize before combining them.
- Neither column goes into `need_index`, so the index and `high_need` are unchanged from the ACS-only table.

**What this means for results.** These two columns vary across counties but not across years. They can explain differences between counties; they cannot explain change over time within a county.

**Evidence.** The `norp_need` block in `data/output/county_need_report.json`:
- 72,742 tracts (contract met), 421 with zero population, `Food Desert Pct` filled for 99.59%.
- 3,142 counties, none with a null value for either measure.
- Need rows with no NORP value are 2.73% of the table (`need_rows_unmatched_by_state_by_year`):
  - Puerto Rico: 78 rows per year. The file has no Puerto Rico tracts.
  - Alaska: two census areas from 2020 on (02063 and 02066), which replaced a county the file still lists under its old code.
  - Connecticut: the nine planning regions in every year. The file uses legacy Connecticut counties, and tract numbers are not unique across Connecticut, so tracts cannot be reassigned to regions by number. Connecticut organizations therefore have ACS need but no housing burden or food desert share (see D5).
- Tests: `tests/test_need.py`, the five `test_tract_need_*`, `test_load_norp_tracts_*` and `test_add_norp_need_*` tests.

**Open question.** 68.28% of tracts with a value carry the food desert flag (`food_desert_flagged_tract_share`), which is high for a food desert definition. We use the column as NORP provides it and have not traced how NORP defines it.

## D7. Part IX ratios come from full Form 990 filers only (CP2 open check, Omar)

**Problem.** In 2018 the panel had 351,711 rows with Part IX but only 263,339 full Form 990 filings, so Part IX could not be only full-990 rows. Part IX of a full 990 has the program / management / fundraising split that three of the four rating ratios need. Form 990EZ has no such split.

**Finding.** `scripts/part9_coverage.py` keeps one filing per (ein, tax_year) with the D1 rule and counts Part IX rows by the return type of the Part I filing:

- The extra rows are all **990EZ**: 538,937 panel rows over 2018–2022 (88,372 / 93,470 / 102,028 / 126,877 / 128,190 by year). No other return type appears.
- Their functional split is usable in at most 0.02% of rows in any year. NCCS fills Part IX for EZ filers with totals only.
- Every full-990 filing has a Part IX row (`full_990_without_p09` is 0 in every year), and its split is usable in 87.3–87.7% of them.
- Part I and Part IX disagree on the return type in 3–37 rows per year.

**Decision.** Every analysis that uses Part IX ratios, including the standard rating, keeps `return_type == "990"`. The theory specs do this in `population.return_type` (`specs/ratings_fairness.json`).

**Evidence.** `data/output/part9_coverage.json`: `by_tax_year`, `not_full_990_return_types`.

## D8. County Health Rankings release ↔ tax year, one lag per measure (CP4 V2, Vien)

**Problem.** CP3 feedback asked for health, education and crime indicators next to the economic need index. County Health Rankings (CHR) publishes one national file per release year (`analytic_data<YEAR>.csv`), but a release does not describe its own year: each measure carries data from earlier years, with a different lag per source. The column labels also change case between releases ("Some college" vs "Some College").

**Decision.** Five columns, read by CHR measure code (the second header row, stable across releases), joined on `(county_fips, tax_year)`. Tax year T uses release T + lag:

| Column | CHR code | Source | Release for tax year T | Data years for T |
|---|---|---|---|---|
| `chr_poor_fair_health` | v002 | BRFSS | T + 3 | T |
| `chr_uninsured` | v085 | Census SAHIE | T + 3 | T |
| `chr_some_college` | v069 | ACS 5-year | T + 2 | T−4 to T |
| `chr_hs_completion` | v168 | ACS 5-year | T + 2 | T−4 to T |
| `chr_violent_crime_rate` | v043 | FBI UCR | 2022 for every T | 2014 and 2016 |

So tax years 2018–2022 use releases 2020–2025 (`theorylab/need.py::CHR_MEASURES`, `chr_release_for`).

Why:
- **Same rule as D4.** The ACS-based measures use the 5-year window ending in the tax year, the way `need_index` does, and the BRFSS/SAHIE measures use the tax year itself. No column uses data from after the tax year.
- **The lags are from CHR's own documentation.** The 2025 "Select Measures and Data Sources" table lists Poor or Fair Health BRFSS 2022, Uninsured SAHIE 2022, and Some College and High School Completion ACS 2019–2023. That gives a lag of 3 and 2. We read the table for 2025 only and assume the lag is the same in 2020–2024. The year-to-year correlations below are consistent with this (no sudden drop for SAHIE or the ACS measures).
- **Crime is frozen in CHR.** We measured that every county's v043 value is identical in the 2019, 2020, 2021 and 2022 releases (2,951 counties, 100% identical), and v043 is absent from the 2023, 2024 and 2025 files. We did not confirm the reason from CHR's documentation; the FBI's switch from UCR to NIBRS in 2021 is the likely one. The 2014/2016 data years come from CHR's labeling of the measure in those releases. We use release 2022 for every tax year. Like D6, the column varies across counties but not across years.

**Gaps (reported, not imputed).** Coverage per year and column is in the `chr` block of `data/output/county_need_report.json` (`match_rate_by_year`, `unmatched_by_state_by_year`).
- **Puerto Rico:** CHR has no Puerto Rico rows (78 need rows per year).
- **Connecticut:** releases 2020–2024 report the 8 legacy counties; the 2025 release adds the 9 planning regions. CT organizations sit in planning regions (D5), so they get CHR values only for `chr_poor_fair_health` and `chr_uninsured` in tax year 2022 (release 2025): 9 rows. The regions cut across counties, so legacy values cannot be reassigned without town-level data. The legacy-county rows in the need table do carry CHR values (they are kept for 2018–2021, D5).
- **`chr_hs_completion` is missing for tax year 2018.** v168 starts in the 2021 release; the 2020 release has only v021 (the cohort graduation rate, a different measure). Using release 2021 for 2018 would break the no-look-ahead rule.
- **Violent crime** is missing for about 8.6% of county rows (`match_rate_by_year` 0.913–0.914), spread over 28 states, the most in Mississippi (50 counties), Indiana (29) and Nebraska (20). This is UCR non-reporting.
- About 97.3% of need rows match for the other four columns.

**Series break: poor or fair health.** The median county value drops from 0.201 (tax 2019, release 2022) to 0.153 (tax 2020, release 2023), then rises to 0.189 by 2022. The county ranking holds (correlation 0.97 between those years), so this is a level shift between releases, most likely a change in CHR's estimation method; we have not confirmed the cause. **Analyses must compare this column within a tax year:** use tax-year effects, or standardize within year. A pooled exposure without year effects would read the shift as signal.

**Evidence.**
- Extract: `data/extracts/chr/chr_county.csv` (releases 2020–2025, county rows, the five codes; sha256 in `county_need_report.json` → `chr.extract_sha256`, raw file hashes in `chr.raw_files_sha256`). Contract: catalog `chr_county`, exact rows per release (3,142; 3,143 in 2024 with Alaska's 02063/02066; 3,152 in 2025 with both CT geographies).
- Values in range: shares in 0–1, crime ≥ 0 (`chr.range_violations`, all 0).
- Rebuild: `python scripts/build_county_need.py --chr-only --chr-dir data/raw/chr` refreshes only the CHR columns and checks that every ACS and NORP column is unchanged.
- Tests: `tests/test_need.py`, the `test_chr_*` and `test_read_chr_*` and `test_add_chr_need_*` tests (vintage rule, code-based reading, missing measures, CT handling, ranges, rerun safety).
