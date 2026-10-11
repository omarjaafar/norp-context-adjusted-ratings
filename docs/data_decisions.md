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
- Test: `tests/test_panel.py::test_prepare_part_latest_timestamp_beats_file_order`.

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

## D6. NORP tract-level need: one snapshot, weighted by population (T5, Vien)

**Problem.** The TA need table was ACS only. Our plan also lists housing burden and food desert share from NORP's `disadvantaged_communities.csv`, which has one row per census tract.

**Decision.** `theorylab/need.py::aggregate_tract_need` takes the population-weighted mean of `Avg Housing Burden` and `Food Desert Pct` across each county's tracts. The results are the `avg_housing_burden` and `food_desert_share` columns of the need table.
- **Same value every year.** The file has no year column, so each county's value is repeated for 2018–2022. These two columns carry no information about change over time.
- **Not part of `need_index`.** The index stays ACS only, so it is defined the same way in every year and does not mix a dated measure with an undated one.
- **Different scales.** `avg_housing_burden` is on a 0–51 scale and `food_desert_share` is 0–1. Standardize before combining them.
- **Tracts that cannot contribute are skipped.** Tracts with zero population carry no weight, and a tract with a blank value is left out of that measure only.
- **County codes are repaired.** The file drops the leading zero of `County Fips` (`9001` for 09001); the loader restores it.

**Evidence.** In the `norp_need` block of `data/output/county_need_report.json`:
- 72,742 tracts, no duplicate or missing tract IDs, 421 with zero population.
- All 3,142 counties in the file get both values (`counties_null` is 0 for both).
- 2.73% of need-table rows have no NORP value (`null_rates`). `need_rows_unmatched_by_state_by_year` breaks them down by state code.

**Known gaps.**
- **Puerto Rico (state code 72):** 78 rows a year. It is not in the NORP file.
- **Connecticut (09):** 9 planning-region rows a year. The NORP file uses legacy counties, and tract numbers are not unique across Connecticut, so tracts cannot be reassigned to regions by number. Connecticut organizations therefore have ACS need but no housing burden or food desert share.
- **Alaska (02):** 2 rows a year from 2020, when the Valdez-Cordova census area was split in two. The NORP file still has the old area.

## D5. NORP tract need is one undated snapshot, aggregated to county by population (T5, Vien)

**Problem.** The TA starter need table was ACS only. NORP's `disadvantaged_communities.csv` has housing burden and a food desert flag, but per census tract and with no year column, while the need table is per county per year.

**Decision.** `theorylab/need.py::aggregate_tract_need` takes the population-weighted county mean of each measure, and `add_norp_need` attaches the same county value to every ACS year.

- `avg_housing_burden` is the weighted mean of the tract `Avg Housing Burden` (a percent).
- `food_desert_share` is the weighted mean of the tract `Food Desert Pct`. That column is a 0/1 flag per tract, so the county value is the share of the county's population living in flagged tracts (0–1).
- A tract with zero population or a missing value does not count toward that measure.
- Neither column goes into `need_index`, so the index and `high_need` are unchanged from the ACS-only table.

**What this means for results.** These two columns vary across counties but not across years. They can explain differences between counties; they cannot explain change over time within a county.

**Evidence.** The `norp_need` block in `data/output/county_need_report.json`:
- 72,742 tracts (contract met), 421 with zero population, `Food Desert Pct` filled for 99.59%.
- 3,142 counties, none with a null value for either measure.
- Need rows with no NORP value are 2.51% of the table: Puerto Rico (78 rows per year; the file has no Puerto Rico tracts), two Alaska census areas from 2020 on (02063 and 02066, which replaced a county the file still lists under its old code), and the nine Connecticut planning regions in 2022 (the file uses legacy Connecticut counties; see T6).
- Tests: `tests/test_need.py`, the five `test_tract_need_*`, `test_load_norp_tracts_*` and `test_add_norp_need_*` tests.

**Open question.** 68.28% of tracts with a value carry the food desert flag (`food_desert_flagged_tract_share`), which is high for a food desert definition. We use the column as NORP provides it and have not traced how NORP defines it.
