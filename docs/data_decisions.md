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

_To be written by Vien._

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
