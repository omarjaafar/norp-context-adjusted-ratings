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
