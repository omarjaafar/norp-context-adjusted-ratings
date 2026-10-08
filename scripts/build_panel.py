"""Checkpoint 2: organization-year panel with join logs.

Inputs (built by the other CP2 scripts):
    data/extracts/nccs/*.csv.gz          build_nccs_extracts.py
    data/derived/ein_county_crosswalk.csv.gz   build_crosswalk.py
    data/derived/county_need.csv         build_county_need.py

Outputs:
    data/derived/org_year_panel.csv.gz   (git-ignored; rebuilt from the inputs above)
    data/output/panel_join_log.json      (committed; per-year match rates, Assumption 4)

Usage:
    python scripts/build_panel.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT, load_catalog  # noqa: E402
from theorylab.panel import assumption_4_check, build_panel, prepare_part  # noqa: E402

CROSSWALK = REPO_ROOT / "data" / "derived" / "ein_county_crosswalk.csv.gz"
PANEL = REPO_ROOT / "data" / "derived" / "org_year_panel.csv.gz"
LOG = REPO_ROOT / "data" / "output" / "panel_join_log.json"


def load_source(source):
    frames, files = [], {}
    for year in source.years:
        path = source.extract_path(year)
        if not path.exists():
            files[str(year)] = "missing"
            continue
        df = pd.read_csv(path, dtype=str)
        files[str(year)] = int(len(df))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no extracts for {source.name}; run build_nccs_extracts.py first")
    prepared, log = prepare_part(pd.concat(frames, ignore_index=True), source)
    log["extract_rows_by_year"] = files
    return prepared, log


def main() -> int:
    catalog = load_catalog()
    parts, part_logs = {}, {}
    for name in ("nccs_p01", "nccs_p09", "nccs_p10"):
        parts[name], part_logs[name] = load_source(catalog[name])

    crosswalk = pd.read_csv(CROSSWALK, dtype=str)
    need = pd.read_csv(catalog["acs_county"].extract_path(), dtype={"county_fips": str})

    panel, log = build_panel(parts["nccs_p01"], parts["nccs_p09"], parts["nccs_p10"],
                             crosswalk, need)
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL, index=False, compression={"method": "gzip", "mtime": 0})

    log = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "parts": part_logs,
        **log,
        "panel": {"path": str(PANEL.relative_to(REPO_ROOT)), "rows": int(len(panel)),
                  "columns": list(panel.columns),
                  "size_mb": round(PANEL.stat().st_size / 1e6, 2)},
        "assumption_4": assumption_4_check(log),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log["assumption_4"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
