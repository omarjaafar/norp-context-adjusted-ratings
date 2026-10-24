"""Checkpoint 3 open check: which return types carry the Part IX rows?

    python scripts/part9_coverage.py

Reads the committed Part I and Part IX extracts (no raw data needed), keeps
one filing per (ein, tax_year) with the same rule as the panel (D1), and
writes data/output/part9_coverage.json.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT, load_catalog  # noqa: E402
from theorylab.checks import load_part, part9_coverage  # noqa: E402
from theorylab.executor import to_jsonable  # noqa: E402

OUT = REPO_ROOT / "data" / "output" / "part9_coverage.json"


def main() -> int:
    catalog = load_catalog()
    p01, _ = load_part(catalog["nccs_p01"])
    p09, _ = load_part(catalog["nccs_p09"])
    report = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
              **part9_coverage(p01, p09)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("panel_rows_with_p09", "panel_rows_with_p09_not_full_990",
                                             "not_full_990_return_types")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
