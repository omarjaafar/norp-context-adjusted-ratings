"""CP4 V1 (CP3 feedback F6): check that a panel rebuilt on this machine matches the reference.

Hashes the *uncompressed* CSV stream of data/derived/org_year_panel.csv.gz (gzip
bytes depend on the zlib build, so they are not compared) and counts its rows,
then compares both with `panel.sha256_csv` / `panel.rows` of the reference join
log. The reference is the join log as committed at --ref (default HEAD), read
with `git show`, because build_panel.py overwrites the working copy.

It also hashes every column separately. When two machines disagree on the
whole-file hash, comparing their per-column hashes (--compare other.json) names
the columns that differ without moving a 400 MB file between machines.
--diff-panel compares two local panels cell by cell and shows the first
differences.

Writes data/output/panel_rebuild_check.json (or --out).

Usage:
    python scripts/check_panel_hash.py [--panel PATH] [--ref COMMIT]
        [--compare OTHER_CHECK.json] [--diff-panel OTHER_PANEL.csv.gz] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT  # noqa: E402

PANEL = REPO_ROOT / "data" / "derived" / "org_year_panel.csv.gz"
LOG = "data/output/panel_join_log.json"
OUT = REPO_ROOT / "data" / "output" / "panel_rebuild_check.json"
CHUNK = 200_000


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def csv_hash_and_rows(path: Path) -> tuple[str, int, int]:
    """sha256 of the decompressed CSV, data rows (lines minus header), uncompressed bytes."""
    h, lines, size = hashlib.sha256(), 0, 0
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            lines += chunk.count(b"\n")
            size += len(chunk)
    return h.hexdigest(), lines - 1, size


def column_hashes(path: Path) -> dict:
    """sha256 per column of the CSV text exactly as written (no parsing of values)."""
    hashers, sizes = {}, {}
    for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False,
                             chunksize=CHUNK):
        for col in chunk.columns:
            text = ("\n".join(chunk[col].tolist()) + "\n").encode("utf-8")
            hashers.setdefault(col, hashlib.sha256()).update(text)
            sizes[col] = sizes.get(col, 0) + len(text)
    return {c: {"sha256": h.hexdigest(), "bytes": sizes[c]} for c, h in hashers.items()}


def compare_columns(mine: dict, other: dict) -> dict:
    cols = list(dict.fromkeys([*mine, *other]))
    differ = [c for c in cols if c in mine and c in other and mine[c]["sha256"] != other[c]["sha256"]]
    return {
        "columns_differing": differ,
        "bytes_difference": {c: mine[c]["bytes"] - other[c]["bytes"] for c in differ},
        "only_here": [c for c in cols if c not in other],
        "only_there": [c for c in cols if c not in mine],
    }


def diff_panels(a: Path, b: Path, examples: int = 5) -> dict:
    """Cell-by-cell comparison of two panels as text; first differing rows per column."""
    out, row0 = {}, 0
    ra = pd.read_csv(a, dtype=str, keep_default_na=False, chunksize=CHUNK)
    rb = pd.read_csv(b, dtype=str, keep_default_na=False, chunksize=CHUNK)
    for ca, cb in zip(ra, rb):
        cb.index = ca.index
        for col in ca.columns.intersection(cb.columns):
            neq = ca[col].to_numpy() != cb[col].to_numpy()
            if neq.any():
                d = out.setdefault(col, {"cells": 0, "examples": []})
                d["cells"] += int(neq.sum())
                for i in np.flatnonzero(neq)[: max(0, examples - len(d["examples"]))]:
                    d["examples"].append({"row": int(row0 + i), "here": ca[col].iat[i],
                                          "there": cb[col].iat[i]})
        row0 += len(ca)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--ref", default="HEAD", help="commit whose committed join log is the reference")
    ap.add_argument("--compare", help="another machine's panel_rebuild_check.json")
    ap.add_argument("--diff-panel", help="another local panel to compare cell by cell")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    panel = Path(args.panel)
    ref_log = json.loads(git("show", f"{args.ref}:{LOG}"))["panel"]
    sha, rows, size = csv_hash_and_rows(panel)
    cols = column_hashes(panel)
    status = git("status", "--porcelain", "--untracked-files=no")
    result = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "machine": {"os": platform.platform(), "machine": platform.machine(),
                    "python": platform.python_version(), "pandas": pd.__version__,
                    "numpy": np.__version__},
        "commit": git("rev-parse", "HEAD"),
        "working_tree_clean": status == "",
        "panel": {"path": panel.as_posix(), "rows": rows, "columns": len(cols),
                  "csv_bytes": size, "sha256_csv": sha},
        "reference": {"source": f"{LOG} @ {git('rev-parse', '--short', args.ref)}",
                      "rows": ref_log["rows"], "sha256_csv": ref_log["sha256_csv"],
                      "columns": len(ref_log["columns"])},
        "match": {"sha256_csv": sha == ref_log["sha256_csv"], "rows": rows == ref_log["rows"],
                  "columns": list(cols) == ref_log["columns"]},
        "column_hashes": cols,
    }
    if args.compare:
        other = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        result["compared_with"] = {"file": args.compare, "machine": other.get("machine"),
                                   "commit": other.get("commit"),
                                   "sha256_csv_equal": other["panel"]["sha256_csv"] == sha,
                                   **compare_columns(cols, other["column_hashes"])}
    if args.diff_panel:
        result["cell_diff"] = {"other_panel": args.diff_panel,
                               "by_column": diff_panels(panel, Path(args.diff_panel))}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("machine", "commit", "panel", "reference", "match")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
