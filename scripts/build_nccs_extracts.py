"""Checkpoint 2: slim NCCS efile extracts (Parts I / IX / X, 2018-2022).

For each table-year in the catalog: download the full CSV once into
data/raw/nccs/ (git-ignored), check its header against the catalog contract,
keep only the contracted columns, and write a gzip extract under
data/extracts/nccs/. MANIFEST.json records rows, bytes, sha256 and whether
each file is under GitHub's 100 MB limit (Assumption 3).

Usage:
    python scripts/build_nccs_extracts.py                       # all sources, all years
    python scripts/build_nccs_extracts.py --sources nccs_p09 --years 2021 2022
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theorylab.catalog import REPO_ROOT, check_contract, load_catalog, missing_columns  # noqa: E402

NCCS_SOURCES = ("nccs_p01", "nccs_p09", "nccs_p10")
RAW_DIR = REPO_ROOT / "data" / "raw" / "nccs"
MANIFEST = REPO_ROOT / "data" / "extracts" / "nccs" / "MANIFEST.json"
GITHUB_LIMIT_BYTES = 100 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "theorylab-cp2"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)
    return "downloaded"


def extract_one(source, year: int) -> dict:
    url = source.origin_for(year)
    raw_path = RAW_DIR / url.rsplit("/", 1)[-1]
    status = download(url, raw_path)

    header = list(pd.read_csv(raw_path, nrows=0, dtype=str).columns)
    missing = missing_columns(header, source)
    if missing:
        nearby = [h for h in header if h.startswith(("ORG_", "TAX_", "RETURN_", "F9_"))]
        return {"source": source.name, "year": year, "url": url, "status": status,
                "error": f"missing contracted columns {missing}",
                "header_sample": nearby[:120]}

    df = pd.read_csv(raw_path, usecols=source.columns(), dtype=str, low_memory=False)
    df = df[source.columns()]
    contract = check_contract(df, source, label=f"{source.name}-{year}")

    out = source.extract_path(year)
    out.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the gzip bytes (and so the hash) identical across reruns.
    df.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    size = out.stat().st_size
    return {
        "source": source.name,
        "year": year,
        "url": url,
        "status": status,
        "raw_bytes": raw_path.stat().st_size,
        "path": str(out.relative_to(REPO_ROOT)),
        "rows": contract["rows"],
        "bytes": size,
        "size_mb": round(size / 1e6, 2),
        "sha256": sha256(out),
        "under_github_limit": size < GITHUB_LIMIT_BYTES,
        "contract": contract,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(NCCS_SOURCES), choices=NCCS_SOURCES)
    ap.add_argument("--years", nargs="+", type=int)
    args = ap.parse_args()

    catalog = load_catalog()
    previous = {}
    if MANIFEST.exists():
        for e in json.loads(MANIFEST.read_text(encoding="utf-8")).get("files", []):
            previous[(e["source"], e["year"])] = e

    for name in args.sources:
        source = catalog[name]
        for year in args.years or source.years:
            if year not in source.years:
                print(f"skip {name} {year}: not in catalog years {source.years}")
                continue
            entry = extract_one(source, year)
            previous[(name, year)] = entry
            print(name, year, entry.get("rows"), entry.get("size_mb"), entry.get("error", "ok"))

    files = [previous[k] for k in sorted(previous)]
    ok = [e for e in files if "error" not in e]
    violations = [v for e in ok for v in e["contract"]["violations"]]
    errors = [f"{e['source']}-{e['year']}: {e['error']}" for e in files if "error" in e]
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "github_limit_bytes": GITHUB_LIMIT_BYTES,
        "files": files,
        "assumption_3": {
            "statement": "column-subset extracts fit under 100 MB per file",
            "files_measured": len(ok),
            "max_size_mb": max((e["size_mb"] for e in ok), default=None),
            "all_under_limit": bool(ok) and all(e["under_github_limit"] for e in ok),
        },
        "contract_violations": violations,
        "errors": errors,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["assumption_3"], indent=2))
    if violations or errors:
        print("\n".join(errors + violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
