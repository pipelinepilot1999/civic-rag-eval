"""Pull a dated, reproducible CIViC snapshot to JSONL.

Writes four files under data/snapshots/<date>/ :

  evidence_items.jsonl     the retrievable corpus (ACCEPTED only)
  molecular_profiles.jsonl what evidence is actually keyed to
  variants.jsonl           coordinates / HGVS / category names, for the matcher
  assertions.jsonl         held-out labels for the contradiction arm

Why four files and not one: the eval treats evidence items as *retrievable* and
assertions as *held out*. Keeping them in separate files makes it structurally
hard to leak an assertion into the index by accident, which is the failure mode
that made the original spec's Arm 3 circular. See docs/spec-review.md #4.

Usage:
    python -m src.ingest                 # today's date
    python -m src.ingest --date 2026-09-02
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys

from . import civic_client as cc

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAPSHOT_ROOT = REPO_ROOT / "data" / "snapshots"

DATASETS = [
    ("evidence_items", cc.EVIDENCE_QUERY, "evidenceItems"),
    ("molecular_profiles", cc.MOLECULAR_PROFILE_QUERY, "molecularProfiles"),
    ("variants", cc.VARIANT_QUERY, "variants"),
    ("assertions", cc.ASSERTION_QUERY, "assertions"),
]


def write_jsonl(path: pathlib.Path, records: list[dict]) -> str:
    """Write records sorted by id, and return the sha256 of the file.

    Sorting matters: it makes the snapshot byte-stable across runs, so the
    checksum only changes when CIViC content changes, not when pagination
    order shifts.
    """
    records = sorted(records, key=lambda r: r["id"])
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat(),
                        help="snapshot directory name (default: today)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing snapshot for this date")
    args = parser.parse_args(argv)

    out_dir = SNAPSHOT_ROOT / args.date
    if out_dir.exists() and not args.force:
        print(f"snapshot {out_dir} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "snapshot_date": args.date,
        "pulled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "api_url": cc.API_URL,
        "source": "CIViC (civicdb.org), Griffith Lab, Washington University. Content is CC0.",
        "files": {},
    }

    for name, query, root in DATASETS:
        print(f"pulling {name} ...", flush=True)
        records = list(cc.paginate(query, root))
        path = out_dir / f"{name}.jsonl"
        digest = write_jsonl(path, records)
        manifest["files"][name] = {"records": len(records), "sha256": digest}
        print(f"  {len(records):>6,} records -> {path.name}  ({digest[:12]})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    latest = SNAPSHOT_ROOT / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(args.date)

    print(f"\nsnapshot written to {out_dir}")
    print(f"'latest' -> {args.date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
