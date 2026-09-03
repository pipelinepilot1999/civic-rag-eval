"""Assert the eval's numbers have not moved outside tolerance.

The spec proposed running the eval on every push and asserting metrics stay in
"tolerance", without saying where a tolerance comes from. For a nondeterministic
system that is how you get a test that fails randomly and is then disabled.

So tolerances are split by what actually varies:

  Deterministic arms (retrieval recall, matcher coverage, the abstain and echo
  baselines) have no run-to-run variance at a fixed snapshot. Tolerance is tight
  and a breach is a real regression.

  Model-dependent arms vary between identical runs. Their tolerance must be
  measured -- run the eval N times unchanged, take the observed spread -- not
  guessed. Until that measurement exists, this script refuses to police them
  rather than inventing a band.

Usage:
    python scripts/check_regression.py results/metrics_abstain.json baselines/abstain.json
"""

from __future__ import annotations

import json
import pathlib
import sys

# Deterministic given a fixed snapshot: any movement is a real change.
DETERMINISTIC_TOLERANCE = 0.005

DETERMINISTIC_PATHS = [
    ("arm1_retrieval", "structured", "recall@1", "point"),
    ("arm1_retrieval", "structured", "recall@5", "point"),
    ("arm1_retrieval", "lexical", "recall@1", "point"),
    ("arm1_retrieval", "lexical", "recall@5", "point"),
    ("arm1_retrieval", "hybrid", "recall@1", "point"),
    ("arm1_retrieval", "hybrid", "recall@5", "point"),
]

DETERMINISTIC_COUNTS = [
    ("matcher_coverage", "reachable_by_any_tier"),
    ("matcher_coverage", "tier1_coordinate_variants"),
    ("matcher_coverage", "tier2_protein_variants"),
    ("run", "corpus_documents"),
]


def dig(data: dict, path: tuple):
    for key in path:
        if data is None:
            return None
        data = data.get(key)
    return data


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    current = json.loads(pathlib.Path(argv[1]).read_text())
    baseline = json.loads(pathlib.Path(argv[2]).read_text())

    failures: list[str] = []
    for path in DETERMINISTIC_PATHS:
        now, then = dig(current, path), dig(baseline, path)
        if now is None or then is None:
            failures.append(f"{'.'.join(path)}: missing (now={now}, baseline={then})")
            continue
        if abs(now - then) > DETERMINISTIC_TOLERANCE:
            failures.append(
                f"{'.'.join(path)}: {then:.4f} -> {now:.4f} "
                f"(moved {abs(now - then):.4f}, tolerance {DETERMINISTIC_TOLERANCE})")

    for path in DETERMINISTIC_COUNTS:
        now, then = dig(current, path), dig(baseline, path)
        if now != then:
            failures.append(f"{'.'.join(path)}: {then} -> {now} (exact match required)")

    if failures:
        print("REGRESSION - deterministic metrics moved:\n")
        for failure in failures:
            print(f"  {failure}")
        print("\nIf this change is intentional, refresh baselines/ in the same commit "
              "and say why in the message.")
        return 1

    print(f"OK - {len(DETERMINISTIC_PATHS) + len(DETERMINISTIC_COUNTS)} "
          "deterministic metrics within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
