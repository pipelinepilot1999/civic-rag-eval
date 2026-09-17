"""Every number quoted in the README must match results/.

A README with stale numbers is worse than one with no numbers, and results
tables drift silently the moment anything upstream changes. This test makes the
README a checked artifact rather than prose.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def _load(name: str) -> dict | None:
    path = ROOT / "results" / name
    return json.loads(path.read_text()) if path.exists() else None


def _rounded(metrics: dict, *path: str, digits: int = 3) -> float:
    node = metrics
    for key in path:
        node = node[key]
    return round(node, digits)


@pytest.mark.skipif(not (ROOT / "results/metrics_dense.json").exists(),
                    reason="dense results absent; run `make eval-dense`")
@pytest.mark.parametrize("retriever,recall1,recall5", [
    ("structured", 0.960, 0.999),
    ("lexical", 0.896, 0.940),
    ("dense", 0.366, 0.566),
    ("hybrid", 0.960, 0.999),
])
def test_readme_retrieval_table(retriever, recall1, recall5):
    arm = _load("metrics_dense.json")["arm1_retrieval"][retriever]
    assert _rounded(arm, "recall@1", "point") == recall1
    assert _rounded(arm, "recall@5", "point") == recall5
    assert f"{recall1:.3f}" in README.read_text()


@pytest.mark.skipif(not (ROOT / "results/metrics_echo.json").exists(),
                    reason="echo results absent; run `make eval`")
@pytest.mark.parametrize("stratum,rate", [
    ("easy", 0.047), ("hard", 0.462),
    ("hard_constructed", 0.807), ("pooled_observed_only", 0.080),
])
def test_readme_negative_control_table(stratum, rate):
    arm = _load("metrics_echo.json")["arm4_negative_controls"][stratum]
    assert _rounded(arm, "confabulation_rate", "point") == rate


@pytest.mark.skipif(not (ROOT / "results/metrics_abstain.json").exists(),
                    reason="abstain results absent; run `make eval`")
def test_constant_abstention_is_still_worthless():
    """The claim the whole metric redesign rests on. If this ever stops holding,
    the metric set has regressed back to something abstention can game."""
    metrics = _load("metrics_abstain.json")
    assert metrics["arm3b_leave_one_out_agreement"]["net_benefit_per_100"] == 0.0
    assert metrics["arm3b_leave_one_out_agreement"]["answer_rate"]["point"] == 0.0
    for stratum in ("easy", "hard", "hard_constructed"):
        arm = metrics["arm4_negative_controls"][stratum]
        assert arm["confabulation_rate"]["point"] == 0.0, (
            f"{stratum}: constant abstention must score a perfect 0.000 here -- "
            "that is exactly why this metric alone cannot be the headline")


@pytest.mark.skipif(not (ROOT / "results/metrics_dense.json").exists(),
                    reason="dense results absent")
def test_readme_matcher_coverage_percentages():
    coverage = _load("metrics_dense.json")["matcher_coverage"]
    total = coverage["civic_variants_total"]
    coordinate_only = 100 * coverage["tier1_coordinate_variants"] / total
    three_tier = 100 * coverage["reachable_by_any_tier"] / total
    assert round(coordinate_only, 1) == 24.5
    assert round(three_tier, 1) == 82.9
    text = README.read_text()
    assert "24.5%" in text and "82.9%" in text


def test_readme_has_no_unfilled_placeholders():
    text = README.read_text()
    assert "0.XX" not in text, "README still contains placeholder numbers"
    assert "TODO" not in text


@pytest.mark.skipif(not (ROOT / "results/metrics.json").exists(),
                    reason="model results absent; run `make eval-real`")
@pytest.mark.parametrize("filename,label", [
    ("metrics.json", "Opus 5"),
    ("metrics_haiku.json", "Haiku 4.5"),
    ("metrics_echo.json", "echo top-1 (no model)"),
    ("metrics_abstain.json", "always-abstain (no model)"),
])
def test_readme_headline_row_matches_results(filename, label):
    """Every model row in the headline table must match its results file.

    This test exists because it did not. The Haiku row said 0.340 agreement and
    27.56 net benefit for a week after the run was rescored to 0.359 / 30.77 --
    the other README tests covered retrieval, negative controls and matcher
    coverage, but nothing checked the table the README opens with. Regenerate
    with `python scripts/results_table.py --write`.
    """
    path = ROOT / "results" / filename
    if not path.exists():
        pytest.skip(f"{filename} absent")
    metrics = json.loads(path.read_text())
    arm = metrics["arm3b_leave_one_out_agreement"]
    row = next((line for line in README.read_text().splitlines()
                if line.startswith(f"| {label} |")), None)
    assert row is not None, f"no headline row for {label!r} in README"
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1] == f"{arm['answer_rate']['point']:.3f}"
    assert cells[2] == f"{arm['agreement_rate']['point']:.3f}"
    assert cells[3] == f"{arm['contradiction_rate']['point']:.3f}"
    assert cells[6] == str(arm["net_benefit_per_100"])


def test_readme_tables_are_generated_not_typed():
    """Every table in the README must sit inside generation markers."""
    text = README.read_text()
    import re as _re
    tables = _re.findall(r"^\|.*\|\n\|[-| ]+\|", text, _re.M)
    starts = text.count("results:")
    assert starts >= 2 * len(tables), (
        f"{len(tables)} tables but only {starts // 2} marked; "
        "a hand-typed table will drift")
