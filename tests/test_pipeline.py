"""Tests, weighted toward the failures actually hit while building this.

Every test below corresponds to a bug that silently produced a wrong number
rather than an exception -- which is the only kind worth this much test code in
a project whose deliverable is a set of numbers.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval.metrics import classify_agreement, net_benefit, wilson   # noqa: E402
from src.generate import AbstainBackend, parse_response            # noqa: E402
from src.hgvs import normalize_chromosome, normalize_protein_change  # noqa: E402
from src.parse_vcf import EmptyVcfError, _trim, read_vcf           # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- normalization

@pytest.mark.parametrize("raw,expected", [
    ("p.Val600Glu", "V600E"),
    ("V600E", "V600E"),
    ("SER214CYS", "S214C"),
    ("p.Arg175His", "R175H"),
    ("p.Thr315Ile", "T315I"),
    ("EXON 17 MUTATIONS", None),
    ("Fusion", None),
    ("", None),
    (None, None),
])
def test_protein_change_normalization(raw, expected):
    assert normalize_protein_change(raw) == expected


def test_three_letter_and_one_letter_agree():
    """The join depends on these collapsing to one key. If they diverge, every
    tier-2 match silently disappears."""
    assert normalize_protein_change("p.Val600Glu") == normalize_protein_change("V600E")


@pytest.mark.parametrize("raw,expected", [("chr7", "7"), ("7", "7"), ("CHR7", "7"),
                                          ("chrM", "MT"), ("MT", "MT"), ("chrX", "X")])
def test_chromosome_normalization(raw, expected):
    assert normalize_chromosome(raw) == expected


# ------------------------------------------------------------------- VCF layer

def test_left_trim_makes_equivalent_indels_identical():
    """CA>C and TCA>TC are the same deletion written two ways. Without trimming
    they never join to the same CIViC record and the miss is silent."""
    assert _trim(100, "CA", "C") == _trim(99, "TCA", "TC")


def test_trim_keeps_at_least_one_base():
    assert _trim(100, "A", "A") == (100, "A", "A")


def _write_vcf(tmp_path: pathlib.Path, filter_value: str) -> pathlib.Path:
    path = tmp_path / "t.vcf.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("##fileformat=VCFv4.1\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        handle.write(f"chr7\t140753336\t.\tA\tT\t.\t{filter_value}\t.\n")
    return path


def test_seqc2_style_filter_is_kept(tmp_path):
    """SEQC2 writes `PASS;HighConf`. An equality test against "PASS" drops all
    39,447 calls and reports a match rate of 0.00, which looks exactly like a
    broken retriever."""
    assert len(list(read_vcf(_write_vcf(tmp_path, "PASS;HighConf")))) == 1


def test_filtered_out_everything_raises(tmp_path):
    """An empty callset is always a bug and must stop the run, not flow on."""
    with pytest.raises(EmptyVcfError):
        list(read_vcf(_write_vcf(tmp_path, "LowConf")))


# --------------------------------------------------------------------- metrics

def test_wilson_bounds_stay_in_range_at_zero():
    """The normal approximation goes negative here; a confabulation rate of 0 is
    the case we most need a correct interval for."""
    rate = wilson(0, 50)
    assert rate.low == 0.0 and 0.0 < rate.high < 0.15


def test_wilson_empty_is_json_serializable():
    """json.dumps emits a bare NaN token, which is invalid JSON and breaks the
    CI regression check that parses results/metrics.json."""
    json.loads(json.dumps(wilson(0, 0).as_dict()))


def test_constant_abstention_scores_zero_net_benefit():
    """The whole point of the corrected metric set: a system that abstains on
    everything scores perfectly on the original spec's metrics and 0 here."""
    assert net_benefit(0, 0, 100) == 0.0
    assert net_benefit(60, 10, 100) == 50.0
    assert net_benefit(10, 60, 100) == -50.0


def test_contradiction_needs_opposing_directions():
    assert classify_agreement("RESISTANCE", "SENSITIVITYRESPONSE") == "contradict"
    assert classify_agreement("SENSITIVITYRESPONSE", "SENSITIVITYRESPONSE") == "agree"
    # Different but not opposing -> hedge, never counted as a contradiction.
    assert classify_agreement("POOR_OUTCOME", "SENSITIVITYRESPONSE") == "hedge"
    assert classify_agreement(None, "SENSITIVITYRESPONSE") == "hedge"


# ------------------------------------------------------------------ generation

def test_parse_handles_fenced_json():
    assert parse_response('```json\n{"significance":"RESISTANCE"}\n```').significance == "RESISTANCE"


def test_unparseable_counts_against_the_system():
    """A response we cannot parse is a failure, not a missing datapoint. It must
    not read as an abstention, or garbage would score as caution."""
    result = parse_response("I'm sorry, I can't help with that.")
    assert result.significance == "UNPARSEABLE"
    assert not result.abstained


def test_abstain_backend_abstains():
    assert AbstainBackend().generate("q", "ctx").abstained


# ------------------------------------------------------------- snapshot wiring

@pytest.mark.skipif(not (ROOT / "data/snapshots/latest").exists(),
                    reason="no CIViC snapshot present; run `make ingest`")
def test_assertions_are_not_in_the_retrievable_corpus():
    """The leakage control that makes Arm 3 non-circular. If assertion text ever
    reaches the index, the model reads the answer instead of inferring it."""
    from src.corpus import build_documents

    snapshot = ROOT / "data/snapshots/latest"
    with (snapshot / "assertions.jsonl").open() as handle:
        summaries = [json.loads(line).get("summary") or "" for line in handle]
    corpus = "\n".join(document.text for document in build_documents(snapshot))
    for summary in summaries:
        if len(summary) > 60:
            assert summary not in corpus
