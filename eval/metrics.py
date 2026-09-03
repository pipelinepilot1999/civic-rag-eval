"""Scoring functions, with intervals, and a metric set that abstention can't game.

The original spec's four headline metrics are all maximized by a system that
abstains on every input (docs/spec-review.md #1). The fix is not a better prompt;
it is a metric set where silence has a cost:

  answer_rate        on variants that DO have supporting evidence. Constant
                     abstention scores 0 here.
  abstention_rate    on negative controls. Constant answering scores 0 here.
  net_benefit        (correct - incorrect) per 100 variants. Abstention scores
                     exactly 0: neither rewarded nor punished. Confident wrongness
                     goes negative.

Every rate is a binomial proportion on a few hundred trials, so every rate is
reported with a Wilson score interval. A bare point estimate of 0.18 on n=147
is not a result; 0.18 [0.13, 0.25] is.
"""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class Rate:
    """A proportion with its 95% Wilson score interval."""
    numerator: int
    denominator: int
    point: float | None
    low: float | None
    high: float | None

    def __str__(self) -> str:
        if self.denominator == 0 or self.point is None:
            return "n/a (n=0)"
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.denominator})"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def wilson(numerator: int, denominator: int, z: float = 1.959963985) -> Rate:
    """95% Wilson score interval.

    Chosen over the normal approximation because these rates live near 0 and 1
    (a good confabulation rate is close to 0), where the normal interval produces
    bounds below zero and badly wrong coverage.
    """
    if denominator == 0:
        # None, not NaN: json.dumps writes a bare NaN token, which is not valid
        # JSON and breaks any strict parser reading results/metrics.json -- which
        # is precisely what the CI regression check does.
        return Rate(numerator, 0, None, None, None)
    proportion = numerator / denominator
    denom = 1 + z * z / denominator
    centre = proportion + z * z / (2 * denominator)
    spread = z * math.sqrt(
        proportion * (1 - proportion) / denominator + z * z / (4 * denominator * denominator)
    )
    return Rate(numerator, denominator, proportion,
                max(0.0, (centre - spread) / denom),
                min(1.0, (centre + spread) / denom))


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> bool:
    """Did any known-relevant evidence item make the top k?"""
    return bool(set(retrieved_ids[:k]) & relevant_ids)


def net_benefit(correct: int, incorrect: int, total: int) -> float | None:
    """Correct minus incorrect answers per 100 variants.

    Abstentions are neither counted as correct nor incorrect, so a system that
    abstains on everything scores exactly 0.0 -- the same as one that answers
    half right and half wrong, and worse than one that answers correctly at all.
    """
    if total == 0:
        return None
    return 100.0 * (correct - incorrect) / total


def citation_validity(cited: list[str], retrieved: list[str]) -> tuple[int, int]:
    """(valid citations, total citations). Programmatic and complete.

    NOTE: this checks that a cited ID was actually in the retrieved context. It
    does NOT check that the claim attached to it is supported by it -- that is a
    separate, non-free metric. The spec conflated the two; see
    docs/spec-review.md #3.
    """
    retrieved_set = set(retrieved)
    return sum(1 for citation in cited if citation in retrieved_set), len(cited)


AGREE, CONTRADICT, HEDGE = "agree", "contradict", "hedge"

# CIViC significance values collapsed into the direction a clinician acts on.
# Anything not listed is treated as unmappable and excluded from the arm rather
# than forced into a bucket.
SIGNIFICANCE_DIRECTION = {
    "SENSITIVITYRESPONSE": "sensitive",
    "REDUCED_SENSITIVITY": "resistant",
    "RESISTANCE": "resistant",
    "ADVERSE_RESPONSE": "adverse",
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "BETTER_OUTCOME": "better",
    "POOR_OUTCOME": "poor",
    "PATHOGENIC": "pathogenic",
    "LIKELY_PATHOGENIC": "pathogenic",
    "BENIGN": "benign",
    "LIKELY_BENIGN": "benign",
    "ONCOGENICITY": "oncogenic",
    "PREDISPOSING": "predisposing",
}

# Which directions actively contradict each other, as opposed to merely differing.
CONTRADICTORY_PAIRS = {
    frozenset({"sensitive", "resistant"}),
    frozenset({"positive", "negative"}),
    frozenset({"better", "poor"}),
    frozenset({"pathogenic", "benign"}),
}


# Models emit clinically-correct significance values in non-canonical spellings:
# PREDICTIVE_RESISTANCE for RESISTANCE, PREDISPOSITION for PREDISPOSING,
# POOR_PROGNOSIS for POOR_OUTCOME. Scoring those as disagreement measures
# vocabulary compliance, not clinical correctness, and understates every model.
# The evidence-type prefix is stripped and known synonyms are folded in.
#
# This is a metric bug, not a model behaviour to be proud of -- the real fix is a
# strict enum via structured outputs. But the metric must be robust either way,
# or a prompt tweak would move the headline numbers without changing behaviour.
SIGNIFICANCE_SYNONYMS = {
    "PREDISPOSITION": "PREDISPOSING",
    "POOR_PROGNOSIS": "POOR_OUTCOME",
    "FAVORABLE_PROGNOSIS": "BETTER_OUTCOME",
    "GOOD_OUTCOME": "BETTER_OUTCOME",
    "SENSITIVITY": "SENSITIVITYRESPONSE",
    "SENSITIVE": "SENSITIVITYRESPONSE",
    "SENSITIVITY_RESPONSE": "SENSITIVITYRESPONSE",
    "RESPONSE": "SENSITIVITYRESPONSE",
    "LIKELY_ONCOGENIC": "ONCOGENICITY",
    "ONCOGENIC": "ONCOGENICITY",
    "LOSS_OF_FUNCTION": "PATHOGENIC",
    "DOMINANT_NEGATIVE": "PATHOGENIC",
    "UNCERTAIN_SIGNIFICANCE": None,   # a genuine hedge, not a mapping failure
    "CONFLICTING_EVIDENCE": None,
}

# Evidence-type prefixes the models attach to the significance value.
_PREFIXES = ("PREDICTIVE_", "PROGNOSTIC_", "DIAGNOSTIC_", "PREDISPOSING_",
             "FUNCTIONAL_", "ONCOGENIC_", "PREDICTED_")


def canonical_significance(value: str | None) -> str | None:
    """Fold a model's significance string onto CIViC's controlled vocabulary."""
    if value is None:
        return None
    text = str(value).upper().strip().replace(" ", "_").replace("-", "_")
    for prefix in _PREFIXES:
        if text.startswith(prefix) and len(text) > len(prefix):
            text = text[len(prefix):]
            break
    if text in SIGNIFICANCE_DIRECTION:
        return text
    if text in SIGNIFICANCE_SYNONYMS:
        return SIGNIFICANCE_SYNONYMS[text]
    return None


def classify_agreement(predicted: str | None, truth: str | None) -> str:
    """Three-way: does the generated call agree with, contradict, or hedge the label."""
    if predicted is None:
        return HEDGE
    predicted_direction = SIGNIFICANCE_DIRECTION.get(canonical_significance(predicted) or "")
    truth_direction = SIGNIFICANCE_DIRECTION.get(canonical_significance(truth) or "")
    if predicted_direction is None or truth_direction is None:
        return HEDGE
    if predicted_direction == truth_direction:
        return AGREE
    if frozenset({predicted_direction, truth_direction}) in CONTRADICTORY_PAIRS:
        return CONTRADICT
    return HEDGE
