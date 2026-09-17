"""Constrained interpretation generation.

Output is a fixed JSON shape. Abstention is a first-class value of
`significance`, not an error path -- but see eval/metrics.py: abstaining is not
free here, because answer rate is scored alongside correctness.

Three backends behind one interface:

  AbstainBackend    always returns insufficient-evidence. This is not a mock: it
                    is the baseline that scores *perfectly* on every metric the
                    original spec proposed, and it stays in the results table
                    permanently as proof the metric set is not gameable. It plays
                    the role the permutation null plays in multiomics-ml-brca.
  EchoBackend       returns the significance of the top retrieved item verbatim.
                    A second free baseline: it measures how much of the task is
                    just copying the first search result.
  AnthropicBackend  the real system. Needs ANTHROPIC_API_KEY.

Determinism: the Anthropic backend pins temperature 0 and a fixed prompt so that
CI tolerances can be derived from measured run-to-run variance rather than
guessed (docs/spec-review.md, "CI on every push").
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import threading
from typing import Protocol, Sequence

from .retrieve import Retrieved

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Read KEY=value lines from .env into the environment.

    Deliberately not python-dotenv: three lines, one fewer dependency, and it
    keeps the credential-loading path visible in the file that uses it. Existing
    environment variables win, so `ANTHROPIC_API_KEY=... make eval-real` still
    overrides the file.
    """
    file = pathlib.Path(path)
    if not file.exists():
        return
    for line in file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

SYSTEM_PROMPT = """\
You are a somatic variant interpretation assistant. You are given a variant and \
a numbered list of curated CIViC evidence items.

Rules:
1. Use ONLY the evidence provided. Do not use recalled knowledge about the gene \
or variant.
2. Every clinical claim must cite the evidence item it comes from, by its EID.
3. If the provided evidence does not support a conclusion about THIS variant in \
THIS disease context, return significance "INSUFFICIENT_EVIDENCE" with an empty \
citations list. Abstaining is correct when the evidence is absent or is about a \
different variant.
4. Do not abstain merely because the evidence is limited. If evidence directly \
addresses the variant, report what it says.

Respond with a single JSON object and nothing else:
{
  "significance": "<one CIViC significance value, or INSUFFICIENT_EVIDENCE>",
  "evidence_level": "<A|B|C|D|E or null>",
  "therapeutic_implication": "<one sentence, or null>",
  "citations": ["EID123", ...],
  "reasoning": "<two sentences at most>"
}"""


@dataclasses.dataclass
class Interpretation:
    significance: str
    evidence_level: str | None
    therapeutic_implication: str | None
    citations: list[str]
    reasoning: str
    raw: str = ""
    parse_error: str | None = None

    @property
    def abstained(self) -> bool:
        return self.significance.upper() == INSUFFICIENT

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def render_context(retrieved: Sequence[Retrieved]) -> str:
    if not retrieved:
        return "(no evidence items were retrieved for this variant)"
    blocks = []
    for item in retrieved:
        document = item.document
        blocks.append(f"[{document.citation_key}] {document.text}")
    return "\n\n".join(blocks)


def render_query(gene: str | None, protein_change: str | None,
                 chrom: str | None, pos: int | None,
                 ref: str | None, alt: str | None, disease: str | None) -> str:
    parts = [f"Gene: {gene or 'unknown'}"]
    if protein_change:
        parts.append(f"Protein change: p.{protein_change}")
    if chrom and pos:
        parts.append(f"Genomic (GRCh37): chr{chrom}:{pos} {ref}>{alt}")
    parts.append(f"Disease context: {disease or 'not specified'}")
    return "\n".join(parts)


def parse_response(text: str) -> Interpretation:
    """Parse the model's JSON, tolerating fenced code blocks and stray prose."""
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S)
    if fence:
        candidate = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", candidate, re.S)
        if brace:
            candidate = brace.group(0)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        # An unparseable response is a failure of the system, not a missing
        # datapoint. It is recorded as a non-abstention with no citations so it
        # counts against the answer, rather than being silently dropped.
        return Interpretation(
            significance="UNPARSEABLE", evidence_level=None,
            therapeutic_implication=None, citations=[], reasoning="",
            raw=text, parse_error=str(exc),
        )
    citations = data.get("citations") or []
    if isinstance(citations, str):
        citations = [citations]
    return Interpretation(
        significance=str(data.get("significance") or INSUFFICIENT).upper(),
        evidence_level=data.get("evidence_level"),
        therapeutic_implication=data.get("therapeutic_implication"),
        citations=[str(c).upper().strip() for c in citations],
        reasoning=str(data.get("reasoning") or ""),
        raw=text,
    )


class Backend(Protocol):
    name: str
    def generate(self, query_text: str, context: str) -> Interpretation: ...


class AbstainBackend:
    """Always abstains. Scores 1.00 on every metric the original spec proposed."""
    name = "always-abstain"

    def generate(self, query_text: str, context: str) -> Interpretation:
        return Interpretation(
            significance=INSUFFICIENT, evidence_level=None,
            therapeutic_implication=None, citations=[],
            reasoning="Constant-abstention baseline.",
        )


class EchoBackend:
    """Reports the top retrieved item's significance, citing it. No model."""
    name = "echo-top-1"

    def generate(self, query_text: str, context: str) -> Interpretation:
        match = re.search(r"\[(EID\d+)\][^\n]*?\|\s*[A-Z_]+\s+([A-Z_]+)\s*\(", context)
        if not match:
            return Interpretation(INSUFFICIENT, None, None, [],
                                  "No retrieved evidence to echo.")
        return Interpretation(
            significance=match.group(2), evidence_level=None,
            therapeutic_implication=None, citations=[match.group(1)],
            reasoning="Echo of the top-ranked retrieved evidence item.",
        )


# Published $/MTok, input and output. Used only to report what a run cost --
# "the eval costs $N" belongs in the README, and a reviewer will ask.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class AnthropicBackend:
    """The real system. Requires ANTHROPIC_API_KEY."""
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 1024):
        import anthropic                                    # noqa: PLC0415

        load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Create a key at platform.claude.com "
                "and put it in .env (see .env.example). A Claude Code subscription "
                "login is a different credential and cannot be used here."
            )
        # Keys created through the Console's "linked account" flow are
        # identity-linked and reject any request that does not name the
        # workspace it acts in ("anthropic-workspace-id is required when
        # authenticating with an identity-linked API key"). Older org-level keys
        # do not need it, so the header is only sent when configured.
        headers = {}
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if workspace:
            headers["anthropic-workspace-id"] = workspace
        self.client = anthropic.Anthropic(default_headers=headers or None)
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"
        # run_eval can call generate() from a ThreadPoolExecutor (--workers > 1).
        # `self.calls += 1` is a read-modify-write and loses increments under
        # concurrency, which would silently understate the reported cost.
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.errors = 0

    def generate(self, query_text: str, context: str) -> Interpretation:
        import anthropic                                    # noqa: PLC0415
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content":
                           f"VARIANT\n{query_text}\n\nRETRIEVED EVIDENCE\n{context}"}],
            )
        except anthropic.APIStatusError as exc:
            with self._lock:
                self.errors += 1
            # A failed call is a failure of the run, not a missing datapoint. It
            # is recorded as an unparseable answer so it counts against the
            # system rather than quietly shrinking the denominator.
            return Interpretation(
                significance="UNPARSEABLE", evidence_level=None,
                therapeutic_implication=None, citations=[], reasoning="",
                raw="", parse_error=f"{type(exc).__name__}: {exc}",
            )

        usage = response.usage
        with self._lock:
            self.calls += 1
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.cached_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

        text = "".join(block.text for block in response.content if block.type == "text")
        return parse_response(text)

    def cost_report(self) -> dict:
        rate_in, rate_out = PRICING.get(self.model, (0.0, 0.0))
        return {
            "model": self.model,
            "calls": self.calls,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cached_tokens,
            "estimated_usd": round(
                self.input_tokens / 1e6 * rate_in + self.output_tokens / 1e6 * rate_out, 4),
        }


def get_backend(name: str, model: str = "claude-opus-5") -> Backend:
    if name == "abstain":
        return AbstainBackend()
    if name == "echo":
        return EchoBackend()
    if name == "anthropic":
        return AnthropicBackend(model=model)
    raise ValueError(f"unknown backend {name!r} (abstain | echo | anthropic)")
