"""Turn a CIViC snapshot into retrievable documents.

One evidence item = one document. The spec called this "the natural
granularity" and measurement agrees: evidence descriptions are short (median 480
characters, p90 944), so there is nothing to split and no chunk-overlap tuning
to do. See docs/spec-review.md, "Chunking is a non-issue".

Each document carries the surrounding structure -- gene, molecular profile,
disease, therapies, evidence level and direction -- because retrieving the
description alone strips exactly the fields a clinical interpretation must cite.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib


@dataclasses.dataclass(frozen=True)
class EvidenceDoc:
    evidence_id: int
    molecular_profile_id: int
    molecular_profile_name: str
    text: str
    metadata: dict

    @property
    def citation_key(self) -> str:
        """The token a generated interpretation must cite: EID1234."""
        return f"EID{self.evidence_id}"


def _therapy_phrase(record: dict) -> str:
    therapies = [t["name"] for t in (record.get("therapies") or [])]
    if not therapies:
        return ""
    interaction = record.get("therapyInteractionType")
    if len(therapies) > 1 and interaction:
        joiner = " + " if interaction == "COMBINATION" else " or "
        return joiner.join(therapies)
    return ", ".join(therapies)


def build_documents(snapshot_dir: str | pathlib.Path) -> list[EvidenceDoc]:
    snapshot_dir = pathlib.Path(snapshot_dir)
    with (snapshot_dir / "evidence_items.jsonl").open() as handle:
        evidence = [json.loads(line) for line in handle]

    documents: list[EvidenceDoc] = []
    for record in evidence:
        profile = record.get("molecularProfile") or {}
        disease = (record.get("disease") or {}).get("name") or "unspecified disease"
        therapy = _therapy_phrase(record)
        source = record.get("source") or {}

        # The header is what makes a document findable by a variant-shaped query;
        # the description is what makes it citable. Both go in the embedded text.
        header = (
            f"{profile.get('name', 'unknown profile')} | {disease}"
            f"{' | ' + therapy if therapy else ''} | "
            f"{record.get('evidenceType', '')} {record.get('significance', '')} "
            f"({record.get('evidenceDirection', '')}, level {record.get('evidenceLevel', '')})"
        )
        text = f"{header}\n{record.get('description') or ''}".strip()

        documents.append(EvidenceDoc(
            evidence_id=record["id"],
            molecular_profile_id=profile.get("id", -1),
            molecular_profile_name=profile.get("name", ""),
            text=text,
            metadata={
                "evidence_type": record.get("evidenceType"),
                "evidence_level": record.get("evidenceLevel"),
                "evidence_direction": record.get("evidenceDirection"),
                "significance": record.get("significance"),
                "variant_origin": record.get("variantOrigin"),
                "disease": disease,
                "therapies": [t["name"] for t in (record.get("therapies") or [])],
                "therapy_interaction": record.get("therapyInteractionType"),
                "source": {
                    "citation_id": source.get("citationId"),
                    "type": source.get("sourceType"),
                    "year": source.get("publicationYear"),
                    "title": source.get("title"),
                },
                "assertion_ids": [a["id"] for a in (record.get("assertions") or [])],
            },
        ))
    return documents


def profile_to_evidence(documents: list[EvidenceDoc]) -> dict[int, list[int]]:
    """molecular profile id -> evidence ids. This is the retrieval ground truth."""
    mapping: dict[int, list[int]] = {}
    for document in documents:
        mapping.setdefault(document.molecular_profile_id, []).append(document.evidence_id)
    return mapping
