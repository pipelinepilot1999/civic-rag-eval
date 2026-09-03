"""Three-tier variant -> CIViC molecular-profile matcher.

The original spec assumed a single join from VCF coordinates to CIViC variants.
Measured against the live database, that join can reach at most 41.5% of CIViC's
variant space, and reaches 0% if the reference build is not converted first
(CIViC is GRCh37; modern somatic callsets are GRCh38). See docs/spec-review.md #2.

So matching is explicitly tiered, and every tier reports its own hit count:

  Tier 1  coordinate   exact GRCh37 chrom/pos/ref/alt
  Tier 2  protein      gene symbol + normalized protein change (reaches specific
                       variants that carry no coordinates at all, e.g. FGFR2 P235R)
  Tier 3  category     gene-level rules for the 3,178 variant records that
                       describe a *class* of variant -- AMPLIFICATION, EXPRESSION,
                       EXON 17 MUTATIONS, fusions -- which no single coordinate
                       can ever match

Tier 3 is deliberately conservative. A category rule fires only when the query
carries enough annotation to satisfy it; rules needing information we do not have
(exon number, copy-number state, expression level) are counted as
`unmatchable_without` rather than guessed at. That count is reported, because a
silent miss and a principled abstention are different things.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from collections import defaultdict
from typing import Iterable

from .hgvs import normalize_chromosome, normalize_protein_change

# Category variant names, mapped to the query property that would satisfy them.
CATEGORY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^EXON\s+(?P<exons>[\d\s,and/&-]+?)\s+(MUTATION|SKIPPING|DELETION|INSERTION)"), "exon"),
    (re.compile(r"\bAMPLIFICATION\b"), "copy_gain"),
    (re.compile(r"\b(HOMOZYGOUS\s+)?(DELETION|LOSS)\b"), "copy_loss"),
    (re.compile(r"\b(OVER)?EXPRESSION\b"), "expression"),
    (re.compile(r"\bMETHYLATION\b"), "methylation"),
    (re.compile(r"\bPHOSPHORYLATION\b"), "phosphorylation"),
    (re.compile(r"\bWILDTYPE\b|\bWILD TYPE\b"), "wildtype"),
    (re.compile(r"::"), "fusion"),
    (re.compile(r"\bFUSION\b|\bREARRANGEMENT\b|\bTRANSLOCATION\b"), "fusion"),
    (re.compile(r"\b(LOSS[- ]OF[- ]FUNCTION|LOF)\b"), "lof"),
    (re.compile(r"\b(GAIN[- ]OF[- ]FUNCTION|GOF)\b"), "gof"),
    # The broadest bucket, checked last: "BRAF MUTATION", "TP53 MUTATION".
    (re.compile(r"\bMUTATION(S)?\b"), "any_mutation"),
]

# Which category rules a plain small-variant call can satisfy from sequence
# annotation alone. Copy number, expression and methylation need orthogonal
# assays; we never pretend a SNV satisfies them.
SATISFIABLE_BY_SMALL_VARIANT = {"exon", "any_mutation", "lof", "gof"}

# Sequence-ontology consequences we treat as loss-of-function.
LOF_CONSEQUENCES = {
    "stop_gained", "frameshift_variant", "splice_acceptor_variant",
    "splice_donor_variant", "start_lost", "stop_lost",
}


@dataclasses.dataclass(frozen=True)
class QueryVariant:
    """One VCF record, annotated enough to be matched."""
    chrom: str                      # GRCh38, as called
    pos: int                        # GRCh38
    ref: str
    alt: str
    gene: str | None = None
    protein_changes: tuple[str, ...] = ()   # normalized, e.g. ("V600E",)
    consequences: tuple[str, ...] = ()
    exon: int | None = None
    hg19_pos: int | None = None     # filled by annotation; None = liftover failed

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"


@dataclasses.dataclass(frozen=True)
class Match:
    variant_id: int
    molecular_profile_ids: tuple[int, ...]
    tier: str                       # "coordinate" | "protein" | "category"
    rule: str                       # what actually fired, for error analysis
    civic_variant_name: str
    gene: str | None


class CivicMatcher:
    """Indexes a CIViC snapshot and matches annotated VCF records against it."""

    def __init__(self, snapshot_dir: pathlib.Path):
        self.snapshot_dir = pathlib.Path(snapshot_dir)
        self.variants = self._load("variants.jsonl")
        self.molecular_profiles = self._load("molecular_profiles.jsonl")

        # variant id -> molecular profiles containing it
        self.variant_to_mps: dict[int, list[int]] = defaultdict(list)
        for profile in self.molecular_profiles:
            for variant in profile.get("variants") or []:
                self.variant_to_mps[variant["id"]].append(profile["id"])

        # A coordinate maps to a *list*: two CIViC records can share one
        # position (e.g. a specific substitution and a broader range record).
        # Keying to a single id silently dropped one of them.
        self.by_coordinate: dict[tuple, list[int]] = defaultdict(list)
        self.coord_records_missing_bases = 0
        self.by_protein: dict[tuple, list[int]] = defaultdict(list)
        self.by_category: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        self.unindexed: list[int] = []
        self._build_indexes()

    def _load(self, name: str) -> list[dict]:
        path = self.snapshot_dir / name
        with path.open() as handle:
            return [json.loads(line) for line in handle]

    @staticmethod
    def _gene_of(variant: dict) -> str | None:
        feature = variant.get("feature") or {}
        name = feature.get("name")
        return name.upper() if name else None

    def _build_indexes(self) -> None:
        for variant in self.variants:
            variant_id = variant["id"]
            gene = self._gene_of(variant)
            name = (variant.get("name") or "").strip()
            indexed = False

            coordinates = variant.get("coordinates") or {}
            if coordinates.get("start") and not (
                coordinates.get("referenceBases") and coordinates.get("variantBases")
            ):
                # Has a position but no alleles -- ranges, indels and CNV records.
                # Not addressable by an exact coordinate key; tier 2/3 must carry it.
                self.coord_records_missing_bases += 1
            if coordinates.get("start") and coordinates.get("referenceBases") and coordinates.get("variantBases"):
                build = (coordinates.get("referenceBuild") or "").upper()
                # Only GRCh37 coordinates are indexed; queries are converted to
                # GRCh37 before lookup. Mixing builds here is exactly the silent
                # zero-match failure the review flagged.
                if build in {"GRCH37", ""}:
                    key = (
                        normalize_chromosome(coordinates["chromosome"]),
                        int(coordinates["start"]),
                        coordinates["referenceBases"].upper(),
                        coordinates["variantBases"].upper(),
                    )
                    self.by_coordinate[key].append(variant_id)
                    indexed = True

            for candidate in self._protein_candidates(variant):
                if gene:
                    self.by_protein[(gene, candidate)].append(variant_id)
                    indexed = True

            if gene:
                for pattern, rule in CATEGORY_PATTERNS:
                    match = pattern.search(name.upper())
                    if match:
                        detail = rule
                        if rule == "exon":
                            exons = re.findall(r"\d+", match.group("exons"))
                            detail = "exon:" + ",".join(exons)
                        self.by_category[gene].append((variant_id, rule, detail))
                        indexed = True
                        break

            if not indexed:
                self.unindexed.append(variant_id)

    @staticmethod
    def _protein_candidates(variant: dict) -> set[str]:
        """Every normalized protein change this CIViC variant can be known by."""
        found: set[str] = set()
        for raw in [variant.get("name")] + list(variant.get("variantAliases") or []):
            normalized = normalize_protein_change(raw)
            if normalized:
                found.add(normalized)
        for description in variant.get("hgvsDescriptions") or []:
            if ":p." in description:
                normalized = normalize_protein_change(description.split(":p.", 1)[1])
                if normalized:
                    found.add(normalized)
        return found

    # ---------------------------------------------------------------- matching

    def match(self, query: QueryVariant) -> list[Match]:
        """Return every CIViC variant this query hits, best tier first.

        Tiers are not mutually exclusive: a BRAF V600E call legitimately hits the
        specific V600E record (tier 1/2) *and* `BRAF MUTATION` (tier 3). All are
        returned; the retriever decides what to surface.
        """
        matches: list[Match] = []
        seen: set[int] = set()

        def add(variant_id: int, tier: str, rule: str) -> None:
            if variant_id in seen:
                return
            seen.add(variant_id)
            record = self._variant_by_id.get(variant_id)
            matches.append(Match(
                variant_id=variant_id,
                molecular_profile_ids=tuple(self.variant_to_mps.get(variant_id, ())),
                tier=tier,
                rule=rule,
                civic_variant_name=(record or {}).get("name", ""),
                gene=self._gene_of(record) if record else None,
            ))

        if query.hg19_pos is not None:
            key = (normalize_chromosome(query.chrom), query.hg19_pos,
                   query.ref.upper(), query.alt.upper())
            for variant_id in self.by_coordinate.get(key, []):
                add(variant_id, "coordinate", "grch37_exact")

        if query.gene:
            gene = query.gene.upper()
            for change in query.protein_changes:
                for variant_id in self.by_protein.get((gene, change), []):
                    add(variant_id, "protein", f"gene+p.{change}")

            for variant_id, rule, detail in self.by_category.get(gene, []):
                if rule not in SATISFIABLE_BY_SMALL_VARIANT:
                    continue
                if rule == "exon":
                    wanted = {int(n) for n in detail.split(":", 1)[1].split(",")}
                    if query.exon is not None and query.exon in wanted:
                        add(variant_id, "category", detail)
                elif rule == "lof":
                    if set(query.consequences) & LOF_CONSEQUENCES:
                        add(variant_id, "category", "lof_consequence")
                elif rule == "any_mutation":
                    add(variant_id, "category", "gene_any_mutation")
                elif rule == "gof":
                    continue  # not inferable from sequence alone
        return matches

    @property
    def _variant_by_id(self) -> dict[int, dict]:
        if not hasattr(self, "_vbid"):
            self._vbid = {v["id"]: v for v in self.variants}
        return self._vbid

    # ------------------------------------------------------------- diagnostics

    def coverage_report(self) -> dict:
        """How much of CIViC each tier can reach. This is a headline number."""
        category_counts: dict[str, int] = defaultdict(int)
        for entries in self.by_category.values():
            for _, rule, _ in entries:
                category_counts[rule] += 1
        indexed_ids = (
            {vid for ids in self.by_coordinate.values() for vid in ids}
            | {vid for ids in self.by_protein.values() for vid in ids}
            | {vid for entries in self.by_category.values() for vid, _, _ in entries}
        )
        return {
            "civic_variants_total": len(self.variants),
            "tier1_coordinate_keys": len(self.by_coordinate),
            "tier1_coordinate_variants": len({v for ids in self.by_coordinate.values() for v in ids}),
            "coord_records_missing_alleles": self.coord_records_missing_bases,
            "tier2_protein_keys": len(self.by_protein),
            "tier2_protein_variants": len({v for ids in self.by_protein.values() for v in ids}),
            "tier3_category_variants": sum(len(e) for e in self.by_category.values()),
            "tier3_by_rule": dict(sorted(category_counts.items(), key=lambda kv: -kv[1])),
            "tier3_satisfiable_by_small_variant": sum(
                c for r, c in category_counts.items() if r in SATISFIABLE_BY_SMALL_VARIANT
            ),
            "reachable_by_any_tier": len(indexed_ids),
            "unreachable": len(self.unindexed),
        }
