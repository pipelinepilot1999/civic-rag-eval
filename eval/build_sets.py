"""Construct the evaluation sets, with the leakage controls the spec lacked.

Three sets, written to data/eval_sets/ :

1. retrieval_holdout.jsonl
   Queryable molecular profiles that have at least one accepted evidence item.
   Ground truth = that profile's evidence ids. Used for Arm 1 (recall@k).

2. assertion_holdout.jsonl
   The 147 accepted CIViC Assertions. The assertion's `significance` is the
   held-out label; its text never enters the index. The model must reach the
   curators' conclusion from the evidence items alone.

   This replaces the spec's Arm 3, which was circular: it scored the model
   against a label printed on the very evidence item being retrieved, and so
   measured transcription rather than interpretation. See docs/spec-review.md #4.

3. negative_controls.jsonl
   Real somatic calls from SEQC2 HCC1395 that match NO CIViC record at any tier,
   split into two strata:
     hard  - the gene IS covered by CIViC at other positions. A plausible-looking
             variant in a familiar gene: the case where confabulation is likely.
     easy  - the gene has no CIViC presence at all.
   Reporting these separately matters, because a system that abstains on
   everything unfamiliar but confabulates on familiar genes has a real weakness
   that a pooled number hides.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.corpus import build_documents, profile_to_evidence          # noqa: E402
from src.hgvs import normalize_chromosome, normalize_protein_change  # noqa: E402
from src.match import (SATISFIABLE_BY_SMALL_VARIANT, CivicMatcher,   # noqa: E402
                       QueryVariant)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data/snapshots/latest"
OUT_DIR = ROOT / "data/eval_sets"
SEED = 20260902


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def query_from_civic_variant(variant: dict) -> dict | None:
    """Build the query an annotated VCF would have produced for this variant.

    Returns None for records a small-variant callset could never produce a query
    for -- fusions, expression, amplification. Excluding them is not hiding a
    failure; it is scoping the eval to the input modality the system accepts.
    """
    feature = variant.get("feature") or {}
    gene = (feature.get("name") or "").upper()
    if not gene:
        return None

    protein = normalize_protein_change(variant.get("name"))
    if not protein:
        for description in variant.get("hgvsDescriptions") or []:
            if ":p." in description:
                protein = normalize_protein_change(description.split(":p.", 1)[1])
                if protein:
                    break

    coordinates = variant.get("coordinates") or {}
    has_coordinates = bool(
        coordinates.get("start") and coordinates.get("referenceBases")
        and coordinates.get("variantBases")
    )
    if not protein and not has_coordinates:
        return None

    return {
        "civic_variant_id": variant["id"],
        "gene": gene,
        "protein_change": protein,
        "chrom": normalize_chromosome(coordinates["chromosome"]) if has_coordinates else None,
        # CIViC coordinates are already GRCh37, so they go straight into hg19_pos.
        "hg19_pos": int(coordinates["start"]) if has_coordinates else None,
        "ref": coordinates["referenceBases"].upper() if has_coordinates else None,
        "alt": coordinates["variantBases"].upper() if has_coordinates else None,
    }


def to_query_variant(record: dict) -> QueryVariant:
    return QueryVariant(
        chrom=record.get("chrom") or "0",
        pos=record.get("hg19_pos") or 0,
        ref=record.get("ref") or "N",
        alt=record.get("alt") or "N",
        gene=record.get("gene"),
        protein_changes=(record["protein_change"],) if record.get("protein_change") else (),
        hg19_pos=record.get("hg19_pos"),
    )


def build_retrieval_holdout() -> list[dict]:
    documents = build_documents(SNAPSHOT)
    profile_evidence = profile_to_evidence(documents)
    variants = {v["id"]: v for v in load_jsonl(SNAPSHOT / "variants.jsonl")}
    profiles = load_jsonl(SNAPSHOT / "molecular_profiles.jsonl")

    rows = []
    for profile in profiles:
        evidence_ids = profile_evidence.get(profile["id"])
        if not evidence_ids:
            continue
        members = profile.get("variants") or []
        # Multi-variant profiles are boolean combinations that a single call
        # cannot satisfy (595 of 5,665). Recorded, but not scored as if one
        # variant should have retrieved them. See docs/spec-review.md, smaller #1.
        if len(members) != 1:
            continue
        variant = variants.get(members[0]["id"])
        if not variant:
            continue
        query = query_from_civic_variant(variant)
        if not query:
            continue
        rows.append({
            **query,
            "molecular_profile_id": profile["id"],
            "molecular_profile_name": profile.get("name"),
            "relevant_evidence_ids": sorted(evidence_ids),
        })
    return rows


def build_assertion_holdout() -> list[dict]:
    variants = {v["id"]: v for v in load_jsonl(SNAPSHOT / "variants.jsonl")}
    profiles = {p["id"]: p for p in load_jsonl(SNAPSHOT / "molecular_profiles.jsonl")}
    assertions = load_jsonl(SNAPSHOT / "assertions.jsonl")

    rows, skipped = [], {"multi_variant": 0, "unqueryable": 0, "no_evidence": 0}
    for assertion in assertions:
        profile = profiles.get((assertion.get("molecularProfile") or {}).get("id", -1))
        if not profile:
            continue
        members = profile.get("variants") or []
        if len(members) != 1:
            skipped["multi_variant"] += 1
            continue
        variant = variants.get(members[0]["id"])
        query = query_from_civic_variant(variant) if variant else None
        if not query:
            skipped["unqueryable"] += 1
            continue
        evidence_ids = [e["id"] for e in (assertion.get("evidenceItems") or [])]
        if not evidence_ids:
            skipped["no_evidence"] += 1
            continue
        rows.append({
            **query,
            "assertion_id": assertion["id"],
            "molecular_profile_id": profile["id"],
            "molecular_profile_name": profile.get("name"),
            # --- held-out labels: never indexed, never shown to the model ---
            "label_significance": assertion.get("significance"),
            "label_assertion_type": assertion.get("assertionType"),
            "label_direction": assertion.get("assertionDirection"),
            "label_amp_level": assertion.get("ampLevel"),
            "label_disease": (assertion.get("disease") or {}).get("name"),
            "label_therapies": [t["name"] for t in (assertion.get("therapies") or [])],
            "supporting_evidence_ids": sorted(evidence_ids),
        })
    return rows, skipped


def build_leave_one_out(min_evidence: int = 3) -> list[dict]:
    """A powered companion to the assertion arm.

    The assertion set is the gold standard -- curated conclusions, never indexed --
    but only 29 of 147 assertions are reachable from a small-variant VCF. 94 of the
    117 that are not concern *fusions* (BCR::ABL1 and similar), which an SNV/indel
    callset cannot express at all. n=29 gives a 95% interval roughly +/-0.17 wide,
    which cannot distinguish a good system from a bad one.

    So a second design runs alongside it: for each molecular profile with at least
    `min_evidence` accepted evidence items, one item is held out of the index. The
    model sees the profile's *other* evidence and must reach the held-out item's
    clinical significance.

    This is weaker than the assertion arm -- the remaining evidence for a profile
    usually points the same way, so it is an easier task -- and it is reported
    separately for that reason, never pooled with the assertion numbers.
    """
    documents = build_documents(SNAPSHOT)
    profile_evidence = profile_to_evidence(documents)
    by_id = {document.evidence_id: document for document in documents}
    variants = {v["id"]: v for v in load_jsonl(SNAPSHOT / "variants.jsonl")}
    profiles = {p["id"]: p for p in load_jsonl(SNAPSHOT / "molecular_profiles.jsonl")}

    rng = random.Random(SEED)
    rows = []
    for profile_id, evidence_ids in sorted(profile_evidence.items()):
        if len(evidence_ids) < min_evidence:
            continue
        profile = profiles.get(profile_id)
        if not profile:
            continue
        members = profile.get("variants") or []
        if len(members) != 1:
            continue
        variant = variants.get(members[0]["id"])
        query = query_from_civic_variant(variant) if variant else None
        if not query:
            continue

        held_out = by_id[rng.choice(sorted(evidence_ids))]
        significance = held_out.metadata.get("significance")
        if significance is None:
            continue
        rows.append({
            **query,
            "molecular_profile_id": profile_id,
            "molecular_profile_name": profile.get("name"),
            "held_out_evidence_id": held_out.evidence_id,
            "remaining_evidence_ids": sorted(set(evidence_ids) - {held_out.evidence_id}),
            "label_significance": significance,
            "label_evidence_type": held_out.metadata.get("evidence_type"),
            "label_direction": held_out.metadata.get("evidence_direction"),
            "label_disease": held_out.metadata.get("disease"),
            "label_therapies": held_out.metadata.get("therapies"),
        })
    return rows


def build_negative_controls(limit_per_stratum: int = 150) -> tuple[list[dict], dict]:
    """Variants with no CIViC evidence, in three strata.

    The spec asked for negative controls including "some that look plausible,
    for example a novel missense in a gene that is well covered for other
    positions". Getting those from a whole-genome somatic callset turns out to be
    structurally hard, and the reason is worth recording:

    Of the 41,072 SEQC2 calls, only ~350 can be annotated to a gene at all. The
    rest are intergenic or intronic and are absent from dbNSFP, CADD and SnpEff
    alike -- they are novel positions that no variant database has ever seen. A
    genome-wide somatic callset is overwhelmingly non-coding; CIViC is a coding,
    clinical resource. Only 13 real calls land in a CIViC-covered gene without
    matching a CIViC record.

    So the hard stratum is built two ways and reported separately, because their
    provenance differs and pooling them would hide that:

      easy_observed     real SEQC2 calls, annotated, no CIViC match, gene absent
                        from CIViC entirely.
      hard_observed     real SEQC2 calls in a CIViC-covered gene, no CIViC match.
                        Only 13 exist. Real, but underpowered on its own.
      hard_constructed  protein changes at positions CIViC does NOT curate, in
                        genes CIViC covers heavily. Guaranteed true negatives by
                        construction, and exactly the "plausible novel missense in
                        a familiar gene" case the spec asked for. Labelled as
                        constructed everywhere it appears; never pooled with
                        observed calls in a headline number.
    """
    cache_path = ROOT / "data/cache/myvariant.jsonl"
    matcher = CivicMatcher(SNAPSHOT)
    civic_genes = {matcher._gene_of(v) for v in matcher.variants if matcher._gene_of(v)}
    civic_genes.discard(None)

    rng = random.Random(SEED)
    rows: list[dict] = []
    stats: dict = {}

    # ---- observed strata, from the real callset -------------------------------
    if cache_path.exists():
        from src.annotate import annotate      # noqa: PLC0415
        from src.parse_vcf import read_vcf     # noqa: PLC0415

        variants = list(read_vcf(ROOT / "data/seqc2/hc_snv.vcf.gz")) + \
                   list(read_vcf(ROOT / "data/seqc2/hc_indel.vcf.gz"))
        annotated, annotation_stats = annotate(variants, cache_path,
                                               verbose=False, offline=True)
        hard_observed, easy_observed = [], []
        for variant in annotated:
            if not variant.gene or variant.hg19_pos is None:
                continue                     # unannotated: cannot assert absence
            if matcher.match(variant):
                continue                     # present in CIViC; not a control
            row = {"query_key": variant.key, "gene": variant.gene,
                   "chrom": variant.chrom, "hg19_pos": variant.hg19_pos,
                   "ref": variant.ref, "alt": variant.alt,
                   "protein_changes": list(variant.protein_changes),
                   "provenance": "observed"}
            (hard_observed if variant.gene in civic_genes else easy_observed).append(row)
        rng.shuffle(hard_observed)
        rng.shuffle(easy_observed)
        stats["annotation"] = annotation_stats
        stats["hard_observed_available"] = len(hard_observed)
        stats["easy_observed_available"] = len(easy_observed)
        rows += [{**r, "stratum": "hard"} for r in hard_observed[:limit_per_stratum]]
        rows += [{**r, "stratum": "easy"} for r in easy_observed[:limit_per_stratum]]
    else:
        stats["error"] = "annotation cache missing; run scripts/annotate_seqc2.py"

    # ---- constructed hard stratum --------------------------------------------
    # Genes CIViC covers heavily, at protein positions CIViC does not curate.
    by_gene: dict[str, set[str]] = {}
    positions_by_gene: dict[str, set[int]] = {}
    for variant in matcher.variants:
        gene = matcher._gene_of(variant)
        if not gene:
            continue
        for change in matcher._protein_candidates(variant):
            by_gene.setdefault(gene, set()).add(change)
            digits = "".join(c for c in change if c.isdigit())
            if digits:
                positions_by_gene.setdefault(gene, set()).add(int(digits))

    # Genes carrying a gene-level category bucket ("BRAF MUTATION", a
    # loss-of-function rule) are excluded outright. In those genes CIViC really
    # does have evidence that applies to a novel variant, so a novel variant
    # there is NOT a true negative -- the first attempt at this set had 242 of
    # 265 candidates correctly rejected for exactly that reason. "No evidence
    # exists" is a much narrower condition than the spec assumed.
    bucketed_genes = {
        gene for gene, entries in matcher.by_category.items()
        if any(rule in SATISFIABLE_BY_SMALL_VARIANT for _, rule, _ in entries)
    }
    well_covered = sorted(
        g for g, changes in by_gene.items()
        if len(changes) >= 5 and g not in bucketed_genes
    )
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    constructed: list[dict] = []
    for gene in well_covered:
        known = positions_by_gene.get(gene, set())
        if not known:
            continue
        low, high = min(known), max(known)
        for _ in range(40):
            position = rng.randint(low, high)
            if position in known:
                continue                     # CIViC curates it: not a negative
            reference, alternate = rng.sample(amino_acids, 2)
            change = f"{reference}{position}{alternate}"
            if change in by_gene[gene]:
                continue
            constructed.append({
                "query_key": f"{gene}:p.{change}", "gene": gene,
                "chrom": None, "hg19_pos": None, "ref": None, "alt": None,
                "protein_changes": [change], "provenance": "constructed",
                "stratum": "hard_constructed",
            })
    rng.shuffle(constructed)
    # Verify by construction: none of these may match anything in CIViC.
    verified = [
        row for row in constructed
        if not matcher.match(QueryVariant(
            chrom="0", pos=0, ref="N", alt="N", gene=row["gene"],
            protein_changes=tuple(row["protein_changes"])))
    ]
    stats["hard_constructed_available"] = len(verified)
    stats["hard_constructed_rejected"] = len(constructed) - len(verified)
    stats["well_covered_genes"] = len(well_covered)
    stats["genes_excluded_gene_level_bucket"] = len(bucketed_genes)
    rows += verified[:limit_per_stratum]
    return rows, stats


def write(name: str, rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"  {len(rows):>5,} -> {path.relative_to(ROOT)}")


def main() -> int:
    print("building eval sets ...")
    retrieval = build_retrieval_holdout()
    write("retrieval_holdout", retrieval)

    assertions, skipped = build_assertion_holdout()
    write("assertion_holdout", assertions)
    print(f"        assertions skipped: {skipped}")

    leave_one_out = build_leave_one_out()
    write("leave_one_out", leave_one_out)

    negatives, negative_stats = build_negative_controls()
    write("negative_controls", negatives)
    print(f"        negative pool: {negative_stats}")

    summary = {
        "retrieval_holdout": len(retrieval),
        "assertion_holdout": len(assertions),
        "assertion_skipped": skipped,
        "leave_one_out": len(leave_one_out),
        "negative_controls": len(negatives),
        "negative_pool": negative_stats,
        "seed": SEED,
    }
    (ROOT / "results/eval_sets_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
