"""Annotate the SEQC2 HCC1395 callset and report tier-by-tier match rates."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.parse_vcf import read_vcf
from src.annotate import annotate
from src.match import CivicMatcher

ROOT = pathlib.Path(__file__).resolve().parent.parent
variants = list(read_vcf(ROOT / "data/seqc2/hc_snv.vcf.gz")) + \
           list(read_vcf(ROOT / "data/seqc2/hc_indel.vcf.gz"))
print(f"{len(variants):,} somatic calls from SEQC2 HCC1395 (GRCh38)", flush=True)

annotated, stats = annotate(variants, ROOT / "data/cache/myvariant.jsonl")
print("\nannotation:", json.dumps(stats, indent=2), flush=True)

matcher = CivicMatcher(ROOT / "data/snapshots/latest")
tier_counts = {"coordinate": 0, "protein": 0, "category": 0}
matched_any = 0
rows = []
for variant in annotated:
    hits = matcher.match(variant)
    if not hits:
        continue
    matched_any += 1
    for tier in {h.tier for h in hits}:
        tier_counts[tier] += 1
    rows.append({
        "query": variant.key, "gene": variant.gene,
        "protein_changes": list(variant.protein_changes),
        "hg19_pos": variant.hg19_pos,
        "hits": [{"variant_id": h.variant_id, "name": h.civic_variant_name,
                  "tier": h.tier, "rule": h.rule,
                  "molecular_profile_ids": list(h.molecular_profile_ids)} for h in hits],
    })

out = ROOT / "data/seqc2_civic_matches.jsonl"
with out.open("w") as handle:
    for row in rows:
        handle.write(json.dumps(row) + "\n")

summary = {
    "somatic_calls": len(annotated),
    "matched_at_least_one_civic_variant": matched_any,
    "match_rate": round(matched_any / len(annotated), 5),
    "calls_hitting_tier": tier_counts,
    "annotation": stats,
}
(ROOT / "results/match_report.json").write_text(json.dumps(summary, indent=2) + "\n")
print("\nmatch report:", json.dumps(summary, indent=2))
