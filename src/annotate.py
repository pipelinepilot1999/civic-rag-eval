"""Annotate GRCh38 calls with gene, protein change, and GRCh37 position.

This module exists because of docs/spec-review.md #2: CIViC coordinates are
GRCh37 (1,916 of 1,918 that declare a build), while somatic callsets -- including
the SEQC2 truth set used here -- are GRCh38. Joining the two without conversion
returns an empty set and looks exactly like a bug in the retriever.

MyVariant.info is used rather than a local VEP install. It returns, in one batch
request: `dbnsfp.genename` (gene), `dbnsfp.hgvsp` (protein changes across
transcripts) and `dbnsfp.hg19.start` (the GRCh37 position). That is annotation
and liftover together, with no 25 GB VEP cache to ship in the Docker image.

Tradeoff, stated plainly: VEP with a pinned cache is the production answer and
gives consequence terms and exon numbers that MyVariant does not. Variants whose
exon we therefore do not know are counted as `unmatchable_without: exon` in the
tier-3 report rather than being guessed at.

Results are cached to disk keyed by variant, so a rerun costs nothing.
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.request
from typing import Iterable, Sequence

from .hgvs import normalize_protein_change
from .match import QueryVariant

ENDPOINT = "https://myvariant.info/v1/variant"
BATCH_SIZE = 900          # service accepts 1000; leave headroom
FIELDS = "dbnsfp.genename,dbnsfp.hgvsp,dbnsfp.hg19,clinvar.gene.symbol,dbsnp.rsid"
MAX_RETRIES = 4


def _hgvs_g(variant: QueryVariant) -> str:
    """MyVariant query id in GRCh38 g. notation."""
    if len(variant.ref) == 1 and len(variant.alt) == 1:
        return f"chr{variant.chrom}:g.{variant.pos}{variant.ref}>{variant.alt}"
    if len(variant.ref) > len(variant.alt):                      # deletion
        start = variant.pos + 1
        end = variant.pos + len(variant.ref) - 1
        span = f"{start}_{end}" if end > start else f"{start}"
        return f"chr{variant.chrom}:g.{span}del"
    inserted = variant.alt[len(variant.ref):]                    # insertion
    return f"chr{variant.chrom}:g.{variant.pos}_{variant.pos + 1}ins{inserted}"


def _post(ids: Sequence[str]) -> list[dict]:
    payload = json.dumps({"ids": list(ids), "fields": FIELDS, "assembly": "hg38"}).encode()
    request = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "civic-rag-eval/0.1"},
    )
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        # OSError, not just URLError: MyVariant drops long-lived connections and
        # raises ConnectionResetError, which is an OSError but NOT a URLError.
        # Catching only URLError killed a 41,072-variant run at 35,100.
        except (OSError, TimeoutError, json.JSONDecodeError):
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt * 3)
    return []


def _first_gene(record: dict) -> str | None:
    dbnsfp = record.get("dbnsfp") or {}
    names = dbnsfp.get("genename")
    if isinstance(names, list) and names:
        return str(names[0]).upper()
    if isinstance(names, str):
        return names.upper()
    symbol = ((record.get("clinvar") or {}).get("gene") or {}).get("symbol")
    return str(symbol).upper() if symbol else None


def _protein_changes(record: dict) -> tuple[str, ...]:
    dbnsfp = record.get("dbnsfp") or {}
    raw = dbnsfp.get("hgvsp") or []
    if isinstance(raw, str):
        raw = [raw]
    found = {normalize_protein_change(item) for item in raw}
    return tuple(sorted(c for c in found if c))


def _hg19_position(record: dict) -> int | None:
    hg19 = ((record.get("dbnsfp") or {}).get("hg19") or {})
    start = hg19.get("start")
    return int(start) if start is not None else None


def annotate(
    variants: Iterable[QueryVariant],
    cache_path: str | pathlib.Path,
    verbose: bool = True,
    offline: bool = False,
) -> tuple[list[QueryVariant], dict]:
    """Return annotated copies plus a stats dict describing what was reachable.

    `offline=True` uses only what is already cached and never touches the
    network. Downstream steps use this so that one hung upstream request cannot
    stall set construction -- variants with no cached annotation are simply
    counted as unannotated and excluded, which is visible in the stats rather
    than silent.
    """
    variants = list(variants)
    cache_path = pathlib.Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if cache_path.exists():
        with cache_path.open() as handle:
            for line in handle:
                entry = json.loads(line)
                cache[entry["q"]] = entry["r"]

    queries = {variant.key: _hgvs_g(variant) for variant in variants}
    missing = sorted({q for q in queries.values() if q not in cache})
    if offline:
        if verbose and missing:
            print(f"offline: {len(missing):,} variants have no cached annotation "
                  f"and will be excluded", flush=True)
        missing = []
    if verbose and missing:
        print(f"annotating {len(missing):,} new variants "
              f"({len(queries) - len(missing):,} cached) ...", flush=True)

    with cache_path.open("a") as handle:
        for start in range(0, len(missing), BATCH_SIZE):
            chunk = missing[start:start + BATCH_SIZE]
            for record in _post(chunk):
                query_id = record.get("query")
                if query_id is None:
                    continue
                cache[query_id] = record
                handle.write(json.dumps({"q": query_id, "r": record}) + "\n")
            handle.flush()
            if verbose:
                print(f"  {min(start + BATCH_SIZE, len(missing)):,}/{len(missing):,}", flush=True)

    annotated: list[QueryVariant] = []
    stats = {"total": len(variants), "with_gene": 0, "with_protein_change": 0,
             "with_hg19": 0, "not_found": 0, "uncached": 0}
    for variant in variants:
        record = cache.get(queries[variant.key])
        if record is None:
            stats["uncached"] += 1
        record = record or {}
        if record.get("notfound"):
            stats["not_found"] += 1
        gene = _first_gene(record)
        changes = _protein_changes(record)
        hg19 = _hg19_position(record)
        stats["with_gene"] += bool(gene)
        stats["with_protein_change"] += bool(changes)
        stats["with_hg19"] += hg19 is not None
        annotated.append(QueryVariant(
            chrom=variant.chrom, pos=variant.pos, ref=variant.ref, alt=variant.alt,
            gene=gene, protein_changes=changes, hg19_pos=hg19,
            consequences=variant.consequences, exon=variant.exon,
        ))
    return annotated, stats
