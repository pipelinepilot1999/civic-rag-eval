"""VCF -> QueryVariant, with the normalization steps that matter for joining.

`cyvcf2` (the spec's choice) is a fine reader but pulls in htslib. Since we only
need CHROM/POS/REF/ALT and a couple of INFO fields from a plain bgzipped VCF,
this reads it directly -- one fewer binary dependency in the Docker image, and
the normalization logic stays visible rather than hidden behind a library call.

Normalization performed, in order:
  1. multi-allelic split      one ALT per record
  2. left-alignment trimming  shared prefix/suffix removed, so that
                              CA>C and TCA>TC both become the same deletion
  3. chromosome naming        chr7 / 7 / CHR7 -> 7

Left-trimming is not cosmetic: without it an indel written one way never joins
to the same indel written another way, and the failure is silent.
"""

from __future__ import annotations

import gzip
import pathlib
from typing import Iterator

from .hgvs import normalize_chromosome
from .match import QueryVariant


def _trim(pos: int, ref: str, alt: str) -> tuple[int, str, str]:
    """Remove shared suffix then shared prefix, keeping at least one base."""
    ref, alt = ref.upper(), alt.upper()
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt, pos = ref[1:], alt[1:], pos + 1
    return pos, ref, alt


class EmptyVcfError(RuntimeError):
    """Raised when filtering removed every record.

    This is deliberately fatal. The SEQC2 truth set writes FILTER as
    `PASS;HighConf`, not `PASS`, so an exact-equality filter silently drops all
    39,447 calls and the pipeline reports a match rate of 0.00 -- which is
    indistinguishable from a broken retriever. An empty callset is always a bug;
    it should stop the run, not flow downstream.
    """


def read_vcf(
    path: str | pathlib.Path,
    pass_only: bool = True,
    strict: bool = True,
) -> Iterator[QueryVariant]:
    """Yield one QueryVariant per ALT allele.

    `pass_only` keeps records whose FILTER contains PASS (or is '.'/empty).
    Membership, not equality: FILTER is a semicolon-separated list, and callers
    routinely add their own confidence tags alongside PASS.
    """
    path = pathlib.Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    seen = kept = 0
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, _id, ref, alts, _qual, filt = fields[:7]
            seen += 1
            flags = set(filt.split(";"))
            if pass_only and not (flags & {"PASS", ".", ""}):
                continue
            kept += 1
            for alt in alts.split(","):
                if alt in {".", "*", ""}:
                    continue
                new_pos, new_ref, new_alt = _trim(int(pos), ref, alt)
                yield QueryVariant(
                    chrom=normalize_chromosome(chrom),
                    pos=new_pos,
                    ref=new_ref,
                    alt=new_alt,
                )
    if strict and seen and not kept:
        raise EmptyVcfError(
            f"{path.name}: all {seen:,} records were removed by the FILTER step. "
            "Check the FILTER column -- pass_only=False to disable filtering."
        )
