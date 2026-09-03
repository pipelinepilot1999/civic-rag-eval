"""Protein-change normalization.

CIViC writes protein changes several ways for the same substitution:
`S214C` (variant name), `p.Ser214Cys` (NP_ HGVS), `SER214CYS` (alias).
Annotation sources add `p.Val600Glu` and `p.V600E` for one variant.

Everything is normalized to a single canonical form -- one-letter codes, no
`p.` prefix, uppercase -- so that `p.Val600Glu`, `V600E` and `VAL600GLU` all
become `V600E` and join to the same CIViC record.
"""

from __future__ import annotations

import re

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "TER": "*", "SEC": "U", "PYL": "O", "XAA": "X",
}

# Substitution: <aa><pos><aa>, either alphabet. Also covers nonsense (*) and
# the `=` synonymous marker.
_SUBSTITUTION = re.compile(
    r"^(?P<ref>[A-Z]{3}|[A-Z*])(?P<pos>\d+)(?P<alt>[A-Z]{3}|[A-Z*=])$"
)
# Frameshift / delins / dup / del / ins keep their suffix verbatim after the
# position, since there is no shorter canonical form that stays unambiguous.
_EXTENDED = re.compile(
    r"^(?P<ref>[A-Z]{3}|[A-Z])(?P<pos>\d+)(?P<rest>(FS|DEL|INS|DUP|DELINS|EXT).*)$"
)


def _one_letter(code: str) -> str | None:
    code = code.upper()
    if len(code) == 1:
        return code
    return THREE_TO_ONE.get(code)


def normalize_protein_change(raw: str | None) -> str | None:
    """Return a canonical one-letter protein change, or None if not one.

    >>> normalize_protein_change("p.Val600Glu")
    'V600E'
    >>> normalize_protein_change("SER214CYS")
    'S214C'
    >>> normalize_protein_change("V600E")
    'V600E'
    >>> normalize_protein_change("EXON 17 MUTATIONS") is None
    True
    """
    if not raw:
        return None
    text = raw.strip().upper()
    for prefix in ("P.", "P"):
        if text.startswith(prefix) and len(text) > len(prefix) and text[len(prefix)].isalpha():
            text = text[len(prefix):]
            break
    text = text.replace("(", "").replace(")", "")

    match = _SUBSTITUTION.match(text)
    if match:
        ref = _one_letter(match["ref"])
        alt = match["alt"] if match["alt"] == "=" else _one_letter(match["alt"])
        if ref and alt:
            return f"{ref}{match['pos']}{alt}"
        return None

    match = _EXTENDED.match(text)
    if match:
        ref = _one_letter(match["ref"])
        if ref:
            return f"{ref}{match['pos']}{match['rest']}"
    return None


def normalize_chromosome(chrom: str) -> str:
    """`chr7` / `7` / `CHR7` -> `7`; `chrM`/`MT` -> `MT`."""
    text = str(chrom).strip().upper()
    if text.startswith("CHR"):
        text = text[3:]
    if text in {"M", "MT"}:
        return "MT"
    return text
