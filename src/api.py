"""Three endpoints. Not a chat interface -- see the spec's scope discipline.

  POST /interpret        one variant in, one cited interpretation out
  POST /interpret/vcf    a VCF upload, interpreted record by record
  GET  /metrics          the most recent evaluation results, served verbatim

/metrics exists so the numbers are part of the running system rather than a
claim in a README. If the eval has not been run, it says so instead of inventing
a result.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .corpus import build_documents
from .generate import get_backend, render_context, render_query
from .match import CivicMatcher, QueryVariant
from .parse_vcf import read_vcf
from .retrieve import HybridRetriever, LexicalRetriever, StructuredRetriever

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data/snapshots/latest"
MAX_VCF_RECORDS = 500

app = FastAPI(
    title="civic-rag-eval",
    description="Retrieval-augmented somatic variant interpretation, with the "
                "evaluation as the deliverable.",
    version="0.1.0",
)

_state: dict = {}


@app.on_event("startup")
def _load() -> None:
    documents = build_documents(SNAPSHOT)
    matcher = CivicMatcher(SNAPSHOT)
    structured = StructuredRetriever(matcher, documents)
    _state["retriever"] = HybridRetriever(structured, LexicalRetriever(documents))
    _state["documents"] = len(documents)
    _state["snapshot"] = json.loads((SNAPSHOT / "manifest.json").read_text())


class VariantRequest(BaseModel):
    gene: str | None = Field(None, examples=["BRAF"])
    protein_change: str | None = Field(None, examples=["V600E"])
    chrom: str | None = Field(None, examples=["7"])
    grch37_pos: int | None = Field(None, examples=[140453136])
    ref: str | None = Field(None, examples=["A"])
    alt: str | None = Field(None, examples=["T"])
    disease: str | None = Field(None, examples=["Skin Melanoma"])
    k: int = 5
    backend: str = "abstain"


def _interpret(request: VariantRequest) -> dict:
    variant = QueryVariant(
        chrom=request.chrom or "0", pos=request.grch37_pos or 0,
        ref=request.ref or "N", alt=request.alt or "N",
        gene=request.gene,
        protein_changes=(request.protein_change,) if request.protein_change else (),
        hg19_pos=request.grch37_pos,
    )
    retrieved = _state["retriever"].retrieve(variant, request.k)
    backend = get_backend(request.backend)
    interpretation = backend.generate(
        render_query(request.gene, request.protein_change, request.chrom,
                     request.grch37_pos, request.ref, request.alt, request.disease),
        render_context(retrieved),
    )
    return {
        "query": request.model_dump(),
        "retrieved": [
            {"citation": item.document.citation_key, "score": round(item.score, 4),
             "source": item.source, "rule": item.rule,
             "molecular_profile": item.document.molecular_profile_name,
             "text": item.document.text}
            for item in retrieved
        ],
        "interpretation": interpretation.as_dict(),
        "backend": backend.name,
    }


@app.post("/interpret")
def interpret(request: VariantRequest) -> dict:
    if not (request.gene or request.grch37_pos):
        raise HTTPException(400, "supply at least a gene or GRCh37 coordinates")
    return _interpret(request)


@app.post("/interpret/vcf")
async def interpret_vcf(file: UploadFile = File(...), k: int = 5,
                        backend: str = "abstain") -> dict:
    """Interpret an uploaded VCF.

    Coordinates are taken as GRCh37. A GRCh38 VCF must be annotated first
    (src/annotate.py) -- silently treating GRCh38 positions as GRCh37 is the
    exact failure this project exists to document, so it is refused rather than
    guessed at.
    """
    suffix = ".vcf.gz" if file.filename and file.filename.endswith(".gz") else ".vcf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(await file.read())
        path = pathlib.Path(handle.name)
    try:
        records = list(read_vcf(path, strict=False))[:MAX_VCF_RECORDS]
        results = [
            _interpret(VariantRequest(
                chrom=record.chrom, grch37_pos=record.pos, ref=record.ref,
                alt=record.alt, k=k, backend=backend))
            for record in records
        ]
    finally:
        path.unlink(missing_ok=True)
    return {"records": len(results), "truncated_at": MAX_VCF_RECORDS,
            "results": results}


@app.get("/metrics")
def metrics() -> dict:
    path = ROOT / "results/metrics.json"
    if not path.exists():
        raise HTTPException(404, "no evaluation has been run yet; see `make eval`")
    return json.loads(path.read_text())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": _state.get("documents"),
            "snapshot": (_state.get("snapshot") or {}).get("snapshot_date")}
