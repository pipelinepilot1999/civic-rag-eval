"""Retrieval, in three flavours, so the eval can say which one earned its place.

The spec proposed dense retrieval only. But variant -> evidence lookup is a join
on an identifier, not a semantic-similarity problem, and a reviewer will think of
that within thirty seconds of seeing the architecture. So the structured baseline
is a first-class retriever here, not an afterthought:

  StructuredRetriever  the three-tier matcher: gene/coordinate/protein/category.
                       No embeddings. This is the honest baseline.
  LexicalRetriever     TF-IDF cosine over evidence text. Pure Python -- at 4,908
                       documents a vector store is not load-bearing.
  DenseRetriever       sentence-transformers embeddings. Optional import: the
                       repo runs without torch installed.
  HybridRetriever      structured hits first, remaining slots filled by a text
                       retriever. Expected to win; the eval reports whether it does.

Every retriever returns the same `Retrieved` records so `eval/run_eval.py` can
swap them without special-casing.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter, defaultdict
from typing import Protocol, Sequence

from .corpus import EvidenceDoc
from .match import CivicMatcher, QueryVariant

# Evidence level A is strongest. Used to order structured hits, which arrive
# unranked -- a set membership test has no notion of "how relevant".
LEVEL_RANK = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, None: 5}

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]+|[A-Z]\d+[A-Z*]")


@dataclasses.dataclass(frozen=True)
class Retrieved:
    document: EvidenceDoc
    score: float
    source: str          # which retriever produced it, for hybrid attribution
    rule: str = ""       # for structured hits: which tier/rule fired


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


class Retriever(Protocol):
    def retrieve(self, variant: QueryVariant, k: int) -> list[Retrieved]: ...


class StructuredRetriever:
    """Matcher -> molecular profiles -> their evidence items. No embeddings."""

    name = "structured"

    def __init__(self, matcher: CivicMatcher, documents: Sequence[EvidenceDoc]):
        self.matcher = matcher
        self.by_profile: dict[int, list[EvidenceDoc]] = defaultdict(list)
        for document in documents:
            self.by_profile[document.molecular_profile_id].append(document)

    def retrieve(self, variant: QueryVariant, k: int) -> list[Retrieved]:
        # Tier order is the ranking signal: an exact coordinate hit outranks a
        # gene-level "BRAF MUTATION" bucket hit for the same query.
        tier_rank = {"coordinate": 0, "protein": 1, "category": 2}
        candidates: list[tuple[tuple, EvidenceDoc, str]] = []
        seen: set[int] = set()
        for match in self.matcher.match(variant):
            for profile_id in match.molecular_profile_ids:
                for document in self.by_profile.get(profile_id, []):
                    if document.evidence_id in seen:
                        continue
                    seen.add(document.evidence_id)
                    sort_key = (
                        tier_rank.get(match.tier, 9),
                        LEVEL_RANK.get(document.metadata.get("evidence_level"), 5),
                        -(document.metadata.get("evidence_direction") == "SUPPORTS"),
                        document.evidence_id,
                    )
                    candidates.append((sort_key, document, match.rule))
        candidates.sort(key=lambda item: item[0])
        return [
            Retrieved(document=document, score=1.0 / (rank + 1),
                      source=self.name, rule=rule)
            for rank, (_, document, rule) in enumerate(candidates[:k])
        ]


class LexicalRetriever:
    """TF-IDF cosine similarity, implemented directly.

    Pure Python and no vector store. At 4,908 documents an exact scan takes
    milliseconds; ChromaDB would add a dependency and an index-freshness failure
    mode without changing a single number in the results table.
    """

    name = "lexical"

    def __init__(self, documents: Sequence[EvidenceDoc]):
        self.documents = list(documents)
        document_frequency: Counter[str] = Counter()
        tokenized: list[Counter[str]] = []
        for document in self.documents:
            counts = Counter(tokenize(document.text))
            tokenized.append(counts)
            document_frequency.update(counts.keys())

        total = len(self.documents)
        self.idf = {
            term: math.log((total + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self.vectors: list[dict[str, float]] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for index, counts in enumerate(tokenized):
            vector = {
                term: (1 + math.log(count)) * self.idf[term]
                for term, count in counts.items()
            }
            norm = math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0
            vector = {term: weight / norm for term, weight in vector.items()}
            self.vectors.append(vector)
            for term in vector:
                self.postings[term].append(index)

    def query_text(self, variant: QueryVariant) -> str:
        parts = [variant.gene or "", *variant.protein_changes]
        return " ".join(part for part in parts if part)

    def retrieve(self, variant: QueryVariant, k: int) -> list[Retrieved]:
        return self.retrieve_text(self.query_text(variant), k)

    def retrieve_text(self, text: str, k: int) -> list[Retrieved]:
        counts = Counter(tokenize(text))
        if not counts:
            return []
        query_vector = {
            term: (1 + math.log(count)) * self.idf.get(term, 0.0)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(w * w for w in query_vector.values())) or 1.0
        query_vector = {term: w / norm for term, w in query_vector.items()}

        scores: dict[int, float] = defaultdict(float)
        for term, weight in query_vector.items():
            if weight <= 0:
                continue
            for index in self.postings.get(term, ()):
                scores[index] += weight * self.vectors[index].get(term, 0.0)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [
            Retrieved(document=self.documents[index], score=score, source=self.name)
            for index, score in ranked
        ]


class DenseRetriever:
    """sentence-transformers embeddings, cosine similarity.

    Imported lazily so that the rest of the project -- ingest, matching, the
    structured and lexical arms, and every metric -- runs on a machine with no
    torch installed.
    """

    name = "dense"

    def __init__(self, documents: Sequence[EvidenceDoc],
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_path: str | None = None):
        from sentence_transformers import SentenceTransformer   # noqa: PLC0415
        import numpy as np                                       # noqa: PLC0415

        self.np = np
        self.documents = list(documents)
        self.model = SentenceTransformer(model_name)
        if cache_path and __import__("pathlib").Path(cache_path).exists():
            self.embeddings = np.load(cache_path)
        else:
            self.embeddings = self.model.encode(
                [document.text for document in self.documents],
                batch_size=64, show_progress_bar=True, normalize_embeddings=True,
            )
            if cache_path:
                np.save(cache_path, self.embeddings)

    def retrieve(self, variant: QueryVariant, k: int) -> list[Retrieved]:
        parts = [variant.gene or "", *variant.protein_changes]
        return self.retrieve_text(" ".join(p for p in parts if p), k)

    def retrieve_text(self, text: str, k: int) -> list[Retrieved]:
        if not text.strip():
            return []
        query = self.model.encode([text], normalize_embeddings=True)[0]
        scores = self.embeddings @ query
        top = self.np.argsort(-scores)[:k]
        return [
            Retrieved(document=self.documents[int(i)], score=float(scores[int(i)]),
                      source=self.name)
            for i in top
        ]


class HybridRetriever:
    """Structured hits first; a text retriever fills the remaining slots.

    Ordering is the point. Structured hits are precise but can return nothing;
    the text retriever always returns something, which is useful for recall and
    dangerous for confabulation. Putting structured first means the model sees
    the exact-match evidence at the top of its context when it exists.
    """

    name = "hybrid"

    def __init__(self, structured: StructuredRetriever, text_retriever: Retriever):
        self.structured = structured
        self.text_retriever = text_retriever

    def retrieve(self, variant: QueryVariant, k: int) -> list[Retrieved]:
        results = self.structured.retrieve(variant, k)
        if len(results) >= k:
            return results[:k]
        seen = {item.document.evidence_id for item in results}
        for item in self.text_retriever.retrieve(variant, k * 3):
            if item.document.evidence_id in seen:
                continue
            results.append(item)
            seen.add(item.document.evidence_id)
            if len(results) >= k:
                break
        return results[:k]
