"""Minimal CIViC GraphQL client.

Deliberately dependency-free (stdlib only). The spec suggested `civicpy`, which
is maintained (v5.4.0, 2026-04-23) and would work. Talking to the GraphQL API
directly was chosen instead for two reasons:

1. Reproducibility. We pin a dated snapshot of exactly the fields we use and
   commit its checksum. A client library's own release cadence then cannot
   silently change what "the corpus" means between runs.
2. Field control. The eval depends on Molecular Profile structure and Assertion
   linkage; querying those explicitly makes the coupling visible in one file.

Every page is retried with exponential backoff. The API is public and
unauthenticated; CIViC content is CC0.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

API_URL = "https://civicdb.org/api/graphql"
PAGE_SIZE = 200
MAX_RETRIES = 5


class CivicError(RuntimeError):
    pass


def _post(query: str, variables: dict[str, Any] | None = None, timeout: int = 90) -> dict[str, Any]:
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "civic-rag-eval/0.1"},
    )
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
            if "errors" in body:
                raise CivicError(f"GraphQL errors: {body['errors']}")
            return body["data"]
        # See src/annotate.py: ConnectionResetError is an OSError, not a URLError.
        except (OSError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise CivicError(f"CIViC request failed after {MAX_RETRIES} attempts: {last_error}")


def paginate(query: str, root: str, variables: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Walk a Relay-style connection, yielding one node at a time.

    `query` must accept an `$after: String` variable and select
    `pageInfo { hasNextPage endCursor }` alongside `nodes`.
    """
    cursor: str | None = None
    while True:
        data = _post(query, {**(variables or {}), "after": cursor})
        connection = data[root]
        yield from connection["nodes"]
        page = connection["pageInfo"]
        if not page["hasNextPage"]:
            return
        cursor = page["endCursor"]


EVIDENCE_QUERY = """
query($after: String) {
  evidenceItems(first: %d, after: $after, status: ACCEPTED) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id name description
      evidenceType evidenceLevel evidenceDirection significance evidenceRating
      variantOrigin status
      # NB: EvidenceItem.variantHgvs is declared non-nullable in the CIViC
      # schema but returns null for some accepted items, which fails the whole
      # query. HGVS is taken from the variant records instead.
      therapyInteractionType
      therapies { id name ncitId }
      disease { id name doid }
      phenotypes { id name hpoId }
      source { id citationId sourceType citation publicationYear title }
      molecularProfile { id name }
      assertions { id }
    }
  }
}
""" % PAGE_SIZE

MOLECULAR_PROFILE_QUERY = """
query($after: String) {
  molecularProfiles(first: %d, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id name description
      molecularProfileScore
      evidenceCountsByType { diagnosticCount prognosticCount predictiveCount predisposingCount functionalCount oncogenicCount }
      variants { id name }
    }
  }
}
""" % PAGE_SIZE

VARIANT_QUERY = """
query($after: String) {
  variants(first: %d, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id name
      feature { id name }
      ... on GeneVariant {
        alleleRegistryId
        clinvarIds
        hgvsDescriptions
        variantAliases
        maneSelectTranscript
        variantTypes { id name soid }
        coordinates {
          referenceBuild ensemblVersion
          chromosome start stop referenceBases variantBases
          representativeTranscript
        }
      }
    }
  }
}
""" % PAGE_SIZE

ASSERTION_QUERY = """
query($after: String) {
  assertions(first: %d, after: $after, status: ACCEPTED) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id name summary description
      assertionType assertionDirection significance
      ampLevel variantOrigin status
      therapyInteractionType
      therapies { id name }
      disease { id name doid }
      phenotypes { id name }
      molecularProfile { id name }
      evidenceItems { id }
      nccnGuideline { name }
      regulatoryApproval fdaCompanionTest
    }
  }
}
""" % PAGE_SIZE
