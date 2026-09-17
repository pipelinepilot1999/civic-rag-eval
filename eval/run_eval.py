"""Run every evaluation arm and write results/metrics.json.

Arms, and what each one is protecting against:

  Arm 1  retrieval recall     bounds everything downstream. Run for all four
                              retrievers, including the structured baseline the
                              spec never proposed, so the results say whether
                              embeddings earned their place.
  Arm 2  citation behaviour   validity (programmatic) reported separately from
                              support (not free) -- docs/spec-review.md #3.
  Arm 3  agreement            assertion-labelled (gold, n=29) and leave-one-out
                              (powered, n=156), reported separately, never pooled.
  Arm 4  negative control     real somatic calls absent from CIViC, split into
                              hard (gene is covered) and easy (gene is not).
  Ablation                    arms 3 and 4 again with retrieval disabled.

Every arm reports its correctness metric *and* the answer rate it was achieved
at, because the whole metric set is otherwise maximized by constant abstention.

    python -m eval.run_eval --backend abstain              # no API key needed
    python -m eval.run_eval --backend anthropic --model claude-opus-5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval.metrics import (AGREE, CONTRADICT, HEDGE, classify_agreement,   # noqa: E402
                          citation_validity, net_benefit, recall_at_k, wilson)
from src.config import load_env                                           # noqa: E402
from src.corpus import build_documents                                    # noqa: E402
from src.generate import get_backend, render_context, render_query        # noqa: E402
from src.match import CivicMatcher, QueryVariant                          # noqa: E402
from src.retrieve import (DenseRetriever, HybridRetriever, LexicalRetriever,  # noqa: E402
                          StructuredRetriever)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data/snapshots/latest"
SETS = ROOT / "data/eval_sets"


def load(name: str) -> list[dict]:
    path = SETS / f"{name}.jsonl"
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def row_to_variant(row: dict) -> QueryVariant:
    changes = ()
    if row.get("protein_change"):
        changes = (row["protein_change"],)
    elif row.get("protein_changes"):
        changes = tuple(row["protein_changes"])
    return QueryVariant(
        chrom=str(row.get("chrom") or "0"),
        pos=int(row.get("pos") or row.get("hg19_pos") or 0),
        ref=row.get("ref") or "N",
        alt=row.get("alt") or "N",
        gene=row.get("gene"),
        protein_changes=changes,
        hg19_pos=row.get("hg19_pos"),
    )


# --------------------------------------------------------------------- arm 1

def arm1_retrieval(retrievers: dict, holdout: list[dict], ks=(1, 5, 10)) -> dict:
    results = {}
    for name, retriever in retrievers.items():
        hits = {k: 0 for k in ks}
        empty = 0
        for row in holdout:
            relevant = set(row["relevant_evidence_ids"])
            retrieved = [item.document.evidence_id
                         for item in retriever.retrieve(row_to_variant(row), max(ks))]
            if not retrieved:
                empty += 1
            for k in ks:
                hits[k] += recall_at_k(retrieved, relevant, k)
        results[name] = {
            **{f"recall@{k}": wilson(hits[k], len(holdout)).as_dict() for k in ks},
            "returned_nothing": wilson(empty, len(holdout)).as_dict(),
        }
    return results


# ------------------------------------------------------------------ arms 2-4

def _interpret(backend, retriever, row: dict, k: int, use_retrieval: bool,
               exclude: set[int] | None = None):
    if use_retrieval:
        retrieved = retriever.retrieve(row_to_variant(row), k + (len(exclude or ()) or 0))
        if exclude:
            retrieved = [r for r in retrieved if r.document.evidence_id not in exclude]
        retrieved = retrieved[:k]
    else:
        retrieved = []
    context = render_context(retrieved)
    query_text = render_query(
        row.get("gene"), row.get("protein_change"),
        row.get("chrom"), row.get("hg19_pos"),
        row.get("ref"), row.get("alt"), row.get("label_disease"),
    )
    return backend.generate(query_text, context), retrieved


def _interpret_all(backend, retriever, rows: list[dict], k: int, use_retrieval: bool,
                   holdout_field: str | None = None, workers: int = 1,
                   label: str = ""):
    """Interpret every row, in parallel, returning results in input order.

    Retrieval is done inside the worker but is pure CPU and thread-safe (the
    indexes are read-only after construction). The API call is the only thing
    worth parallelizing; 967 sequential calls take over an hour.
    """
    results: list = [None] * len(rows)
    done = [0]
    lock = threading.Lock()

    def work(index_row):
        index, row = index_row
        exclude = {row[holdout_field]} if holdout_field and row.get(holdout_field) else None
        out = _interpret(backend, retriever, row, k, use_retrieval, exclude)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(rows):
                print(f"    {label} {done[0]}/{len(rows)}", flush=True)
        return index, out

    if workers <= 1:
        for index, row in enumerate(rows):
            results[index] = work((index, row))[1]
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, out in pool.map(work, enumerate(rows)):
            results[index] = out
    return results


def arm23_agreement(backend, retriever, rows: list[dict], k: int,
                    label_field: str, use_retrieval: bool = True,
                    holdout_field: str | None = None, workers: int = 1,
                    label: str = "") -> dict:
    counts = {AGREE: 0, CONTRADICT: 0, HEDGE: 0}
    answered = abstained = unparseable = 0
    valid_citations = total_citations = 0
    outputs_with_invalid_citation = 0
    records = []

    computed = _interpret_all(backend, retriever, rows, k, use_retrieval,
                              holdout_field, workers, label)
    for row, (interpretation, retrieved) in zip(rows, computed):
        available = [item.document.citation_key for item in retrieved]

        if interpretation.significance == "UNPARSEABLE":
            unparseable += 1
        if interpretation.abstained:
            abstained += 1
            verdict = HEDGE
        else:
            answered += 1
            verdict = classify_agreement(interpretation.significance, row[label_field])
        counts[verdict] += 1

        valid, total = citation_validity(interpretation.citations, available)
        valid_citations += valid
        total_citations += total
        if total and valid < total:
            outputs_with_invalid_citation += 1

        records.append({
            "gene": row.get("gene"), "protein_change": row.get("protein_change"),
            "label": row.get(label_field), "predicted": interpretation.significance,
            "verdict": verdict, "citations": interpretation.citations,
            "available": available, "abstained": interpretation.abstained,
            "reasoning": interpretation.reasoning,
        })

    n = len(rows)
    return {
        "n": n,
        "answer_rate": wilson(answered, n).as_dict(),
        "abstention_rate": wilson(abstained, n).as_dict(),
        "agreement_rate": wilson(counts[AGREE], n).as_dict(),
        "contradiction_rate": wilson(counts[CONTRADICT], n).as_dict(),
        "hedge_rate": wilson(counts[HEDGE], n).as_dict(),
        "agreement_given_answered": wilson(counts[AGREE], answered).as_dict(),
        "net_benefit_per_100": (
            None if (nb := net_benefit(counts[AGREE], counts[CONTRADICT], n)) is None
            else round(nb, 2)
        ),
        "citation_validity": wilson(valid_citations, total_citations).as_dict(),
        "outputs_with_invalid_citation": wilson(outputs_with_invalid_citation, n).as_dict(),
        "unparseable": unparseable,
        "records": records,
    }


def arm4_negative(backend, retriever, rows: list[dict], k: int,
                  use_retrieval: bool = True, workers: int = 1,
                  label: str = "") -> dict:
    by_stratum: dict[str, dict] = {}
    for stratum in sorted({row.get("stratum") for row in rows} - {None}):
        subset = [row for row in rows if row.get("stratum") == stratum]
        if not subset:
            continue
        abstained = 0
        confident = []
        computed = _interpret_all(backend, retriever, subset, k, use_retrieval,
                                  None, workers, f"{label}:{stratum}")
        for row, (interpretation, _retrieved) in zip(subset, computed):
            if interpretation.abstained:
                abstained += 1
            else:
                confident.append({
                    "gene": row.get("gene"), "query": row.get("query_key"),
                    "provenance": row.get("provenance"),
                    "predicted": interpretation.significance,
                    "citations": interpretation.citations,
                    "reasoning": interpretation.reasoning,
                })
        by_stratum[stratum] = {
            "n": len(subset),
            "abstention_rate": wilson(abstained, len(subset)).as_dict(),
            "confabulation_rate": wilson(len(subset) - abstained, len(subset)).as_dict(),
            "examples": confident[:10],
        }
    # Observed and constructed negatives are never pooled into one headline
    # number: they have different provenance and pooling would hide that the
    # hard stratum is mostly constructed. Reported per stratum, plus a pooled
    # figure over observed calls only.
    observed = [key for key in by_stratum if key in {"hard", "easy"}]
    observed_n = sum(by_stratum[key]["n"] for key in observed)
    observed_confab = sum(
        by_stratum[key]["confabulation_rate"]["numerator"] for key in observed)
    by_stratum["pooled_observed_only"] = {
        "n": observed_n,
        "confabulation_rate": wilson(observed_confab, observed_n).as_dict(),
    }
    return by_stratum


# ------------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="abstain",
                        choices=["abstain", "echo", "anthropic"])
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--retriever", default="hybrid",
                        choices=["structured", "lexical", "dense", "hybrid"])
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--dense", action="store_true",
                        help="include the dense retriever in Arm 1 (needs torch)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel API calls; 1 for the free backends")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap rows per arm, for smoke runs")
    parser.add_argument("--out", default="results/metrics.json")
    args = parser.parse_args(argv)

    load_env()
    started = time.time()
    documents = build_documents(SNAPSHOT)
    matcher = CivicMatcher(SNAPSHOT)
    structured = StructuredRetriever(matcher, documents)
    lexical = LexicalRetriever(documents)

    retrievers = {
        "structured": structured,
        "lexical": lexical,
        "hybrid": HybridRetriever(structured, lexical),
    }
    if args.dense:
        dense = DenseRetriever(documents, cache_path=str(ROOT / "data/cache/dense.npy"))
        retrievers["dense"] = dense
        retrievers["hybrid_dense"] = HybridRetriever(structured, dense)

    backend = get_backend(args.backend, model=args.model)
    retriever = retrievers[args.retriever]

    def cap(rows):
        return rows[:args.limit] if args.limit else rows

    print(f"backend={backend.name}  retriever={args.retriever}  k={args.k}", flush=True)

    print("arm 1: retrieval recall ...", flush=True)
    arm1 = arm1_retrieval(retrievers, cap(load("retrieval_holdout")))

    print("arm 3a: assertion-labelled agreement ...", flush=True)
    arm3a = arm23_agreement(backend, retriever, cap(load("assertion_holdout")),
                            args.k, "label_significance",
                            workers=args.workers, label="arm3a")

    print("arm 3b: leave-one-out agreement ...", flush=True)
    arm3b = arm23_agreement(backend, retriever, cap(load("leave_one_out")),
                            args.k, "label_significance",
                            holdout_field="held_out_evidence_id",
                            workers=args.workers, label="arm3b")

    print("arm 4: negative controls ...", flush=True)
    arm4 = arm4_negative(backend, retriever, cap(load("negative_controls")), args.k,
                         workers=args.workers, label="arm4")

    print("ablation: retrieval disabled ...", flush=True)
    ablation = {
        "arm3b_no_retrieval": arm23_agreement(
            backend, retriever, cap(load("leave_one_out")), args.k,
            "label_significance", use_retrieval=False,
            holdout_field="held_out_evidence_id",
            workers=args.workers, label="abl3b"),
        "arm4_no_retrieval": arm4_negative(
            backend, retriever, cap(load("negative_controls")), args.k,
            use_retrieval=False, workers=args.workers, label="abl4"),
    }

    results = {
        "run": {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "backend": backend.name, "retriever": args.retriever, "k": args.k,
            "model": args.model if args.backend == "anthropic" else None,
            "snapshot": json.loads((SNAPSHOT / "manifest.json").read_text())["snapshot_date"],
            "corpus_documents": len(documents),
            "seconds": round(time.time() - started, 1),
        },
        "matcher_coverage": matcher.coverage_report(),
        "cost": backend.cost_report() if hasattr(backend, "cost_report") else None,
        "arm1_retrieval": arm1,
        "arm3a_assertion_agreement": arm3a,
        "arm3b_leave_one_out_agreement": arm3b,
        "arm4_negative_controls": arm4,
        "ablation": ablation,
    }

    out_path = pathlib.Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    try:
        shown = out_path.relative_to(ROOT)
    except ValueError:
        shown = out_path
    print(f"\nwrote {shown} in {results['run']['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
