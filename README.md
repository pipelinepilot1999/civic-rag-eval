# civic-rag-eval

Retrieval-augmented somatic variant interpretation, **with the evaluation as the
deliverable**.

Given a somatic variant, retrieve curated [CIViC](https://civicdb.org) evidence,
generate an interpretation constrained to cite evidence IDs, and measure how
often the system retrieves correctly, cites faithfully, contradicts curated
ground truth, and **confabulates when no evidence exists**.

Every number below was produced by `make eval`; the tables are generated from
`results/*.json` by `scripts/results_table.py`, never typed by hand.

---

## Results

CIViC snapshot `2026-09-02` · 4,908 evidence items ·
k=5 · hybrid retriever

| system | answer rate | agreement | contradiction | confabulation (hard) | citation validity | net benefit /100 |
|---|---|---|---|---|---|---|
| Opus 5 | 0.904 | 0.635 | 0.090 | 0.227 | 1.000 | 54.49 |
| Haiku 4.5 | 0.833 | 0.340 | 0.064 | 0.000 | 1.000 | 27.56 |
| echo top-1 (no model) | 1.000 | 0.500 | 0.128 | 0.807 | 1.000 | 37.18 |
| always-abstain (no model) | 0.000 | 0.000 | 0.000 | 0.000 | n/a | 0.0 |

**Read the last column first.** Net benefit is (correct − incorrect) per 100
variants. A system that abstains on everything scores exactly 0 — and scores
*perfectly* on the naive versions of the other four metrics. That row is in the
table permanently for exactly that reason.

### The finding

**Retrieval is what causes confabulation.**

| model | answer rate (RAG) | answer rate (no RAG) | agreement (RAG) | agreement (no RAG) | confab. hard (RAG) | confab. hard (no RAG) |
|---|---|---|---|---|---|---|
| Opus 5 | 0.904 | 0.000 | 0.635 | 0.000 | 0.227 | 0.000 |
| Haiku 4.5 | 0.833 | 0.000 | 0.340 | 0.000 | 0.000 | 0.000 |

With no retrieved context, Opus 5 abstains on 100% of variants and confabulates
on none. Give it evidence, and it answers 90% of the time at 0.635 agreement —
but invents a clinical significance for **22.7%** of plausible-looking variants
that have no CIViC record at all.

The mechanism is visible in the failures. Four different invented DDR2 variants —
`C358G`, `K594R`, `M562Q`, `E610M` — each returned `SENSITIVITYRESPONSE`, each
citing the *same five* real DDR2 evidence items. The model generalizes
gene-level evidence to any position in that gene. Every citation was valid; not
one was appropriate.

That is the number the spec predicted almost everyone skips, and it is the
project's reason to exist.

### Negative controls

| negative stratum (Opus 5) | n | confabulation rate |
|---|---|---|
| easy | 150 | 0.000 [0.000, 0.025] |
| hard | 13 | 0.154 [0.043, 0.422] |
| hard_constructed | 150 | 0.227 [0.167, 0.300] |
| pooled_observed_only | 163 | 0.012 [0.003, 0.044] |

`easy` = the gene has no CIViC presence. `hard` = the gene *is* covered at other
positions. Opus 5 abstains perfectly on unfamiliar genes and fails only on
familiar ones — a pooled number would have hidden that entirely.

`hard` (n=13) is every real SEQC2 call meeting the criterion; `hard_constructed`
resamples the same failure mode at n=150 to get a usable interval. The observed
stratum is small and its interval says so.

### Retrieval

| retriever | recall@1 | recall@5 | recall@10 | returned nothing |
|---|---|---|---|---|
| structured | 0.960 | 0.999 | 1.000 | 0.000 |
| lexical | 0.896 | 0.940 | 0.948 | 0.000 |
| hybrid | 0.960 | 0.999 | 1.000 | 0.000 |
| dense | 0.366 | 0.566 | 0.632 | 0.000 |
| hybrid_dense | 0.960 | 0.999 | 1.000 | 0.000 |

**A dictionary lookup beats the embeddings by 0.59 recall@1**, and the hybrid
adds nothing on top of the structured tier. Dense retrieval
(`all-MiniLM-L6-v2`) finds the right evidence item first only **36.6%** of the
time; the structured matcher does it **96.0%** of the time.

That is not a tuning failure. Variant→evidence lookup is a join on an
identifier — `BRAF` + `V600E` either matches a CIViC record or it does not — and
cosine similarity over prose is the wrong instrument for an exact-match problem.
The embedding step is the part of a "RAG system" everyone assumes is load-bearing,
and here it is the part that could be deleted with no loss.

It stays in the repo, behind an optional dependency, because the measurement is
the point.

### Model comparison

Haiku 4.5 never confabulates (0.000 on hard negatives) but reaches only 0.340
agreement — *below* the no-model echo baseline's 0.500. Opus 5 is the only
configuration that beats echo on both agreement and net benefit. Caution and
capability trade off, and a project reporting one model would have seen neither
half of that.

---

## What the original spec got wrong

Five substantive problems, found by checking the plan against the live API before
writing code. Full detail with measurements in [docs/spec-review.md](docs/spec-review.md).

1. **Every headline metric was gamed by constant abstention.** Four proposed
   metrics; a four-line function that always abstains scores perfectly on all
   four. Fixed by scoring answer rate alongside correctness, and adding net
   benefit.
2. **The coordinate join was capped at ~25% and silently failed at 0%.** CIViC is
   GRCh37 (1,242 of 5,069 variants are addressable by exact
   coordinate); modern somatic callsets are GRCh38. Joining without liftover
   returns an empty set, not an error. Fixed with a three-tier matcher reaching
   **82.9%** of CIViC versus **24.5%** for coordinates alone.
3. **"Citation checking is programmatic" was false.** Verifying a cited ID was
   retrieved is not verifying the claim follows from it. Reported as two separate
   metrics.
4. **The contradiction arm was circular** — it scored the model against a label
   printed on the evidence it was reading. Replaced with held-out CIViC
   Assertions and a leave-one-out design.
5. **The data premise was false.** No somatic VCF existed on either machine.
   Replaced with the public SEQC2/SomaticSeq HCC1395 truth set (39,447
   high-confidence somatic SNVs + 1,625 indels, GRCh38).

---

## Architecture

```
SEQC2 VCF (GRCh38)
   -> parse_vcf.py      multi-allelic split, left-trim normalization
   -> annotate.py       MyVariant.info: gene, protein change, GRCh37 position
   -> match.py          tier 1 coordinate | tier 2 gene+protein | tier 3 category
   -> retrieve.py       structured | lexical | dense | hybrid
   -> generate.py       constrained JSON, citations required, abstention allowed
   -> eval/             four arms + ablation, Wilson intervals throughout
```

The core pipeline is **standard library only**. `requirements.txt` covers the API
server; the dense retriever is optional and isolated in `requirements-dense.txt`,
since it lost.

## Run it

```bash
make ingest    # dated CIViC snapshot (~30s)
make data      # SEQC2 truth set + annotation
make sets      # build the evaluation sets
make eval      # all arms, free baselines, no API key needed
make eval-real # all arms against Claude (needs ANTHROPIC_API_KEY)
make test      # 41 tests
make api       # three endpoints on :8000
```

`make eval` runs every arm with the abstain and echo baselines and needs no
credentials — Arm 1 and the whole harness are verifiable without spending
anything.

## Reproducibility

- CIViC snapshot is dated and checksummed (`data/snapshots/2026-09-02/manifest.json`).
- Eval-set construction is seeded.
- Deterministic metrics are pinned in `baselines/` and enforced by
  `scripts/check_regression.py` in CI on every push.
- The model arms run nightly, not per-push: a nondeterministic billed job on
  every commit becomes a test that flakes and then gets disabled.

## Known limits

- **Fusions are out of scope.** 94 of the 118 unusable CIViC assertions concern
  fusions and rearrangements, which an SNV/indel VCF cannot express. This is a
  ceiling on the task, not a bug in the matcher.
- **n=29 on the gold assertion arm.** Reported, but the leave-one-out arm
  (n=156) carries the statistical weight. Both are shown; they are never pooled.
- **Annotation is 96.4% complete** (39,600 of 41,072 calls). Unannotated variants
  are excluded and counted, not silently dropped.
- **Scoring normalizes model vocabulary.** Both models emit non-canonical
  significance strings (`PREDICTIVE_RESISTANCE`, `PREDISPOSITION`) a few percent
  of the time. `canonical_significance()` folds these; the raw rate is reported
  as `answers_outside_civic_vocabulary`. A strict enum via structured outputs is
  the better fix.

## Data and licence

CIViC content is CC0 (Griffith Lab, Washington University). The SEQC2 HCC1395
truth set is public via NCBI. Neither is redistributed here — `make ingest` and
`make data` fetch them.
