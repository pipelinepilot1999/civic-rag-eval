# Spec review: what the original plan got wrong

The original spec (`civic-rag-eval-spec.md`, 2026-09-02) was reviewed against the
live CIViC GraphQL API and against the actual contents of the author's machines
before any code was written. Every number below was measured, not assumed.

Two findings would have sunk the project as written. Three more would have
produced numbers that could not survive an interview question.

---

## 1. Every headline metric is gamed by constant abstention (blocking)

The spec makes abstention a first-class output — correctly. It then defines four
metrics:

| Metric | Score of a system that abstains on every input |
|---|---|
| Citation faithfulness (fraction with an unsupported claim) | **1.00 — perfect** |
| Contradiction rate | **0.00 — perfect** |
| Confabulation on negatives | **0.00 — perfect** |
| Confabulation without RAG | **0.00 — perfect** |

A four-line function that ignores its input and returns
`{"significance": "insufficient_evidence", "citations": []}` scores a perfect
result on the entire README table. Only Arm 1 (retrieval recall) is unaffected,
and Arm 1 does not touch the language model at all.

This is not a hypothetical failure. Abstention is the cheapest local optimum for
a model told that abstention is safe, and the spec's own prompt design pushes
toward it.

**Correction.** Every arm is reported as a *pair* — a correctness rate and the
answer rate it was achieved at — and the README leads with a single number that
cannot be gamed in either direction:

- **Answer rate** on variants that *do* have supporting evidence (over-abstention
  is now a visible failure, not a free win).
- **Abstention rate** on negative controls (under-abstention, the original Arm 4).
- **Net benefit** = correct answers − incorrect answers, per 100 variants, so
  confident wrongness is penalized and silence is neither rewarded nor punished.

A constant-abstain system now scores 0 on answer rate and 0 net benefit, which is
what it deserves. The constant-abstain baseline is included in the results table
as a permanent reference row, the same role the permutation null plays in
`multiomics-ml-brca`.

## 2. The variant→evidence join is capped at 41% and silently fails at 0% (blocking)

The spec flags the join as "the hard part" and says to document the match rate.
It is worse than that: the ceiling is structural, and the most likely failure
mode returns zero matches without raising anything.

Measured against the live API (2026-09-02):

| | Count | Share |
|---|---|---|
| CIViC variants total | 5,069 | |
| …carrying genomic coordinates | 1,922 | **37.9%** |
| …with no coordinates at all | 3,147 | 62.1% |
| …of those, fusions / rearrangements | 445 | |

(Counted per unique variant. Counting per *profile membership* instead gives
6,668 slots and 2,764 with coordinates — 41.5% — because variants are reused
across molecular profiles. Either way the ceiling is well under half.)

The 3,147 without coordinates are not data-entry gaps. They are *category*
variants — `NPM1 EXON 11 MUTATION`, `AMPLIFICATION`, `LOSS-OF-FUNCTION`,
`EXPRESSION`, 392 fusions. They describe classes of variant that no single VCF
coordinate can match, and they carry some of the most clinically actionable
evidence in the database. A coordinate-only matcher does not merely miss them;
it cannot represent them.

**And the build is wrong.** Of CIViC variants that declare a reference build:

- GRCh37: **1,916**
- GRCh38: **2**
- unset: 4

CIViC is a GRCh37 resource. Modern somatic pipelines emit GRCh38 — including the
SEQC2 truth set this project now uses (`##reference=GRCh38.d1.vd1.fa`). Joining
GRCh38 coordinates against GRCh37 coordinates does not throw an error. It
returns an empty set, and a match rate of 0.00 looks exactly like a bug in your
embedding code. This is the single most likely way the original plan would have
consumed a weekend.

**Correction.** A three-tier matcher, with the match rate reported *per tier* so
the ceiling is visible in the results rather than discovered in week three:

1. **Exact coordinate** — after explicit GRCh38→GRCh37 liftover, which is a named
   pipeline step with its own failure count, not an implicit assumption.
2. **Gene + protein change** (HGVSp) — requires an annotation step (VEP/SnpEff);
   this is what actually reaches most specific variants.
3. **Category rules** — gene + consequence class + exon, matched against the
   3,178 category variants by rule, not by embedding similarity.

## 3. "Citation checking is programmatic" is false as stated

The spec's Arm 2 says that because CIViC IDs are structured, faithfulness is
"checkable programmatically rather than by reading."

Checking that a cited ID appears in the retrieved set verifies that the ID is
**valid**. It says nothing about whether the claim the model attached to that ID
is **supported** by it. A model that retrieves EID116 (NPM1, diagnostic, AML) and
writes "confers sensitivity to imatinib [EID116]" passes the programmatic check
perfectly while being completely wrong.

These are two different metrics and the spec collapses them into one number.

**Correction.** They are reported separately:

- **Citation validity** — programmatic, 100% coverage, cheap. Genuinely automatic.
- **Citation support** — does the claim follow from the cited evidence? Scored on
  a stratified subsample by an LLM judge, with a human-labeled calibration set and
  reported judge–human agreement. Never presented as if it were free.

The honest version of the talking point is "I made *half* of citation checking
programmatic, and measured how well the automated half proxies the half that
isn't" — which is a better interview answer than the original claim, because it
survives follow-up.

## 4. Arm 3 (contradiction rate) is circular as written

"Hold out variants where CIViC records a clear clinical significance … scored
against the held-out label."

The clinical significance *is a field on the evidence item being retrieved*. If
the evidence item is in the index, the model reads the label out of its context
window and the arm measures transcription accuracy. If the evidence item is
removed from the index, the variant has no supporting evidence and the arm has
become Arm 4. There is no configuration in which the spec's Arm 3 measures what
it claims to measure.

**Correction.** Use the two-level structure CIViC actually has:

- **Corpus**: Evidence Items (4,908 accepted) — the retrievable units.
- **Held-out label**: Assertions (147 accepted) — curated summary judgments that
  aggregate multiple evidence items, and whose text is excluded from the index.

The model sees evidence and must reach the conclusion the curators reached. That
is a real inference step, not a lookup.

**Caveat, stated up front:** n=147 is small. At that size a rate of 0.20 carries a
95% Wilson interval of roughly ±0.07. Every rate in this project ships with an
interval for exactly this reason, and the README says so rather than quoting bare
point estimates.

## 5. The data premise is false — there is no somatic VCF

The spec assumes "your existing GATK4 output" and "your full somatic VCF."

Both machines were searched. What exists is entirely germline:
`clinvar.vcf.gz`, the GIAB HG002/HG003/HG004 trio, and rare-disease case VCFs
under `~/rdacmg`. There is no somatic callset anywhere. CIViC is predominantly a
somatic resource (3,636 of 4,908 evidence items are `SOMATIC`), so the mismatch
is real and not cosmetic.

**Correction.** The SEQC2/SomaticSeq **HCC1395** somatic reference truth set —
a real breast-cancer cell line, publicly downloadable from NCBI, GRCh38, with
high-confidence somatic SNVs and indels and a defined high-confidence region BED.
It is a better input than a private callset anyway: anyone can reproduce the
numbers, which is the whole point of the project.

---

## Smaller corrections

- **Molecular Profiles, not variants.** CIViC keys evidence to Molecular Profiles,
  595 of 5,665 of which are boolean combinations of multiple variants. A single
  VCF variant cannot satisfy these. Retrieval ground truth handles partial
  satisfaction explicitly instead of silently scoring them as misses.
- **recall@10 is close to meaningless here.** Only 1,974 of 5,665 molecular
  profiles have any accepted evidence, and the median profile that does has
  exactly **one** evidence item. Recall@10 against a single-item target mostly
  measures how much irrelevant context you are willing to stuff into the prompt.
  Reported, but the headline is recall@1 and recall@5, and top-k is treated as a
  precision/context-cost tradeoff rather than a number to maximize.
- **Chunking is a non-issue.** Evidence descriptions are short — median 480
  characters, p90 944. One item per chunk is correct and needs no experimentation.
  The spec allocates a decision step to this; that time moves to the matcher.
- **CI on every push does not work for this.** The eval is nondeterministic, costs
  money, and needs a secret. Corrected to: fixed seed and temperature, a small
  frozen subset, nightly rather than per-push, and a tolerance derived from
  measured run-to-run variance across repeated identical runs — not a guessed band.
- **No confidence intervals anywhere in the original.** Every rate here is a
  binomial proportion on a few hundred trials. Wilson intervals throughout.

## The baseline the spec never mentions

Variant→evidence lookup is a *join on an identifier*, not a semantic search
problem. Dense retrieval may well lose to a dictionary keyed on gene + protein
change.

If that is true, it needs to be in the README, because a reviewer will think of
it within thirty seconds of reading the architecture diagram. The eval therefore
includes a **structured-lookup baseline** alongside dense retrieval and the
hybrid matcher, and reports all three. "I built a RAG system and measured that
the RAG part earned its place" is a defensible result. So is "it didn't, and here
is the number." Not measuring it is the only outcome that isn't.
