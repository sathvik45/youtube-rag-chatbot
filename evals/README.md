# Retrieval eval

```
datasets/golden_dataset_SD.json   the answer key: 305 queries, 430 gold spans
scorers/Retrieval_metrics.py      grades a retriever against the key
scorers/validate_golden.py        checks the key itself is well-formed
```

`Retrieval_metrics.py` and `validate_golden.py` do different jobs and neither
substitutes for the other. The metrics module scores retrieved chunks against
the gold spans — it assumes the spans are right. `validate_golden.py` is what
tells you they are: a span pointing at the wrong 30 seconds still scores
cleanly, so nothing in the metrics module can catch a mislabelled row.

Run the validator whenever you edit the dataset. Run the metrics whenever you
change chunking, embeddings, or `retrieve()`.

## The dataset

305 queries over the 102 ByteByteGo transcripts in `data/transcripts/`. Every
one of the 102 videos appears in at least one query's scope, so a regression
anywhere in the corpus shows up somewhere in the set.

## Format

A JSON array of objects, one per query:

```json
{"id": "q001", "query": "difference between git merge and git rebase",
 "scope": ["0chZFIZLR_0"],
 "gold_spans": [{"video_id": "0chZFIZLR_0", "start_ms": 81720, "end_ms": 106500}],
 "query_type": "comparison", "note": "the two definitions back to back"}
```

`scope` is the set of videos the query is asked against. `gold_spans` are the
character-for-character regions that answer it. `note` is optional and, where
present, says what the row is actually testing.

`query_type` is one of `pointwise`, `terse`, `comparison`, `aggregation`,
`absent` — the five in the original spec, nothing added.

| type | rows | what it is |
|---|---|---|
| pointwise | 143 | one span answers it |
| aggregation | 65 | the answer is genuinely split across spans |
| comparison | 42 | two or more things held against each other |
| terse | 39 | keyword-shaped rewrite of a fuller question |
| absent | 16 | not covered in scope; correct answer is to return nothing |

430 gold spans total. 85 rows have more than one span, 41 span more than one
video, 45 have a multi-video scope. Span length: min 4s, median 29s, max 154s.

## Span convention

Spans start on a segment's `start_ms` and end on a segment's
`start_ms + duration_ms`, so a span never slices a segment in half. Note that
several transcripts use overlapping ASR windows (`7_wkWQ9rB5I`, `-RDyEFvnTXI`,
`LQuuoHTyYz8` and others), so a span will pull in the tail of the previous
window. That is a property of the source data, not a span error.

## Validating

```bash
python -m evals.scorers.validate_golden
```

Run from the repo root as a module -- both `evals.*` and `src.*` resolve that
way, and neither script manipulates `sys.path` any more.

Checks schema, id uniqueness, that every scoped video exists, that every gold
span's video is inside its row's scope, that `absent` rows have no spans and
nothing else is empty, and that both span edges land on real segment
boundaries. Add `--show` (optionally with ids) to print the transcript text each
span covers — the only way to confirm a span answers its query rather than
merely sitting near it.

## Wiring it to the retriever

`evaluate()` wants `retrieve_fn(query, video_ids, k) -> list[dict]` with
`video_id`, `start_ms`, `end_ms` and `chunk_index` keys. `rag.retriever.retrieve`
returns LangChain `Document` objects, so it needs an adapter:

```python
from src.rag.retriever import retrieve

def retrieve_fn(query, video_ids, k):
    return [d.metadata for d in retrieve(query, video_ids, k=k)]
```

That adapter is wired into `scorers/run_retrieval_eval.py` through an explicit
two-phase workflow. Capture candidate scores once from the real index, then
replay threshold choices locally:

```powershell
python -m evals.scorers.run_retrieval_eval `
  --capture-candidates evals/results/retrieval-candidates.json `
  --allow-live-retrieval

python -m evals.scorers.run_retrieval_eval `
  --replay-candidates evals/results/retrieval-candidates.json `
  --thresholds 0.10 0.15 0.20 0.25 `
  --report evals/results/threshold-sweep.json
```

The capture is the only operation that contacts Pinecone; replay is offline.
It applies the same score-threshold and per-video-cap ordering as production,
and stores reproducibility metadata without transcript text or API keys. See
[`docs/no-answer-gate.md`](../docs/no-answer-gate.md) for how to choose and
activate a threshold.

`junk_rate()` is the one part with no data behind it yet: it needs a set of
boilerplate chunk ids in `"{video_id}#{chunk_index}"` form. The channel intro
and the "subscribe at blog.bytebytego.com" outro repeat across all 102 videos,
which is ~200 near-identical vectors competing for slots, so it is worth
populating.

Sanity baseline, running the dataset through `evaluate()` with an oracle that
returns exactly the gold spans:

```
  k    hit  cover  vidRec    MRR
  1   1.00   0.84    0.93   0.95
  5   1.00   1.00    1.00   0.95
```

Coverage below 1.0 at k=1 is correct, not a bug: 85 rows have more than one gold
span and one chunk cannot cover them all. If a real run shows `cover` at k=10
well below these numbers, the gap is your retriever, not the key.

## What the set is built to catch

The corpus has real defects, and the rows lean on them deliberately.

**ASR mangling.** `7_wkWQ9rB5I` writes Kafka as "CFKA" and partitions as
"petitions". `z_NbVtbgBJw` writes Redis as "Reddus" and sorted set as "sort of
set". `LQuuoHTyYz8` writes load balancer as "low balancer". `BHwzDmr6d7s` writes
sargable as "solvable". `AWVTKBUnoIg` writes feature toggles as "future tacos".
`iJ_eIsA5E1U` writes load average as "low average". Rows q015, q036, q111, q134,
q212 and q224 query these using the *correct* spelling, so lexical retrieval
scores near zero and only semantic matching can find them.

**Near-duplicate documents.** `4vLxWqE94l4` and `PNRbanEKGtw` are the same
script, the second truncated at 1.3 min. `TlHvYWVUZyc` and `lv0DdVLZuHc` are two
Kubernetes explainers with heavy overlap. `yIAcHMJzqJc` and `Ajz6dBp_EB4` are two
"top 5 Kafka use cases" videos. q068/q069 and q160–q163 are paired so that the
same query must return a span under one scope and nothing under the other — a
deduping index that leaks across the pair fails one half.

**Sponsor reads.** `hltLrjabkiY` (90000–153920), `2g1G8Jr88xU` (126280–190720),
`7_wkWQ9rB5I` (113520–150840) and `taSmwcqdkQk` (90760–132400) carry mid-roll
ads. q045, q199 and q262 target content immediately adjacent to them; the
Snowflake ad in `taSmwcqdkQk` even name-drops Iceberg, which is exactly the term
q262 is about.

**Structural oddities.** `BTjxUS_PylA` has real content *after* its closing
outro (q186). `kGT4PcTEPP8` has 11 segments of ~28s each, far coarser than
anything else (q197, q265). `XpFsMB6FoOs` and `x3cANGNPyx0` contain duplicated
near-zero-duration segments from on-screen tables (q233, q277). `lv0DdVLZuHc` and
`JTp0TY_2hXM` have a null title, which breaks title-boosted ranking (q159, q281).
`RlM9AfWf1WU` carries double-encoded HTML entities (q244).

**Coverage traps.** q129 has seven sibling spans, q291 has six, q188 and q204
have five. On all of these `hit=True` is compatible with coverage below 0.2, so
report coverage separately or the metric hides the failure.

**Stance.** `OTfYFl3rzjg` is satire that reads as sincere AWS advice (q303,
q304). `fKc050dvNIE` at 78480–111780 lists microservices, CQRS and eventual
consistency as things Stack Overflow *did not* use, which makes it a magnet for
false positives on any of those terms (q276). q302 and q216 give opposite
answers on whether early-stage startups need tests.

## Caveat

Spans were placed by reading each transcript and are checked mechanically for
boundary alignment, but "does this span answer this query" is a judgement call,
not a verified fact. Where a row is borderline the `note` says so. Treat
disagreements as a reason to re-read the span, not as a retrieval bug.

---

# Generation eval

```
scorers/oracle_context.py       rebuilds the context the retriever SHOULD return
scorers/deepeval_judge.py       a DeepEval judge backed by Groq
scorers/run_generation_eval.py  generates answers and scores them
```

The retrieval eval answers "did the right chunks come back". This one answers
"given chunks, did the generator behave". They are separate on purpose: run the
generator on live retrieval and a bad score is unreadable, because bad retrieval
and bad generation produce the same number.

## Context is held fixed

Context comes from the golden set's `gold_spans`, not from Pinecone.
`splitters.chunk_transcript` is pure -- no model, no network -- so
`oracle_context.py` rebuilds exactly the chunks ingestion produced and keeps the
ones overlapping each gold span. The generator sees production-shaped input
(overlapping windows, ASR noise, boundaries mid-sentence) with only the
retrieval error removed.

Two consequences: **this eval needs no Pinecone and no embedding model**, just
transcripts on disk; and context is ordered chronologically, not by rank, so it
is blind to position effects. Testing those needs a different condition.

`absent` rows are excluded -- no gold spans means no oracle context, and an empty
context is the refusal path, not the generation path.

The block cap (`--max-blocks`, default 12) is filled round-robin across gold
spans rather than by global overlap. Rank globally and one long span takes every
slot on q129 (seven spans) or q291 (six), which silently turns an aggregation row
into a pointwise one.

## Running

```bash
python -m evals.scorers.run_generation_eval --save
python -m evals.scorers.run_generation_eval --n 8              # smoke test
python -m evals.scorers.run_generation_eval --ids q129 q291    # specific rows
python -m evals.scorers.run_generation_eval --concurrency 2 --throttle 1.0
```

Set `GRADER_MODEL` in `.env` first. It defaults to `LLM_MODEL`, which means the
generator grades its own homework -- self-preference bias, and scores come back
flattering. `openai/gpt-oss-20b` against a `gpt-oss-120b` generator buys
independence at the cost of a weaker judge, so hand-label ten rows and check the
judge agrees before trusting a number from it.

Sampling is seeded (`--seed`, default 0) and stratified by `query_type`. Keep the
seed fixed or two runs score different rows and nothing is comparable.

## When the table comes back blank

A `-` in a score column means the judge never returned a score for those rows.
The run prints a `JUDGE ERRORS` section underneath with the actual exception and
how many rows it hit -- read that first, because the means above it are computed
over whatever *did* score and are not comparable to a full run.

The usual cause is the grader checkpoint. Not every Groq model supports tool
calling, and DeepEval asks for structured output on every call. `deepeval_judge`
discovers what the model can do (tools -> json_mode -> raw text) once and then
sticks to it, but if all three fail the metric errors out.

Probe before you sweep:

```bash
python -m evals.scorers.run_generation_eval --judge-check
```

One call, ~200 tokens, prints the mode it settled on. If it reports `raw`,
expect occasional parse failures on the longer faithfulness prompts. If it fails
outright, change `GRADER_MODEL`.

The other cause is the token budget, and it is the one that actually bites.
Groq's on-demand tier meters **tokens** per minute, and gpt-oss-20b gets 8,000.
One faithfulness evaluation sends the retrieval context twice and costs 5-6k
tokens, so a single row can eat most of a minute's allowance. What you see when
this happens:

```
judge calls=39 retries=38 ... mode=raw
  x3  Timed out/cancelled while evaluating metric
  x1  Evaluation LLM outputted an invalid JSON
  x1  Error code: 429 ... tokens per minute (TPM): Limit 8000
```

Retries near-equal to calls means every call is waiting. The timeouts are
DeepEval killing metrics that are stuck in backoff, and `mode=raw` means the
rate limiting was severe enough to look like a missing capability -- which then
produces the JSON parse failures.

The judge paces itself against a token bucket (`--tpm`, default 8000) so calls
wait for budget instead of being rejected. Runs get slower and stop failing. The
run prints its estimated token cost and wall-clock time before starting; if that
number is unpleasant, the levers in order of effect are:

```
--max-blocks 4     every block is sent to the judge twice
--n 15             fewer rows
--truths-limit 10  caps what faithfulness extracts from the context
```

Raising `--tpm` above your actual tier limit does not help -- it just moves the
failure back to the server.

## The two metrics

Both are LLM-judged. Neither is arithmetic the way span overlap was -- read them
as estimates with error bars, and rerun a subset to see how much they move.

| metric | how it works | what a low score means |
|---|---|---|
| faithfulness | splits the answer into atomic claims, asks per claim whether the context supports it | the model asserted things the transcripts do not say |
| answer relevancy | splits the answer into statements, asks per statement whether it addresses the question | the model answered a nearby question, or padded |

They fail in opposite directions, which is why both are needed. Copy a context
block verbatim and faithfulness is ~1.0 while relevancy collapses. Answer crisply
from parametric memory and relevancy is high while faithfulness collapses.


## Is the judge any good?

A small judge agrees with everything. Watch for a run where every score is
exactly 1.00 and the reasons end in "Great job!" -- that is a rubber stamp, not
a measurement, and it will keep reading 1.00 through a real regression.

Before trusting the number, hand-label ten rows: read the answer against its
context and decide yourself whether each claim holds. If the judge agrees on
eight or more, the metric is usable. If it does not, the fix is a stronger
GRADER_MODEL, not a threshold change.

## Citation markers are stripped before scoring

The generator emits `[c1: "verbatim quote"]` inline, and those quotes are copied
character-for-character from the context. Left in, they inflate faithfulness (the
judge sees claims that trivially match the source) and depress relevancy (quote
fragments read as statements that do not answer the question). Metrics run on
`Citations.strip_answer(...)`. Pass `--raw-answer` to score the unstripped text
and measure the gap.

## The citation block is free

`Citations.resolve` / `verify_quote` is arithmetic -- longest contiguous token run
against the cited chunk -- so these cost nothing and measure something the judge
cannot: whether the model's *stated evidence* exists.

```
cite_rate          answers carrying at least one citation
quoted_cite_rate   citations carrying a checkable quote (bare [cN] cannot be verified)
quote_verify_rate  quotes found verbatim in the cited chunk
block_utilization  context blocks the answer actually used
hallucinated_ids   cited ids that were never in context
over_refusal_rate  refusals despite being handed the gold span
```

Read them against the judged scores:

```
faithful high + verify high  healthy
faithful high + verify low   right answer, invented quotes -- the app links those
faithful low  + verify high  quotes real, inferences reaching past them
block_utilization low        aggregation rows answered from one span of several
over_refusal_rate > 0        declined a question it was handed the answer to
```

`block_utilization` is the generator's analogue of retrieval `coverage`, and it
is the one to watch on `aggregation` rows.

## What this does not measure

Correctness. A faithful, relevant, fully-cited answer can still be the wrong
answer -- the judge checks claims against context, not against the world, and
nothing here compares the answer to a reference. That needs hand-written gold
answers on a subset.

Also unmeasured: refusal on missing context, robustness to distractors, conflict
handling (q302 vs q216), and the negation trap in `fKc050dvNIE`. All of those need
context conditions this file does not build.
