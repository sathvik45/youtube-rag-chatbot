# Calibrating the no-answer evidence gate

## What this protects

Similarity search always finds the nearest chunks, even when none of them
actually answer the question. The evidence gate keeps only chunks whose score
is at least `SCORE_THRESHOLD`. When none qualify, the graph takes its
structural refusal branch:

```text
retrieve candidates
→ keep score >= SCORE_THRESHOLD
→ no qualifying chunks
→ refused_no_context
→ no answer-generation call and no citations
```

The comparison is inclusive: a score exactly equal to the threshold is kept.
The threshold runs before the playlist per-video cap, and the same shared
policy is used by production retrieval and offline replay.

`SCORE_THRESHOLD` deliberately defaults to unset (`None`). That keeps existing
behavior while the value is unknown; choosing a random cutoff can make a useful
video look empty. Enable it only after the calibration below.

## Step 1: Validate the offline answer key

This reads local transcript files and makes no Pinecone or LLM call:

```powershell
.\.venv\Scripts\python.exe -m evals.scorers.validate_golden
```

It should report the row count, the 16 `absent` questions, and aligned spans.

## Step 2: Capture real candidate scores once

This is the only live operation. It embeds each golden question and queries
the configured Pinecone index once, so use a development index and run it
deliberately. The acknowledgement flag prevents an accidental provider sweep.

```powershell
.\.venv\Scripts\python.exe -m evals.scorers.run_retrieval_eval `
  --capture-candidates evals/results/retrieval-candidates.json `
  --allow-live-retrieval
```

The capture contains candidate rank, score, video ID, chunk ID, and timestamps.
It never saves transcript text, API keys, or prompts. It also records the
dataset fingerprint, git state, index identity, embedding model, and chunking
settings so a future comparison has context. `evals/results/` is ignored by
Git.

Do not put this command in CI or a startup script. A capture becomes stale if
you change the Pinecone index, embedding model, chunking, `RETRIVE_K`, or the
playlist cap.

## Step 3: Sweep thresholds offline

This next command uses only the saved JSON capture. It does not import the
retriever, load embeddings, or contact Pinecone:

```powershell
.\.venv\Scripts\python.exe -m evals.scorers.run_retrieval_eval `
  --replay-candidates evals/results/retrieval-candidates.json `
  --thresholds 0.10 0.15 0.20 0.25 0.30 `
  --report evals/results/threshold-sweep.json
```

Read the row for the production `RETRIVE_K` (currently the normal chat context
size). Compare these measurements:

| Measurement | Good direction |
|---|---|
| `absent_refusal_rate` | Higher: unrelated questions returned no evidence. |
| `answerable_refusal_rate` | Lower: real questions were not incorrectly rejected. |
| `answerable_hit_rate` / `answerable_coverage` | Keep close to the no-gate baseline. |
| `newly_refused_ids` | Read these boundary cases before choosing a value. |

Choose the **lowest** threshold that meets the refusal target you set for the
product without an unacceptable loss of answerable coverage. With only 16
absent rows, one changed row moves the refusal rate by 6.25%, so inspect the
IDs rather than trusting one rounded percentage.

## Step 4: Activate the selected value

In your ignored local `.env`, add the chosen numeric value:

```dotenv
SCORE_THRESHOLD=<chosen-after-replay>
```

Restart the API and worker processes so their settings are reloaded. The score
scale is tied to this project’s cosine Pinecone index and normalized embedding
model; it is not a portable industry constant.

## Step 5: Verify the behavior safely

The deterministic tests use fake vector results and never call an external
provider:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_no_answer_gate.py tests\test_retrieval_threshold_replay.py -q
```

For a real manual chat check, use a ready development source and one question
you know is not covered. Expect an HTTP `200` response with
`assistant_message.status: "refused_no_context"` and an empty citation list.
This confirms refusal is a valid grounded outcome, not a server error.

## Important limits

The gate proves only that no chunk met the score cutoff. A high-scoring chunk
can still be incomplete or misleading, so citation verification and generation
evaluation remain necessary. On a follow-up question, the query-rewrite model
may run before retrieval; the gate guarantees that **answer generation** does
not run without qualifying evidence.
