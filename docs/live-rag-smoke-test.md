# Opt-in live RAG smoke test

## Purpose

Use this guide to make **one deliberate, real** chat request after changing
provider configuration, retrieval behavior, or the chat API. It checks the
complete path: an already-ingested video, embeddings, Pinecone retrieval,
Groq generation, and persisted citations.

This is a manual development check, not a pytest test. Reading this guide,
starting the API, and opening `/docs` do not call an LLM, embed a query, or
search Pinecone. The live call happens only when you explicitly send
`POST /threads/{thread_id}/messages` in step 5.

> **Cost and data notice:** the final request can use provider quota and sends
> the question plus retrieved transcript context to the configured LLM. Use a
> non-sensitive public video, a small question, and a disposable development
> account. Do not run this against production data as a routine test.

## Before you begin

1. Use a local/development database, with migrations already applied.
2. Keep provider credentials in your existing ignored `.env` file or secret
   manager. Do **not** paste `GROQ_API_KEY`, `PINECONE_API_KEY`, database URLs,
   or tokens into this document, shell history, source code, or Swagger
   request bodies.
3. Confirm the private configuration contains the provider values your current
   RAG setup needs:

   - `GROQ_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`
   - optional model settings such as `LLM_MODEL` and `EMBEDDING_MODEL`

4. Use a source you have **already ingested**. Its status must be `ready`, or
   `partial` with at least one ready video. Do not submit a new source as part
   of this smoke test: ingestion is a separate provider workflow.
5. Leave `CHAT_RAG_TIMEOUT_SECONDS` at its default of `45` seconds unless you
   are intentionally testing timeout behavior. Its allowed value is greater
   than zero and at most `300` seconds.

## Run the check

### 1. Start the API

From the repository root, activate the normal development environment and run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs>. At this point, no RAG provider request has
been made.

### 2. Authenticate as a development user

In the interactive API documentation:

1. Register a disposable user with `POST /auth/register`, if needed.
2. Authorize using `POST /auth/token` and that user's credentials.
3. Use the resulting authorization only in the local Swagger session.

Do not use a personal production account or copy access tokens into committed
files.

### 3. Verify an already-ready source

Call `GET /sources/{source_id}` for the source you plan to use. Continue only
when its response has either:

- `status: "ready"`, or
- `status: "partial"` and `video_counts.ready` is greater than zero.

If the source is still `pending` or `processing`, wait for the ingestion worker
to complete. If it is `failed` or has zero ready videos, fix ingestion before
testing chat; a chat smoke test cannot repair a missing transcript or index.

### 4. Create a source-scoped thread

Call `POST /threads` with the verified `source_id` and a descriptive test
title. Save the returned thread `id`.

```json
{
  "source_id": "your-ready-source-id",
  "title": "Manual live RAG smoke test"
}
```

Creating a thread only records database state. It does not contact the RAG
providers.

### 5. Make exactly one known-answer chat request

Call `POST /threads/{thread_id}/messages` using the thread ID from step 4.
Ask a short question whose answer is plainly stated in the selected video, for
example a term, recommendation, or sequence you noted while watching it.

```json
{
  "content": "Ask one specific question answered in this video."
}
```

This is the explicit live-provider call. Do not automate retries or put this
request into a startup script or CI job.

For a successful known-answer check, expect HTTP `200` and verify all of the
following:

- `assistant_message.status` is `"answered"`.
- `assistant_message.content` answers the selected video's question.
- `assistant_message.citations` is non-empty.
- Every citation points to a video belonging to this thread's source and has a
  usable YouTube timestamp URL.
- `assistant_message.grounded` is `true` when the citation quote was verified.

### 6. Confirm the durable record

Call `GET /threads/{thread_id}/messages`. You should see the user question and
the assistant response in chronological order. The answer's citations should
match the response from step 5.

## Expected failures and safe next actions

| Result | Meaning | Next action |
|---|---|---|
| `409` | The thread has no ready videos in its source scope. | Finish or repair ingestion; do not broaden the thread's scope manually. |
| `503` | The graph exceeded the request deadline or a provider failed. The service attempts to store a safe `temporary_error` turn. | Inspect safe application logs for duration and failure type, then correct configuration or provider availability before one retry. |
| `200` with `refused_no_context` | Retrieval found no usable context for the question. | Check that the selected video is really ready and ask one more specific, in-video question. |
| `404` | The thread does not exist for the authenticated user. | Reauthorize as the owner and use the correct thread ID. |

The request-level deadline prevents the API from waiting indefinitely. It does
not necessarily cancel work a provider has already started, so use a single
controlled request rather than repeated rapid retries.

## What to record

For a development note, record only non-sensitive facts: the date, source and
thread IDs, HTTP status, final message status, number of citations, and total
duration. Do not record question text, answer text, tokens, credentials, or
database connection strings in shared logs or tickets.
