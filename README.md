# YouTube RAG Service and Evaluation Toolkit

<strong>rag_eval</strong> is a source-scoped, citation-aware RAG application for YouTube videos and playlists. Users register, submit a video or playlist, wait for it to be indexed, and chat only over the videos that belong to that submitted source.

It combines FastAPI and a same-origin browser UI with PostgreSQL, a durable ingestion queue, Supadata transcript retrieval, HuggingFace embeddings, Pinecone vector search, Groq models, LangGraph orchestration, and retrieval/generation evaluation tools.

## What it does

- Registers users with Argon2 password hashing and JWT bearer authentication.
- Accepts YouTube video and playlist URLs.
- Creates durable ingestion jobs with leases, retries, and stale-job recovery.
- Fetches, caches, chunks, embeds, and indexes transcripts.
- Provides source-bound chat threads with saved history and pagination.
- Retrieves only from ready videos in the active thread's source.
- Refuses structurally when no qualifying evidence is retrieved.
- Returns timestamped citations and verifies the model's quoted evidence.
- Serves a framework-free browser UI at the root URL.
- Includes golden-set retrieval calibration and generation evaluation tooling.

## Architecture

~~~mermaid
flowchart LR
    Client[Browser or API client] --> API[FastAPI]
    API --> DB[(PostgreSQL)]
    API --> Submit[Source submission]
    Submit --> DB

    Worker[Separate ingestion worker] --> DB
    Worker --> Supadata
    Supadata --> Cache[Local transcript cache]
    Cache --> Embed[HuggingFace embeddings]
    Embed --> Pinecone[Pinecone]
    Worker --> DB

    API --> Chat[Chat service]
    Chat --> DB
    Chat --> Graph[LangGraph]
    Graph --> Pinecone
    Graph --> Groq
    Graph --> Chat
~~~

The API and worker are deliberately separate processes:

1. The API accepts requests, serves the UI, and creates ingestion jobs.
2. The worker claims and processes jobs. Submitting a source does not start a worker in the API process.

## Application lifecycle

### Authentication

<code>POST /auth/register</code> normalizes an email, validates a 12–128 character password, hashes it with Argon2, and creates a user. <code>POST /auth/token</code> accepts OAuth2 form fields; use the email as <code>username</code>. It returns an HS256 bearer token.

Sources and threads are owner-scoped. A missing resource and a resource owned by another user intentionally look the same to the caller.

### Source submission and ingestion

<code>POST /sources</code> accepts a YouTube video or playlist URL. It saves a pending source, resolves its videos outside a database transaction, creates or reuses global video records, creates <code>source_videos</code> links, and queues work only when it is needed.

~~~text
queued or retrying job
  -> atomically claimed by a worker
  -> transcript fetched from Supadata
  -> JSON cached under DATA_DIR
  -> timestamped chunks created
  -> embeddings upserted to Pinecone
  -> expected vector count verified as visible
  -> job, video, source-video, and source states updated
~~~

The worker takes a 15-minute lease, retries transient errors at most three times, uses exponential retry delays from 30 seconds capped at five minutes, and recovers expired leases. Provider and embedding work occurs outside database transactions, then results are saved in short transactions.

Videos are globally deduplicated by YouTube video ID. If another source references an already-ready video, the existing video is linked without a duplicate ingestion job.

### Chat

A thread belongs to one user and one source. It can be created only when the source is <code>ready</code> or <code>partial</code> and has at least one ready video.

~~~text
user message
  -> ownership and ready-video scope check
  -> save user message
  -> LangGraph outside a database transaction
       rewrite a follow-up query
       -> retrieve scoped Pinecone evidence
       -> no evidence: deterministic refusal
       -> evidence: Groq answer generation and citation verification
  -> atomically save answer, rewritten query, and citations
~~~

The default whole-turn RAG timeout is 45 seconds. Provider, timeout, or graph failures return HTTP 503 and are recorded as a retryable temporary-error assistant message whenever persistence is available.

## Status model

| Entity | Statuses | Meaning |
|---|---|---|
| Source | <code>pending</code>, <code>processing</code>, <code>ready</code>, <code>partial</code>, <code>failed</code> | Aggregate state of a submitted video or playlist. Partial means some content is usable and some has a terminal non-ready outcome. |
| Video and source-video | <code>pending</code>, <code>processing</code>, <code>ready</code>, <code>no_transcript</code>, <code>failed</code> | Global video state and source-specific membership state. |
| Ingestion job | <code>queued</code>, <code>running</code>, <code>retrying</code>, <code>succeeded</code>, <code>failed</code> | Durable worker state, including attempts, availability time, and lease metadata. |
| Assistant message | <code>answered</code>, <code>refused_no_context</code>, <code>temporary_error</code>, and other persisted outcomes | Distinguishes an answer, a no-evidence refusal, and a retryable failure. |

## Technology stack

| Concern | Implementation |
|---|---|
| API and UI hosting | FastAPI, Uvicorn, static browser assets |
| Authentication | pwdlib Argon2, PyJWT, OAuth2 bearer tokens |
| Relational persistence | PostgreSQL, SQLAlchemy 2, Alembic, psycopg |
| Source processing | Supadata and local transcript JSON |
| Embeddings | HuggingFace sentence-transformers |
| Vector search | Pinecone |
| LLM orchestration | Groq through LangChain and LangGraph |
| Testing | pytest and FastAPI TestClient |
| Evaluation | Golden data, retrieval replay, DeepEval/Groq judging |
| Dependency management | Python 3.11 and uv |

## Repository layout

~~~text
rag_eval/
├── src/
│   ├── api/                 Routes, dependencies, and cursor handling
│   ├── cli/                 Source, status, and worker commands
│   ├── core/                Settings, logging, and security
│   ├── db/                  Sessions, models, and repositories
│   ├── graph/               LangGraph nodes, prompts, and graph builder
│   ├── llm/                 Groq and embedding client factories
│   ├── rag/                 Transcripts, chunks, vectors, retrieval, citations
│   ├── services/            Auth, ingestion, threads, and chat orchestration
│   ├── web/                 Same-origin HTML, CSS, and JavaScript UI
│   ├── youtube/             YouTube URL parsing
│   └── main.py              FastAPI application
├── migrations/              Alembic environment and revisions
├── tests/                   Unit, API, integration, worker, and UI tests
├── evals/                   Datasets, scorers, and evaluation documentation
├── docs/                    Focused design and operating guides
├── Dockerfile               Optional production-oriented API image
├── pyproject.toml           Project metadata
└── uv.lock                  Reproducible dependency lockfile
~~~

## Prerequisites

- Python 3.11 or newer.
- uv for lock-backed dependency installation.
- PostgreSQL.
- Groq, Supadata, and Pinecone credentials for live application use.
- Hugging Face model access if the selected embedding model requires it.

Run every command from the repository root so <code>src.*</code> imports resolve correctly.

## Local setup

### 1. Install locked dependencies

~~~powershell
uv sync --locked
~~~

Use the checked-in <code>uv.lock</code> as the reproducible dependency source. The legacy <code>requirements.txt</code> is not the primary install path.

### 2. Create a root .env file

The project loads a repository-root <code>.env</code> file, which is ignored by Git. There is no checked-in <code>.env.example</code>; create your own with placeholders only:

~~~dotenv
# Required to import and run the database-backed API
DATABASE_URL=postgresql+psycopg://postgres:change-me@127.0.0.1:5432/rag_eval

# Required for token creation and validation
JWT_SECRET_KEY=replace-with-a-long-random-development-secret

# Required for live chat
GROQ_API_KEY=replace-me

# Required for live transcript ingestion and retrieval
SUPADATA_API_KEY=replace-me
PINECONE_API_KEY=replace-me

# Defaults shown for clarity
PINECONE_INDEX_NAME=youtube-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
LLM_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
~~~

Never commit a real environment file, token, connection string, or API key.

### 3. Apply database migrations

~~~powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
~~~

Alembic reads <code>DATABASE_URL</code> through the application settings. The application and container do not run migrations automatically.

### 4. Start the API

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
~~~

Open:

- <a href="http://127.0.0.1:8000/">http://127.0.0.1:8000/</a> for the local browser UI.
- <a href="http://127.0.0.1:8000/docs">http://127.0.0.1:8000/docs</a> for interactive API documentation.
- <a href="http://127.0.0.1:8000/health">http://127.0.0.1:8000/health</a> for the health response.

If uv cannot access its cache after installation, use the project virtual-environment Python executable shown above.

### 5. Start the worker in another terminal

~~~powershell
# Recover stale jobs and process at most one runnable job
.\.venv\Scripts\python.exe -m src.cli.ingestion_worker --once

# Keep polling until stopped; default idle delay is two seconds
.\.venv\Scripts\python.exe -m src.cli.ingestion_worker --loop

# Change the idle polling interval
.\.venv\Scripts\python.exe -m src.cli.ingestion_worker --loop --poll-interval 5
~~~

The API queues work but does not process it. Keep a worker running while using the UI.

## Using the browser UI

1. Open the root URL and register with an email and a password of at least 12 characters.
2. Sign in. The UI keeps the bearer token only for the current browser session.
3. Submit a YouTube video or playlist URL.
4. Wait until at least one linked video is ready.
5. Create a chat from that source and ask a question.
6. Open returned citations to view supporting moments on YouTube.

The UI polls sources that are still pending or processing. Deleting a chat permanently removes its thread, messages, and citations, but preserves the submitted source and indexed vectors.

## HTTP API

All source and thread routes require:

~~~text
Authorization: Bearer <access-token>
~~~

| Method and path | Auth | Request / behavior |
|---|---|---|
| <code>GET /health</code> | No | Returns <code>{"status":"ok"}</code>. |
| <code>POST /auth/register</code> | No | JSON body with email and password. |
| <code>POST /auth/token</code> | No | OAuth2 form body with email in <code>username</code> and a password. |
| <code>POST /sources</code> | Yes | JSON: <code>{"youtube_url":"..."}</code>; creates a source and queues work where needed. |
| <code>GET /sources</code> | Yes | Lists caller-owned sources newest first; supports <code>cursor</code> and <code>limit</code> from 1 through 100. |
| <code>GET /sources/{source_id}</code> | Yes | Returns aggregate video/job progress. |
| <code>POST /threads</code> | Yes | JSON with source ID and non-blank title; requires a chat-ready source. |
| <code>GET /threads</code> | Yes | Lists caller-owned threads by latest activity with cursor pagination. |
| <code>DELETE /threads/{thread_id}</code> | Yes | Permanently removes an owned chat and its history. |
| <code>POST /threads/{thread_id}/messages</code> | Yes | JSON with non-blank question content; returns the persisted turn and citations. |
| <code>GET /threads/{thread_id}/messages</code> | Yes | Returns a chronological display page of message history and citations. |

Important response behavior:

- Invalid request data or cursors return HTTP 422.
- Missing and non-owned sources/threads return HTTP 404.
- A source with no ready video returns HTTP 409 for thread creation or chat.
- Provider and timeout errors return a generic HTTP 503 response.
- The current API returns completed JSON; there is no HTTP streaming or SSE chat endpoint.

## Operational CLI commands

~~~powershell
# Inspect source progress
.\.venv\Scripts\python.exe -m src.cli.source_status <source-uuid>

# Submit a source for a known user
.\.venv\Scripts\python.exe -m src.cli.submit_source <user-uuid> "<youtube-url>"

# Submit and immediately run that source's jobs
.\.venv\Scripts\python.exe -m src.cli.submit_source <user-uuid> "<youtube-url>" --run
~~~

There is also a lower-level bulk ingestion module:

~~~powershell
.\.venv\Scripts\python.exe -m src.rag.ingest "<youtube-url-or-id>"
~~~

It makes real Supadata, embedding, and Pinecone calls, so it is an operational tool rather than a normal setup command.

## Configuration reference

Settings are defined in <code>src/core/config.py</code>. Process environment variables override matching values in <code>.env</code>.

### Credentials and runtime essentials

| Variable | Default | Purpose |
|---|---:|---|
| <code>DATABASE_URL</code> | none | PostgreSQL SQLAlchemy URL. It is required even to import the database-backed FastAPI app. |
| <code>JWT_SECRET_KEY</code> | none | Required to sign and validate access tokens. |
| <code>ACCESS_TOKEN_EXPIRE_MINUTES</code> | 30 | Token expiration period. |
| <code>GROQ_API_KEY</code> or <code>GROQ_API_KEYS</code> | empty | Groq credential for chat and generation judging. |
| <code>SUPADATA_API_KEY</code> | empty | Credential for live transcript and playlist retrieval. |
| <code>PINECONE_API_KEY</code> | empty | Credential for vector index operations. |
| <code>PINECONE_INDEX_NAME</code> | <code>youtube-rag</code> | Pinecone index name. |
| <code>PINECONE_CLOUD</code> | <code>aws</code> | Pinecone serverless cloud. |
| <code>PINECONE_REGION</code> | <code>us-east-1</code> | Pinecone serverless region. |

### Models, retrieval, and data

| Variable | Default | Purpose |
|---|---:|---|
| <code>LLM_MODEL</code> | <code>openai/gpt-oss-120b</code> | Groq chat model. |
| <code>LLM_TEMPERATURE</code> | 0.0 | Chat generation temperature. |
| <code>GRADER_MODEL</code> | main model | Optional separate generation-evaluation judge. |
| <code>GRADER_TEMPERATURE</code> | 0.0 | Judge temperature. |
| <code>EMBEDDING_MODEL</code> | <code>sentence-transformers/all-mpnet-base-v2</code> | HuggingFace embedding model. |
| <code>HF_TOKEN</code> | empty | Optional Hugging Face credential. |
| <code>HF_HUB_OFFLINE</code> | false | Prevents Hugging Face downloads when true. |
| <code>RETRIVE_K</code> | 5 | Final retrieval chunk count. This is the current implementation spelling. |
| <code>SCORE_THRESHOLD</code> | unset | Rejects provider-scored candidates below the threshold. Tune with replay before enabling. |
| <code>PER_VIDEO_CAP</code> | unset | Caps chunks from one video in a multi-video source. |
| <code>CHUNK_SIZE</code> | 800 | Target transcript chunk size. |
| <code>CHUNK_OVERLAP_SEGMENTS</code> | 2 | Segment overlap between chunks. |
| <code>DOWNLOAD_LIMIT</code> | 200 | Limit for bulk video resolution/download work. |
| <code>DATA_DIR</code> | <code>data</code> | Local transcript-cache base directory, resolved from the repository root when relative. |

### Reliability and observability

| Variable | Default | Purpose |
|---|---:|---|
| <code>CHAT_RAG_TIMEOUT_SECONDS</code> | 45 | Whole graph deadline for one chat request; valid values are greater than 0 through 300 seconds. |
| <code>INDEX_VERIFY_ATTEMPTS</code> | 10 | Number of Pinecone visibility checks after upsert. |
| <code>INDEX_VERIFY_DELAY</code> | 2 seconds | Delay between vector visibility checks. |
| <code>LOG_LEVEL</code> | <code>INFO</code> | Application log level. |
| <code>STREAMING</code> | false | Passed to the Groq client, but the API still returns completed JSON responses. |

## Grounding and citations

### Source-scoped retrieval

The Pinecone index is global, but each chat passes a metadata filter containing only ready videos linked to that thread's source. Clients do not supply an arbitrary vector scope.

### Structural no-answer gate

~~~text
START -> rewrite -> retrieve -> generate -> cite -> END
                              |
                              +-> refuse -> END
~~~

The no-context route never reaches the generator. That enforces a lack-of-evidence refusal structurally instead of relying only on a model instruction.

### Citation verification

The model emits context IDs and verbatim supporting quotes, not raw timestamps. The application resolves IDs into YouTube time ranges, checks quote-token continuity against the cited chunk, and stores a verification score. Adjacent spans may be merged for display.

An unverified quote marks the answer as not fully grounded. The current behavior annotates that outcome instead of suppressing an otherwise useful answer.

Changing the embedding model, chunk size, or overlap makes existing vectors and evaluation baselines incomparable. Re-ingest and re-evaluate deliberately after those changes.

## Persistence model

~~~text
users
 ├─ sources
 │   ├─ source_videos ── videos
 │   └─ ingestion_jobs ─ videos
 └─ threads
     └─ messages
         └─ citations ─── videos
~~~

| Table | Responsibility |
|---|---|
| <code>users</code> | Email, password hash, active flag, timestamps. |
| <code>sources</code> | A user's submitted video or playlist URL and aggregate state. |
| <code>videos</code> | Globally deduplicated YouTube identity, transcript/index state, and counts. |
| <code>source_videos</code> | Source-specific video membership, ordering, state, and errors. |
| <code>ingestion_jobs</code> | Durable source/video work and retry/lease metadata. |
| <code>threads</code> | User-owned chats scoped to one source. |
| <code>messages</code> | User/assistant turns, rewritten retrieval query, grounding, and outcome. |
| <code>citations</code> | Ordered assistant-message evidence linked to videos and timestamps. |

PostgreSQL is required: the schema uses PostgreSQL UUIDs and native enums. The migration chain creates the core schema, adds worker scheduling and lease fields/indexes, then removes an unused archived-thread column.

## Testing

Run the normal suite:

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
~~~

The test suite covers URL parsing, authentication, source/thread APIs, web UI behavior, cursor pagination, retrieval gating, citations, worker behavior, and CLIs. A collection check currently discovers 128 tests.

### Opt-in database integration tests

Integration tests are deliberately fail-closed. They require:

- <code>RUN_DB_INTEGRATION=1</code>.
- PostgreSQL on a literal loopback IP such as <code>127.0.0.1</code>.
- Database name exactly <code>rag_eval_test</code>.
- A <code>DATABASE_URL</code> without query parameters.
- Migrations already applied to that local test database.

~~~powershell
$env:DATABASE_URL = "postgresql+psycopg://postgres:change-me@127.0.0.1:5432/rag_eval_test"
$env:RUN_DB_INTEGRATION = "1"
.\.venv\Scripts\python.exe -m pytest -q
~~~

Integration fixtures use an outer transaction and savepoints, then roll back test rows at the end.

## Evaluation workflow

The detailed guide is in <a href="evals/README.md">evals/README.md</a>.

| Activity | Provider use | Purpose |
|---|---|---|
| Golden validation | Offline | Checks golden dataset and transcript-span assumptions. |
| Candidate capture | Live Pinecone | Captures provider-ranked retrieval candidates for calibration. |
| Retrieval replay | Offline | Compares thresholds and context choices against captured candidates. |
| Generation evaluation | Live Groq | Judges answers against oracle context; this can incur model cost. |

~~~powershell
# Validate the golden set
.\.venv\Scripts\python.exe -m evals.scorers.validate_golden

# Capture candidates from Pinecone after explicitly acknowledging provider use
.\.venv\Scripts\python.exe -m evals.scorers.run_retrieval_eval --capture-candidates evals/results/retrieval-candidates.json --allow-live-retrieval

# Replay candidates locally while comparing thresholds
.\.venv\Scripts\python.exe -m evals.scorers.run_retrieval_eval --replay-candidates evals/results/retrieval-candidates.json --thresholds 0.10 0.15 0.20 --report evals/results/threshold-sweep.json

# Probe the generation judge before a full run
.\.venv\Scripts\python.exe -m evals.scorers.run_generation_eval --judge-check
~~~

The <code>data</code> cache and <code>evals/results</code> are ignored runtime artifacts, not committed source data.

## Docker

The repository includes a production-oriented <code>Dockerfile</code> for the API:

~~~powershell
docker build -t rag-eval .
docker run --rm --env-file .env -p 8080:8080 rag-eval
~~~

It uses Python 3.11 slim, locked uv dependencies, a non-root user, and port 8080 by default. It can use a Cloud Run-style injected <code>PORT</code>.

The image does not run migrations or start an ingestion worker. Treat migrations, worker deployment, PostgreSQL, and external providers as explicit operational dependencies. There is no Docker Compose configuration in this repository.

## Documentation map

- <a href="docs/local-browser-app.md">Local browser application guide</a> — UI behavior and manual smoke testing.
- <a href="docs/live-rag-smoke-test.md">Live RAG smoke test</a> — manual verification of a real source/chat flow.
- <a href="docs/no-answer-gate.md">No-answer gate</a> — threshold calibration and refusal trade-offs.
- <a href="docs/rag-contract.md">RAG contract</a> — grounding and response-contract design notes.
- <a href="evals/README.md">Evaluation guide</a> — dataset, retrieval replay, and generation judging.

## Current limitations and production considerations

- The local UI keeps its bearer token in <code>sessionStorage</code>; revisit that design before a hostile-browser production deployment.
- Chat is synchronous HTTP today; no streaming endpoint is exposed.
- Embeddings and provider calls add latency, cost, rate limits, and failure modes.
- A threshold that is too high causes safe refusals; one that is too low lets weak evidence reach the generator. Calibrate from captured candidates rather than intuition.
- The SQLAlchemy pool is small and synchronous. Measure real usage before changing API process count, worker concurrency, or database capacity.
- No CI workflow, Docker Compose configuration, or license file is currently checked in.
- The declared <code>rag-eval</code> console entry point is not wired to an implementation; use the module commands in this README.
