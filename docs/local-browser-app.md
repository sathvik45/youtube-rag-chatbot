# Local browser app

This is the smallest complete local workflow for the YouTube RAG project. The
browser UI is served by FastAPI, so it uses the same origin as the API and does
not need a separate frontend server, Node installation, CORS configuration, or
browser access to provider secrets.

## What the app does

1. Register or sign in.
2. Submit a YouTube video or playlist URL.
3. Poll its ingestion progress while the worker runs separately.
4. Create a chat once at least one video is ready.
5. Ask questions and read saved answers with their citations.
6. Delete a chat when you no longer need its saved messages and citations.
   Deleting a chat keeps its uploaded source and indexed vector data.

Each chat remains scoped to the source used to create it. Adding another source
does not broaden an existing chat.

## Before starting

Use your local PostgreSQL database and make sure its migrations are current:

~~~powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
~~~

Your ignored local .env needs the database and JWT settings, plus the provider
settings needed for real ingestion and chat. Keep those secrets in the backend
.env; never put them in browser JavaScript.

## Start the application

Open two PowerShell terminals at the repository root.

In the first terminal, start the API and browser UI:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8000
~~~

In the second terminal, start the ingestion worker:

~~~powershell
.\.venv\Scripts\python.exe -m src.cli.ingestion_worker --loop
~~~

Then open <http://127.0.0.1:8000/>.

The worker is intentionally separate from the API. Downloading transcripts,
creating embeddings, and writing to the vector index are slow external jobs;
running them inside an HTTP request would make the web app unreliable.

## Expected behaviour

- A new source first shows pending or processing.
- Once it is ready, or partial with at least one ready video, its
  **Create chat** button becomes available.
- A chat request may take up to the configured RAG timeout while the model and
  vector search run. The UI disables duplicate sends while it waits.
- An unsupported question can return a normal saved refused_no_context
  response. That means the app found no usable evidence in that thread's
  videos; it is not a server failure.
- Deleting a chat permanently removes that chat, its messages, and its saved
  citations. It does not delete the uploaded source, videos, or vector data.

For the API-only reference and a deliberate live-provider smoke test, see
[live-rag-smoke-test.md](live-rag-smoke-test.md).

## Local-security note

For this learning app, the browser keeps the access token in sessionStorage.
It disappears when the tab session ends. A production frontend should use
secure HTTP-only cookies or a purpose-designed refresh-token flow instead.
