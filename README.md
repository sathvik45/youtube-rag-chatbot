rag_eval/
├── .env
├── .env.example
├── pyproject.toml
├── docker-compose.yml          # postgres for local dev
├── alembic.ini
├── migrations/                 # alembic versions
│
├── src/
│   ├── main.py                 # FastAPI app, router mounting, lifespan
│   │
│   ├── core/
│   │   ├── config.py           # your settings_config.py
│   │   ├── logging.py          # your logger_config.py
│   │   └── exceptions.py       # TranscriptUnavailable, IngestionFailed, etc.
│   │
│   ├── db/
│   │   ├── session.py          # engine, SessionLocal, get_db dependency
│   │   ├── models.py           # SQLAlchemy: User, Video, Thread, Message, Citation
│   │   └── repositories/
│   │       ├── threads.py      # thread + thread_videos queries
│   │       ├── messages.py
│   │       └── videos.py       # status transitions, dedup lookups
│   │
│   ├── schemas/                # Pydantic request/response models
│   │   ├── thread.py
│   │   ├── message.py
│   │   └── video.py
│   │
│   ├── rag/                    # zero FastAPI imports in here
│   │   ├── transcripts.py      # Supadata fetch, URL parsing, flat string + span table
│   │   ├── splitters.py        # chunk_transcript, _make_chunk
│   │   ├── vectorstore.py      # ensure_index, upsert_documents, delete_video
│   │   ├── retriever.py        # retrieve, build_context, resolve_citations
│   │   ├── prompts.py          # system prompts, citation instructions
│   │   └── chain.py            # retrieve → prompt → LLM → parse citations
│   │
│   ├── services/               # orchestration: the seam between API and rag/db
│   │   ├── ingestion.py        # fetch → chunk → embed → status updates
│   │   ├── chat.py             # routing (full-context vs retrieval), persistence
│   │   └── threads.py          # create thread, resolve scope, list for user
│   │
│   ├── api/
│   │   ├── deps.py             # get_current_user, get_db
│   │   └── routes/
│   │       ├── auth.py
│   │       ├── threads.py      # POST /threads, GET /threads, GET /threads/{id}
│   │       └── messages.py     # POST /threads/{id}/messages
│   │
│   └── workers/
│       └── ingest.py           # background job entry point
│
├── data/
│   └── transcripts/            # raw JSON cache, gitignored
│
└── tests/
    ├── conftest.py
    ├── test_splitters.py       # the bugs we found are all unit-testable
    ├── test_retriever.py
    └── fixtures/
        └── transcripts/        # your 5 JSONs, committed


        src/
        ├── graph/                      # replaces rag/chain.py
        │   ├── state.py                # ChatState TypedDict
        │   ├── nodes/
        │   │   ├── route.py            # full-context vs retrieval fork
        │   │   ├── retrieve.py         # wraps rag/retriever.py
        │   │   ├── load_full.py        # raw_transcript + time markers
        │   │   ├── generate.py
        │   │   └── verify.py           # citation resolution + quote anchoring
        │   ├── edges.py                # conditional routing functions
        │   ├── builder.py              # build_graph() → compiled graph
        │   └── checkpointer.py         # PostgresSaver setup
        │
        ├── rag/                        # unchanged — nodes call into it
        │   ├── transcripts.py
        │   ├── splitters.py
        │   ├── vectorstore.py
        │   ├── retriever.py
        │   └── prompts.py