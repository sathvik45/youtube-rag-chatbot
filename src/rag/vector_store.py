"""Pinecone index lifecycle and document upsert.

Shared corpus: vectors are keyed by video_id only, so a video ingested by one
user is instantly available to any thread referencing it. Ownership lives in
Postgres (user_videos), not here.

Vector IDs are deterministic -- f"{video_id}#{chunk_index}" -- which makes
re-ingestion an overwrite rather than a duplicate, and makes per-video deletion
possible on serverless indexes via prefix listing.
"""

import time
from functools import lru_cache

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.core.config import settings
from src.core.logging import get_logger
from src.llm.client import get_embeddings

log = get_logger(__name__)


# ---------------------------------------------------------------
# Clients (cached: model loading and index handshakes are slow)
# ---------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """Probe the model instead of hardcoding. A dimension mismatch against an
    existing index is otherwise a silent corruption."""
    return len(get_embeddings().embed_query("dimension probe"))


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache(maxsize=1)
def ensure_index():
    """Create the index if absent, verify dimension if present."""
    index_name = settings.pinecone_index_name
    pc = get_pinecone_client()
    dim = get_embedding_dimension()

    if index_name not in pc.list_indexes().names():
        log.info(f"Creating index '{index_name}' (dim={dim}, metric=cosine)")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        log.info(f"Index '{index_name}' ready")
    else:
        actual = pc.describe_index(index_name).dimension
        if actual != dim:
            raise ValueError(
                f"Index '{index_name}' has dimension {actual} but model "
                f"'{settings.embedding_model}' produces {dim}. "
                "Changing embedding model requires a new index."
            )

    return pc.Index(index_name)


@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(
        index=ensure_index(),
        embedding=get_embeddings(),
        text_key="text",
    )


# ---------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------

# Pinecone metadata accepts str, int, float, bool, list[str] only.
_ALLOWED = (str, int, float, bool)

_REQUIRED_METADATA = ("video_id", "chunk_index", "start_ms", "end_ms")


def _clean_metadata(meta: dict) -> dict:
    """Drop None values, coerce anything Pinecone won't accept."""
    out = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, _ALLOWED):
            out[key] = value
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _vector_id(doc: Document) -> str:
    return f"{doc.metadata['video_id']}#{doc.metadata['chunk_index']}"


def _validate(documents: list[Document]) -> None:
    """Fail loudly before embedding. A bad batch caught here costs nothing;
    caught after upsert it costs a delete-and-rebuild."""
    for i, doc in enumerate(documents):
        missing = [k for k in _REQUIRED_METADATA if k not in doc.metadata]
        if missing:
            raise ValueError(f"Document {i} missing metadata: {missing}")
        if not doc.page_content.strip():
            raise ValueError(f"Document {i} has empty page_content")
        for key in ("start_ms", "end_ms"):
            if not isinstance(doc.metadata[key], int):
                raise TypeError(
                    f"Document {i} metadata['{key}'] must be int, got "
                    f"{type(doc.metadata[key]).__name__}"
                )


def upsert_documents(documents: list[Document], batch_size: int = 100) -> int:
    """Embed and upsert chunks. Idempotent -- safe to re-run for a video."""
    if not documents:
        log.warning("upsert_documents called with no documents")
        return 0

    _validate(documents)

    vs = get_vectorstore()
    total = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]

        prepared = [
            Document(
                page_content=doc.page_content,
                metadata=_clean_metadata(doc.metadata),
            )
            for doc in batch
        ]
        ids = [_vector_id(doc) for doc in batch]

        vs.add_documents(documents=prepared, ids=ids)
        total += len(batch)
        log.info(f"Upserted {total}/{len(documents)}")

    return total


# ---------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------

def delete_video(video_id: str) -> int:
    """Remove every vector for a video.

    Serverless indexes do NOT support delete-by-metadata-filter -- the SDK
    accepts a `filter` kwarg but the server returns 400. Prefix listing over
    deterministic IDs is the supported path.
    """
    index = ensure_index()
    prefix = f"{video_id}#"

    deleted = 0
    for page in index.list(prefix=prefix):
        if not page:
            continue
        index.delete(ids=list(page))
        deleted += len(page)

    log.info(f"Deleted {deleted} vectors for video_id={video_id}")
    return deleted


# ---------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------

def index_stats() -> dict:
    stats = ensure_index().describe_index_stats()
    return {
        "total_vectors": stats.get("total_vector_count", 0),
        "dimension": stats.get("dimension"),
    }


def video_vector_count(video_id: str) -> int:
    """How many chunks are indexed for a video. Use this to confirm ingestion
    completed before flipping status to 'ready'."""
    index = ensure_index()
    return sum(len(page) for page in index.list(prefix=f"{video_id}#"))


if __name__ == "__main__":
    from src.rag.splitters import get_chunks

    ensure_index()
    log.info(f"Index stats before: {index_stats()}")

    chunks = get_chunks()
    count = upsert_documents(chunks)

    time.sleep(5)  # serverless upserts are eventually consistent
    log.info(f"Upserted {count} chunks")
    log.info(f"Index stats after: {index_stats()}")