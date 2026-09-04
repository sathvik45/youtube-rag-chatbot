"""Thread-scoped retrieval over the shared corpus.

A thread is a filter, not a store. Single-video, playlist, and whole-library
threads use the same query path with a different video_id list.

No LangGraph or FastAPI imports here -- nodes and eval runners both call these
functions directly.
"""

from langchain_core.documents import Document

from src.rag.vector_store import get_vectorstore
from src.core.logging import get_logger
from src.core.config import settings
from src.rag.evidence import (
    ScoredCandidate,
    candidate_fetch_limit,
    select_evidence,
)

log = get_logger(__name__)


def _scope_filter(video_ids: list[str]) -> dict:
    if not video_ids:
        raise ValueError("Cannot retrieve with an empty video scope")
    if len(video_ids) == 1:
        return {"video_id": video_ids[0]}
    return {"video_id": {"$in": video_ids}}


def get_retriever(video_ids: list[str], k: int = settings.retrive_K):
    """LangChain retriever bound to a thread's scope.

    Build per request, not at startup -- the filter changes with the thread.
    """
    return get_vectorstore().as_retriever(
        search_kwargs={"k": k, "filter": _scope_filter(video_ids)}
    )


def retrieve(
    query: str,
    video_ids: list[str],
    k: int = settings.retrive_K,
    per_video_cap: int | None = None,
    score_threshold: float | None = None,
) -> list[Document]:
    """Scoped similarity search.

    per_video_cap stops one long video from monopolising context in a playlist
    thread. Ignored for single-video scopes, where it has no meaning.

    score_threshold drops weak matches. Its value is tied to the current
    embedding model and Pinecone index, so calibrate it on the golden set
    rather than guessing a universal cosine cutoff.
    """
    single = len(video_ids) == 1
    candidates = retrieve_scored(
        query,
        video_ids,
        k=k,
        per_video_cap=per_video_cap,
    )
    selection = select_evidence(
        candidates,
        k=k,
        per_video_cap=per_video_cap,
        single_video=single,
        score_threshold=score_threshold,
    )

    top_score = candidates[0].score if candidates else None
    log.info(
        "retrieval_evidence_gate scope_video_count=%s candidate_count=%s "
        "qualifying_count=%s selected_count=%s top_score=%s "
        "score_threshold=%s",
        len(video_ids),
        len(candidates),
        len(selection.qualifying),
        len(selection.selected),
        top_score,
        score_threshold,
    )

    documents: list[Document] = []
    for candidate in selection.selected:
        candidate.value.metadata["score"] = candidate.score
        documents.append(candidate.value)
    return documents


def retrieve_scored(
    query: str,
    video_ids: list[str],
    *,
    k: int,
    per_video_cap: int | None = None,
) -> list[ScoredCandidate[Document]]:
    """Fetch the ranked, pre-threshold candidate pool for one scoped query.

    The calibration capture calls this once per golden query, then replays
    threshold choices locally. Production uses the same pool before applying
    ``select_evidence``.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    scope = _scope_filter(video_ids)
    fetch_k = candidate_fetch_limit(
        k=k,
        scope_size=len(video_ids),
        per_video_cap=per_video_cap,
    )
    scored = get_vectorstore().similarity_search_with_score(
        query,
        k=fetch_k,
        filter=scope,
    )

    candidates: list[ScoredCandidate[Document]] = []
    for document, raw_score in scored:
        video_id = document.metadata.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("retrieval returned a chunk without a video_id.")
        if not document.page_content.strip():
            # Treat a corrupt vector record as unusable evidence. Returning it
            # would let the graph call the generator with an empty context.
            log.warning("skipping_empty_retrieved_chunk video_id=%s", video_id)
            continue

        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise ValueError("retrieval returned a non-numeric score.") from error

        candidates.append(
            ScoredCandidate(
                value=document,
                score=score,
                video_id=video_id,
            )
        )

    return candidates


def retrieve_by_time(
    video_id: str, start_ms: int, end_ms: int, limit: int = 10
) -> list[Document]:
    """Fetch chunks overlapping a time range -- for 'what was said around 4:20'
    and for expanding context around a citation."""
    return get_vectorstore().similarity_search(
        query="",
        k=limit,
        filter={
            "video_id": video_id,
            "start_ms": {"$lte": end_ms},
            "end_ms": {"$gte": start_ms},
        },
    )
