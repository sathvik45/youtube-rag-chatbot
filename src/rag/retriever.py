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

    score_threshold drops weak matches. With cosine on normalised embeddings
    scores run 0-1; measure on your golden set before setting it, since a
    threshold that is too high silently returns nothing.
    """
    vs = get_vectorstore()
    scope = _scope_filter(video_ids)

    single = len(video_ids) == 1
    fetch_k = k if (per_video_cap is None or single) else min(k * 4, k + 20 * len(video_ids))

    scored = vs.similarity_search_with_score(query, k=fetch_k, filter=scope)

    if score_threshold is not None:
        before = len(scored)
        scored = [(d, s) for d, s in scored if s >= score_threshold]
        if not scored:
            log.warning(
                f"score_threshold={score_threshold} dropped all {before} results"
            )

    for doc, score in scored:
        doc.metadata["score"] = float(score)

    candidates = [doc for doc, _ in scored]

    if per_video_cap is None or single:
        return candidates[:k]

    return _cap_per_video(candidates, k, per_video_cap)


def _cap_per_video(
    candidates: list[Document], k: int, per_video_cap: int
) -> list[Document]:
    """Round-robin across videos so short videos are not crowded out by
    whichever one happens to have the most chunks."""
    by_video: dict[str, list[int]] = {}
    for idx, doc in enumerate(candidates):
        by_video.setdefault(doc.metadata["video_id"], []).append(idx)

    picked: list[int] = []
    for rank in range(per_video_cap):
        for indices in by_video.values():
            if rank < len(indices):
                picked.append(indices[rank])
        if len(picked) >= k:
            break

    # candidates is already in relevance order; restore it after round-robin
    picked.sort()
    return [candidates[i] for i in picked[:k]]


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