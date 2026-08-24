"""Oracle context: the chunks the retriever *should* have returned.

Evaluating the generator only means something if the context is held fixed. Run
the real retriever and a low faithfulness score is unreadable -- bad retrieval
and bad generation produce the same number. So context here comes from the
golden set's `gold_spans`, never from Pinecone.

Chunks, not raw transcript text, on purpose. `splitters.chunk_transcript` is
pure -- string joins and integer arithmetic, no model, no network -- so we can
rebuild exactly the chunks ingestion produced and keep the ones overlapping each
gold span. The generator then sees the shape of input it sees in production
(overlapping windows, ASR noise, boundaries mid-sentence) with only the
retrieval error removed.

Consequences worth knowing before you read the numbers:

  * No Pinecone and no embedding model. Transcripts on disk are enough, so this
    eval runs offline apart from the LLM calls.
  * Ordering is chronological, not by relevance rank. Production hands the
    generator a ranked list; oracle context has no ranking to give. That makes
    this blind to position effects -- a separate condition, not this one.
  * `absent` rows have no gold spans and therefore no oracle context. They are
    excluded. Measuring refusal needs deliberately-wrong context, which is a
    different case set entirely.
"""

from __future__ import annotations

import random
from functools import lru_cache

from langchain_core.documents import Document

from src.core.logging import get_logger
from src.rag.splitters import get_chunks_for_video

log = get_logger(__name__)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


@lru_cache(maxsize=None)
def _chunks(video_id: str) -> tuple[Document, ...]:
    """Chunk one video once per process.

    Rows share videos heavily -- 305 queries over 102 transcripts -- and
    re-chunking the same file for every row is the difference between a
    two-second setup and a two-minute one.
    """
    docs = get_chunks_for_video(video_id)
    if not docs:
        log.warning(f"no transcript on disk for {video_id}")
    return tuple(docs)


def _as_state_chunk(doc: Document) -> dict:
    """The shape Generate_node expects in state['chunks'].

    retrieve_node builds exactly this from a retrieved Document, so building it
    the same way here is what lets the eval call the production node unchanged.
    """
    return {**doc.metadata, "text": doc.page_content}


def oracle_chunks(
    item: dict, min_overlap_ms: int = 1000, max_blocks: int = 12
) -> list[dict]:
    """Chunks covering this row's gold spans, chronologically ordered.

    `min_overlap_ms` drops chunks that merely graze a span edge. With
    overlap_segments=2 every span touches its neighbours by a second or two, and
    those neighbours carry no answer content -- they just inflate the context and
    give the model more surface to be unfaithful on.

    `max_blocks` matters more than it looks. q129 has seven gold spans and q291
    has six; unbounded, those rows get a 20-block context that no other row gets,
    and their faithfulness score then measures long-context behaviour rather than
    the thing every other row measures. The cap is filled ROUND-ROBIN across
    spans, not by global overlap ranking: rank globally and one long span can
    take every slot, silently deleting the other six spans from the context and
    turning an aggregation row into a pointwise one.
    """
    golds = item.get("gold_spans", [])
    if not golds:
        return []

    # Per span: candidate chunks, best overlap first.
    per_span: list[list[Document]] = []
    for gold in golds:
        scored = []
        for doc in _chunks(gold["video_id"]):
            ov = _overlap(
                int(doc.metadata["start_ms"]),
                int(doc.metadata["end_ms"]),
                gold["start_ms"],
                gold["end_ms"],
            )
            if ov >= min_overlap_ms:
                scored.append((ov, doc))
        scored.sort(key=lambda t: -t[0])
        per_span.append([doc for _, doc in scored])

    picked: list[Document] = []
    seen: set[tuple[str, int]] = set()
    cursors = [0] * len(per_span)

    while len(picked) < max_blocks:
        progressed = False
        for i, candidates in enumerate(per_span):
            while cursors[i] < len(candidates):
                doc = candidates[cursors[i]]
                cursors[i] += 1
                key = (doc.metadata["video_id"], doc.metadata["chunk_index"])
                if key in seen:
                    continue
                seen.add(key)
                picked.append(doc)
                progressed = True
                break
            if len(picked) >= max_blocks:
                break
        if not progressed:
            break

    picked.sort(key=lambda d: (d.metadata["video_id"], int(d.metadata["start_ms"])))
    return [_as_state_chunk(d) for d in picked]


def stratified_sample(
    dataset: list[dict], n: int, seed: int = 0, exclude_absent: bool = True
) -> list[dict]:
    """Pick n rows keeping the query_type mix of the full set.

    Seeded and sorted, so two runs a week apart score the SAME rows. An unseeded
    sample makes every run incomparable to the last, which quietly destroys the
    only thing an eval is for.

    Proportional allocation with largest-remainder, so `comparison` (42 of 305)
    does not vanish at n=40 the way a naive round() would drop it.
    """
    pool = [
        d for d in dataset
        if not (exclude_absent and d["query_type"] == "absent")
    ]
    if n >= len(pool):
        return sorted(pool, key=lambda d: d["id"])

    groups: dict[str, list[dict]] = {}
    for d in pool:
        groups.setdefault(d["query_type"], []).append(d)

    exact = {k: len(v) * n / len(pool) for k, v in groups.items()}
    quota = {k: int(v) for k, v in exact.items()}

    # Hand out the rounding remainder to the largest fractional parts.
    short = n - sum(quota.values())
    for k in sorted(exact, key=lambda k: -(exact[k] - quota[k]))[:short]:
        quota[k] += 1

    rng = random.Random(seed)
    out: list[dict] = []
    for qtype, rows in groups.items():
        rows = sorted(rows, key=lambda d: d["id"])
        out.extend(rng.sample(rows, min(quota[qtype], len(rows))))

    return sorted(out, key=lambda d: d["id"])


def build_cases(
    dataset: list[dict], min_overlap_ms: int = 1000, max_blocks: int = 12
) -> tuple[list[dict], list[str]]:
    """Attach oracle context to each row. Returns (usable, skipped_ids).

    A row is skipped when its transcript is missing from disk or every candidate
    chunk fell under min_overlap_ms. Skipping loudly beats generating from an
    empty context: an empty context is the refusal path, and a refusal scored as
    a generation is a metric reading zero for the wrong reason.
    """
    usable, skipped = [], []
    for item in dataset:
        chunks = oracle_chunks(item, min_overlap_ms, max_blocks)
        if not chunks:
            skipped.append(item["id"])
            continue
        usable.append({**item, "chunks": chunks})
    return usable, skipped
