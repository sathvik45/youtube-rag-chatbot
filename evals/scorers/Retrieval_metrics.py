"""Span-based retrieval metrics.

Ground truth is a TIME SPAN in a video, never a chunk id. That makes the golden
dataset survive changes to chunk_size, overlap, and embedding model -- label
once, valid forever.

No LLM judge here. Span overlap is arithmetic, so these metrics are
deterministic, free, and instant.
"""

from statistics import mean


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _union(intervals: list[tuple[int, int]]) -> list[list[int]]:
    """Merge overlapping intervals.

    Required, not optional: chunks overlap each other by design
    (overlap_segments), so summing raw intersections double-counts and can
    report coverage above 1.0.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    out = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def span_coverage(gold: dict, retrieved: list[dict]) -> float:
    """Fraction of one gold span covered by retrieved chunks from that video."""
    duration = gold["end_ms"] - gold["start_ms"]
    if duration <= 0:
        return 0.0

    same_video = [
        (c["start_ms"], c["end_ms"])
        for c in retrieved
        if c["video_id"] == gold["video_id"]
    ]
    covered = sum(
        _overlap(s, e, gold["start_ms"], gold["end_ms"])
        for s, e in _union(same_video)
    )
    return min(1.0, covered / duration)


def score_query(item: dict, retrieved: list[dict], hit_threshold: float = 0.5) -> dict:
    """Score one query.

    hit      -- found at least one gold span. The pointwise metric.
    coverage -- found what FRACTION of gold spans. The aggregation metric,
                and the one that catches "answered 2 of 3 strategies".
    video_recall -- playlist scope only: did we reach the right videos at all?
                A miss here is a filter/embedding problem; a coverage miss
                with video_recall=1.0 is a ranking problem.
    """
    golds = item.get("gold_spans", [])

    # Absent query: correct behaviour is returning nothing.
    if not golds:
        empty = len(retrieved) == 0
        return {
            "id": item["id"],
            "query_type": item["query_type"],
            "hit": empty,
            "coverage": 1.0 if empty else 0.0,
            "mean_span_coverage": 1.0 if empty else 0.0,
            "video_recall": 1.0,
            "n_gold": 0,
        }

    covs = [span_coverage(g, retrieved) for g in golds]
    hits = [c >= hit_threshold for c in covs]

    gold_videos = {g["video_id"] for g in golds}
    got_videos = {c["video_id"] for c in retrieved}

    return {
        "id": item["id"],
        "query_type": item["query_type"],
        "hit": any(hits),
        "coverage": sum(hits) / len(hits),
        "mean_span_coverage": mean(covs),
        "video_recall": len(gold_videos & got_videos) / len(gold_videos),
        "n_gold": len(golds),
    }


def mrr(item: dict, retrieved: list[dict], min_overlap: float = 0.1) -> float:
    """Reciprocal rank of the first chunk touching any gold span.

    This is the metric a reranker moves. Uses a low overlap bar because it
    measures ranking, not completeness.
    """
    for rank, chunk in enumerate(retrieved, start=1):
        for gold in item.get("gold_spans", []):
            if chunk["video_id"] != gold["video_id"]:
                continue
            duration = gold["end_ms"] - gold["start_ms"]
            if duration <= 0:
                continue
            ov = _overlap(
                chunk["start_ms"], chunk["end_ms"], gold["start_ms"], gold["end_ms"]
            )
            if ov / duration >= min_overlap:
                return 1.0 / rank
    return 0.0


def junk_rate(retrieved: list[dict], boilerplate_ids: set[str]) -> float:
    """Fraction of retrieved chunks that are known boilerplate.

    Playlist-specific. Channel intros and outros repeat across every video, so
    at 50 videos you have ~50 near-identical vectors competing for slots.
    """
    if not retrieved:
        return 0.0
    keys = [f"{c['video_id']}#{c['chunk_index']}" for c in retrieved]
    return sum(k in boilerplate_ids for k in keys) / len(keys)


def evaluate(dataset: list[dict], retrieve_fn, ks=(1, 3, 5, 10)) -> list[dict]:
    """Run the whole set at several k.

    retrieve_fn(query, video_ids, k) -> list of chunk metadata dicts.
    Retrieves once at max(k) and slices, so this costs one search per query.
    """
    rows = []
    for item in dataset:
        got = retrieve_fn(item["query"], item["scope"], max(ks))
        for k in ks:
            row = score_query(item, got[:k])
            row["k"] = k
            row["mrr"] = mrr(item, got[:k])
            rows.append(row)
    return rows


def summarize(rows: list[dict], ks=(1, 3, 5, 10)) -> str:
    """Recall CURVE, not a single number.

    Read the shape:
      hit@10 high, hit@3 low  -> right chunks found, ranked badly. Add a reranker.
      hit@10 low              -> right chunks never found. Reranker cannot help;
                                 fix chunking, embeddings, or add hybrid search.
    """
    lines = [f"{'k':>3} {'hit':>6} {'cover':>6} {'vidRec':>7} {'MRR':>6}"]
    for k in ks:
        at_k = [r for r in rows if r["k"] == k]
        if not at_k:
            continue
        lines.append(
            f"{k:>3} {mean(r['hit'] for r in at_k):6.2f} "
            f"{mean(r['coverage'] for r in at_k):6.2f} "
            f"{mean(r['video_recall'] for r in at_k):7.2f} "
            f"{mean(r['mrr'] for r in at_k):6.2f}"
        )

    lines.append("")
    lines.append("by query type (at k=5):")
    at5 = [r for r in rows if r["k"] == 5]
    for qtype in sorted({r["query_type"] for r in at5}):
        sub = [r for r in at5 if r["query_type"] == qtype]
        lines.append(
            f"  {qtype:14} n={len(sub):3}  hit={mean(r['hit'] for r in sub):.2f}  "
            f"cover={mean(r['coverage'] for r in sub):.2f}"
        )
    return "\n".join(lines)