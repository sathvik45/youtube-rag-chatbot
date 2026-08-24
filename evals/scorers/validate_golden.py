"""Validate the retrieval golden set against the transcripts on disk.

This checks the ANSWER KEY, not a retriever. Retrieval_metrics.py grades a
retriever's output against this dataset; nothing there can tell you the dataset
itself is wrong, because a span pointing at the wrong 30 seconds still scores
cleanly. That is what this catches.

Checks, per row:
  * schema  - required keys, id uniqueness, known query_type
  * scope   - every video_id exists in data/transcripts, and every gold span's
              video_id is inside that row's scope
  * spans   - start < end, and both edges land on real segment boundaries
              (within TOL ms) so a span never slices a segment in half
  * absent  - query_type "absent" has no gold spans, and nothing else is empty

Run with --show to print the transcript text each span covers, which is the
only way to confirm a span actually answers its query.

    python -m evals.scorers.validate_golden
    python -m evals.scorers.validate_golden --show q122 q123
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.core.config import PROJECT_ROOT

# Root-anchored, not cwd-relative: the validator is run from editors, from
# CI, and from the repo root, and a relative path silently finds nothing
# from two of those three.
TRANSCRIPTS = PROJECT_ROOT / "data" / "transcripts"
GOLDEN = PROJECT_ROOT / "evals" / "datasets" / "golden_dataset_SD.json"
TOL = 2000  # ms of slack when matching a span edge to a segment edge
QUERY_TYPES = {"pointwise", "terse", "comparison", "aggregation", "absent"}


def load_transcripts() -> dict:
    out = {}
    for p in TRANSCRIPTS.glob("*.json"):
        if p.name == "manifest.json":
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        segs = r["segments"]
        out[r["video_id"]] = {
            "title": r.get("title"),
            "segments": segs,
            "starts": {s["start_ms"] for s in segs},
            "ends": {s["start_ms"] + s["duration_ms"] for s in segs},
        }
    return out


def near(value: int, edges: set) -> bool:
    return any(abs(value - e) <= TOL for e in edges)


def span_text(vid: dict, start: int, end: int) -> str:
    parts = [
        s["text"]
        for s in vid["segments"]
        if s["start_ms"] < end and s["start_ms"] + s["duration_ms"] > start
    ]
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", nargs="*", default=None,
                    help="print covered text for these ids (or all if bare)")
    args = ap.parse_args()

    vids = load_transcripts()
    rows = json.loads(GOLDEN.read_text(encoding="utf-8"))

    errors: list[str] = []
    seen_ids: set = set()
    type_counts: dict = {}
    covered_videos: set = set()

    for row in rows:
        rid = row.get("id", "<missing id>")

        for key in ("id", "query", "scope", "gold_spans", "query_type"):
            if key not in row:
                errors.append(f"{rid}: missing key {key!r}")
        if rid in seen_ids:
            errors.append(f"{rid}: duplicate id")
        seen_ids.add(rid)

        qt = row.get("query_type")
        type_counts[qt] = type_counts.get(qt, 0) + 1
        if qt not in QUERY_TYPES:
            errors.append(f"{rid}: unknown query_type {qt!r}")

        scope = row.get("scope", [])
        for v in scope:
            if v not in vids:
                errors.append(f"{rid}: scope video {v!r} not in transcripts")
        covered_videos.update(scope)

        spans = row.get("gold_spans", [])
        if qt == "absent" and spans:
            errors.append(f"{rid}: absent query must have no gold_spans")
        if qt != "absent" and not spans:
            errors.append(f"{rid}: non-absent query has no gold_spans")

        for i, sp in enumerate(spans):
            v, st, en = sp["video_id"], sp["start_ms"], sp["end_ms"]
            tag = f"{rid} span{i} ({v} {st}-{en})"
            if v not in vids:
                errors.append(f"{tag}: unknown video")
                continue
            if v not in scope:
                errors.append(f"{tag}: video not listed in scope")
            if st >= en:
                errors.append(f"{tag}: start >= end")
                continue
            if not near(st, vids[v]["starts"]):
                errors.append(f"{tag}: start does not land on a segment boundary")
            if not near(en, vids[v]["ends"]):
                errors.append(f"{tag}: end does not land on a segment boundary")
            if not span_text(vids[v], st, en).strip():
                errors.append(f"{tag}: covers no text")

    print(f"{len(rows)} rows, {len(covered_videos)} of {len(vids)} videos in scope")
    print("query_type:", ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())))
    missing = sorted(set(vids) - covered_videos)
    if missing:
        print(f"videos with no query ({len(missing)}): {', '.join(missing)}")

    if errors:
        print(f"\n{len(errors)} PROBLEM(S):")
        for e in errors:
            print("  -", e)
    else:
        print("\nall spans align to segment boundaries")

    if args.show is not None:
        wanted = set(args.show) if args.show else None
        for row in rows:
            if wanted and row["id"] not in wanted:
                continue
            print(f"\n=== {row['id']} [{row['query_type']}] {row['query']}")
            if not row["gold_spans"]:
                print("    (absent - no gold spans)")
            for sp in row["gold_spans"]:
                v = vids.get(sp["video_id"])
                if not v:
                    continue
                txt = span_text(v, sp["start_ms"], sp["end_ms"])
                print(f"    [{sp['video_id']} {sp['start_ms']}-{sp['end_ms']}] {txt}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
