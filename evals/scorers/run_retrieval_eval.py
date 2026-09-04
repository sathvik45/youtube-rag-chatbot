"""Capture and replay retrieval candidates for no-answer calibration.

There are deliberately two separate commands:

* Capture is explicitly opt-in and is the only mode that contacts Pinecone.
* Replay is offline: it reads a saved candidate snapshot and compares score
  thresholds without embedding a query or opening a vector-store client.

Run from the repository root so both ``src.*`` and ``evals.*`` resolve:

    python -m evals.scorers.run_retrieval_eval `
      --capture-candidates evals/results/retrieval-candidates.json `
      --allow-live-retrieval

    python -m evals.scorers.run_retrieval_eval `
      --replay-candidates evals/results/retrieval-candidates.json `
      --thresholds 0.10 0.15 0.20 0.25 `
      --report evals/results/threshold-sweep.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from evals.scorers import Retrieval_metrics as metrics
from src.core.config import PROJECT_ROOT, settings
from src.rag.evidence import (
    ScoredCandidate,
    candidate_fetch_limit,
    select_evidence,
)


DATASET = PROJECT_ROOT / "evals" / "datasets" / "golden_dataset_SD.json"
CAPTURE_SCHEMA_VERSION = 1
SWEEP_SCHEMA_VERSION = 1


def _read_dataset() -> list[dict[str, Any]]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def _dataset_fingerprint(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    raw = DATASET.read_bytes()
    try:
        dataset_path = str(DATASET.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        # Useful for unit tests that point DATASET at a temporary fixture.
        dataset_path = str(DATASET)
    return {
        "path": dataset_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(dataset),
        "query_ids": [item["id"] for item in dataset],
    }


def _git_provenance() -> dict[str, str | bool | None]:
    """Best-effort local provenance; a missing git executable is harmless."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}

    return {"revision": revision, "dirty": dirty}


def _score_distribution(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    scores = sorted(
        candidate["score"]
        for candidates in records.values()
        for candidate in candidates
    )
    if not scores:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}

    def percentile(percent: float) -> float:
        index = round((len(scores) - 1) * percent)
        return scores[index]

    return {
        "count": len(scores),
        "min": scores[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": scores[-1],
    }


def _serialise_candidate(
    candidate: ScoredCandidate[Any],
    *,
    rank: int,
) -> dict[str, Any]:
    document = candidate.value
    metadata = document.metadata
    try:
        chunk_index = int(metadata["chunk_index"])
        start_ms = int(metadata["start_ms"])
        end_ms = int(metadata["end_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "retrieval candidate is missing valid chunk/timestamp metadata."
        ) from error

    return {
        "rank": rank,
        "score": candidate.score,
        "video_id": candidate.video_id,
        "chunk_index": chunk_index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "vector_id": f"{candidate.video_id}#{chunk_index}",
    }


def capture_candidates(
    dataset: list[dict[str, Any]],
    *,
    ks: tuple[int, ...],
    per_video_cap: int | None,
) -> dict[str, Any]:
    """Contact the real retriever once per query and save no transcript text."""
    # These imports stay inside the opt-in live path. Importing this module for
    # offline replay therefore cannot initialise embeddings or Pinecone.
    from src.rag.retriever import retrieve_scored
    from src.rag.vector_store import index_stats

    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain only positive values.")
    if settings.retrive_K not in ks:
        raise ValueError(
            "ks must include the configured RETRIVE_K so the report contains "
            "the actual production context size."
        )

    max_k = max(ks)
    candidates_by_id: dict[str, list[dict[str, Any]]] = {}
    for item in dataset:
        ranked = retrieve_scored(
            item["query"],
            item["scope"],
            k=max_k,
            per_video_cap=per_video_cap,
        )
        candidates_by_id[item["id"]] = [
            _serialise_candidate(candidate, rank=rank)
            for rank, candidate in enumerate(ranked, start=1)
        ]

    # Index stats are useful reproducibility metadata. The query calls already
    # made this an opt-in live operation, so one lightweight stats read does
    # not change the safety boundary.
    stats = index_stats()
    metadata = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "dataset": _dataset_fingerprint(dataset),
        "git": _git_provenance(),
        "retrieval": {
            "provider": "pinecone",
            "metric": "cosine",
            "score_comparator": ">=",
            "selection_order": [
                "scope_filter",
                "score_threshold",
                "per_video_cap",
                "take_k",
            ],
            "ks": list(ks),
            "max_k": max_k,
            "production_k": settings.retrive_K,
            "per_video_cap": per_video_cap,
            "candidate_fetch_policy": (
                "k for a single-video/no-cap scope; otherwise "
                "min(k * 4, k + 20 * scope_size)"
            ),
            "query_mode": "golden question directly; no graph rewrite",
        },
        "index": {
            "name": settings.pinecone_index_name,
            "cloud": settings.pinecone_cloud,
            "region": settings.pinecone_region,
            **stats,
        },
        "embedding": {
            "model": settings.embedding_model,
            "normalized": True,
        },
        "chunking": {
            "chunk_size": settings.chunk_size,
            "overlap_segments": settings.chunk_overlap_segments,
        },
    }
    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "metadata": metadata,
        "score_distribution": _score_distribution(candidates_by_id),
        "candidates": candidates_by_id,
    }


def _validate_capture(
    capture: dict[str, Any],
    dataset: list[dict[str, Any]],
) -> None:
    if capture.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise ValueError("Unsupported retrieval candidate capture schema.")

    metadata = capture.get("metadata")
    candidates = capture.get("candidates")
    if not isinstance(metadata, dict) or not isinstance(candidates, dict):
        raise ValueError("Candidate capture is missing metadata or candidates.")

    recorded_dataset = metadata.get("dataset")
    if not isinstance(recorded_dataset, dict):
        raise ValueError("Candidate capture is missing dataset provenance.")

    current = _dataset_fingerprint(dataset)
    if recorded_dataset.get("sha256") != current["sha256"]:
        raise ValueError(
            "Golden dataset changed since capture. Capture candidates again "
            "before replaying thresholds."
        )
    if recorded_dataset.get("query_ids") != current["query_ids"]:
        raise ValueError("Candidate capture query ids do not match the dataset.")
    if set(candidates) != set(current["query_ids"]):
        raise ValueError("Candidate capture does not contain every golden query.")


def _captured_dataset(
    full_dataset: list[dict[str, Any]],
    capture: dict[str, Any],
) -> list[dict[str, Any]]:
    """Restore the exact captured subset in its original query order."""
    try:
        query_ids = capture["metadata"]["dataset"]["query_ids"]
    except (KeyError, TypeError) as error:
        raise ValueError("Candidate capture is missing dataset query ids.") from error
    if not isinstance(query_ids, list):
        raise ValueError("Candidate capture query ids must be a list.")

    by_id = {item["id"]: item for item in full_dataset}
    try:
        return [by_id[query_id] for query_id in query_ids]
    except KeyError as error:
        raise ValueError(
            "Golden dataset no longer contains a query from this capture."
        ) from error


def _candidate_from_record(record: object) -> ScoredCandidate[dict[str, Any]]:
    if not isinstance(record, dict):
        raise ValueError("Candidate records must be objects.")

    try:
        score = float(record["score"])
        video_id = record["video_id"]
        if not isinstance(video_id, str):
            raise TypeError("video_id must be a string")
        metadata = {
            "video_id": video_id,
            "chunk_index": int(record["chunk_index"]),
            "start_ms": int(record["start_ms"]),
            "end_ms": int(record["end_ms"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Candidate record has invalid retrieval metadata.") from error

    if not video_id or metadata["start_ms"] < 0 or metadata["end_ms"] < metadata["start_ms"]:
        raise ValueError("Candidate record has an invalid video id or time range.")

    return ScoredCandidate(value=metadata, score=score, video_id=video_id)


def _score_candidate_rows(
    item: dict[str, Any],
    candidates: list[ScoredCandidate[dict[str, Any]]],
    *,
    ks: tuple[int, ...],
    per_video_cap: int | None,
    score_threshold: float,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """Replay each context size with the same candidate pool production uses."""
    rows: list[dict[str, Any]] = []
    retrieved_by_k: dict[int, list[dict[str, Any]]] = {}
    for k in ks:
        fetch_k = candidate_fetch_limit(
            k=k,
            scope_size=len(item["scope"]),
            per_video_cap=per_video_cap,
        )
        selection = select_evidence(
            candidates[:fetch_k],
            k=k,
            per_video_cap=per_video_cap,
            single_video=len(item["scope"]) == 1,
            score_threshold=score_threshold,
        )
        retrieved = [candidate.value for candidate in selection.selected]
        row = metrics.score_query(item, retrieved)
        row["k"] = k
        row["mrr"] = metrics.mrr(item, retrieved)
        row["retrieved_count"] = len(retrieved)
        rows.append(row)
        retrieved_by_k[k] = retrieved
    return rows, retrieved_by_k


def summarize_evidence_gate(rows: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    """Keep absent refusals and answerable misses separate at one context size."""
    at_k = [row for row in rows if row["k"] == k]
    absent = [row for row in at_k if row["query_type"] == "absent"]
    answerable = [row for row in at_k if row["query_type"] != "absent"]
    if not absent or not answerable:
        raise ValueError("A threshold report needs absent and answerable rows.")

    return {
        "k": k,
        "absent_count": len(absent),
        # For absent rows, `hit` deliberately means correct empty retrieval.
        "absent_refusal_rate": mean(row["hit"] for row in absent),
        "answerable_count": len(answerable),
        "answerable_refusal_rate": mean(
            row["retrieved_count"] == 0 for row in answerable
        ),
        "answerable_hit_rate": mean(row["hit"] for row in answerable),
        "answerable_coverage": mean(row["coverage"] for row in answerable),
        "answerable_mrr": mean(row["mrr"] for row in answerable),
    }


def replay_thresholds(
    capture: dict[str, Any],
    dataset: list[dict[str, Any]],
    *,
    thresholds: tuple[float, ...],
    ks: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate saved pre-threshold candidates without any provider access."""
    _validate_capture(capture, dataset)
    if not thresholds:
        raise ValueError("Provide at least one score threshold to replay.")

    retrieval_metadata = capture["metadata"]["retrieval"]
    captured_ks = tuple(retrieval_metadata["ks"])
    requested_ks = captured_ks if ks is None else ks
    if not requested_ks or any(k <= 0 for k in requested_ks):
        raise ValueError("ks must contain only positive values.")
    if max(requested_ks) > int(retrieval_metadata["max_k"]):
        raise ValueError("Replay cannot request k larger than the captured pool.")

    production_k = int(retrieval_metadata["production_k"])
    if production_k not in requested_ks:
        raise ValueError("Replay ks must include the captured production_k.")

    per_video_cap = retrieval_metadata.get("per_video_cap")
    if per_video_cap is not None:
        per_video_cap = int(per_video_cap)

    by_id = capture["candidates"]
    reports: list[dict[str, Any]] = []
    previously_refused: set[str] = set()
    for threshold in sorted(set(thresholds)):
        rows: list[dict[str, Any]] = []
        refused_ids: set[str] = set()
        for item in dataset:
            raw_candidates = by_id[item["id"]]
            if not isinstance(raw_candidates, list):
                raise ValueError(f"Candidates for {item['id']} must be a list.")
            candidates = [
                _candidate_from_record(record) for record in raw_candidates
            ]
            scored_rows, retrieved_by_k = _score_candidate_rows(
                item,
                candidates,
                ks=requested_ks,
                per_video_cap=per_video_cap,
                score_threshold=threshold,
            )
            rows.extend(scored_rows)
            if not retrieved_by_k[production_k]:
                refused_ids.add(item["id"])

        reports.append(
            {
                "threshold": threshold,
                "summary": summarize_evidence_gate(rows, k=production_k),
                "newly_refused_ids": sorted(refused_ids - previously_refused),
                "refused_ids": sorted(refused_ids),
                "rows": rows,
            }
        )
        previously_refused = refused_ids

    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capture_metadata": capture["metadata"],
        "score_distribution": capture.get("score_distribution"),
        "thresholds": reports,
    }


def _print_threshold_table(report: dict[str, Any]) -> None:
    print(
        "threshold  absent_refuse  answerable_refuse  answerable_hit  "
        "answerable_cover"
    )
    for threshold_report in report["thresholds"]:
        summary = threshold_report["summary"]
        print(
            f"{threshold_report['threshold']:9.4f}  "
            f"{summary['absent_refusal_rate']:14.2%}  "
            f"{summary['answerable_refusal_rate']:17.2%}  "
            f"{summary['answerable_hit_rate']:14.2%}  "
            f"{summary['answerable_coverage']:16.2%}"
        )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture/replay retrieval candidates for no-answer tuning."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-candidates", type=Path)
    mode.add_argument("--replay-candidates", type=Path)
    parser.add_argument(
        "--allow-live-retrieval",
        action="store_true",
        help="Required acknowledgement before capture contacts Pinecone.",
    )
    parser.add_argument("--thresholds", type=float, nargs="+")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Context sizes to diagnose. Capture defaults to 1, 3, the current "
            "RETRIVE_K, and 10; replay defaults to the captured values."
        ),
    )
    parser.add_argument("--per-video-cap", type=int, default=settings.per_video_cap)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.capture_candidates is not None:
        dataset = _read_dataset()
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("limit must be positive.")
            dataset = dataset[: args.limit]
        if not args.allow_live_retrieval:
            raise ValueError(
                "Capture contacts Pinecone. Re-run with --allow-live-retrieval "
                "to acknowledge that live provider call."
            )
        if args.thresholds is not None or args.report is not None:
            raise ValueError("thresholds/report are replay-only options.")

        ks = (
            tuple(args.ks)
            if args.ks is not None
            else tuple(sorted({1, 3, settings.retrive_K, 10}))
        )

        capture = capture_candidates(
            dataset,
            ks=ks,
            per_video_cap=args.per_video_cap,
        )
        _write_json(args.capture_candidates, capture)
        print(
            f"captured {len(dataset)} queries to {args.capture_candidates} "
            f"(no transcript text saved)"
        )
        print(f"score distribution: {capture['score_distribution']}")
        return 0

    if args.allow_live_retrieval:
        raise ValueError("Replay is offline; --allow-live-retrieval is not valid here.")
    if not args.thresholds:
        raise ValueError("Replay requires one or more --thresholds values.")
    if args.limit is not None:
        raise ValueError(
            "Replay always uses the exact query set stored in its capture; "
            "--limit is capture-only."
        )

    capture = json.loads(args.replay_candidates.read_text(encoding="utf-8"))
    dataset = _captured_dataset(_read_dataset(), capture)
    report = replay_thresholds(
        capture,
        dataset,
        thresholds=tuple(args.thresholds),
        ks=None if args.ks is None else tuple(args.ks),
    )
    _print_threshold_table(report)
    if args.report is not None:
        _write_json(args.report, report)
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
