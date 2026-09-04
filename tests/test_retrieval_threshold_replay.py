"""Offline calibration tests for the score-based no-answer gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.scorers import run_retrieval_eval as runner


FIXTURE_DATASET = Path(__file__).parent / "fixtures" / "retrieval_small_golden.json"


def _dataset() -> list[dict]:
    return json.loads(FIXTURE_DATASET.read_text(encoding="utf-8"))


def _candidate(*, score: float, video_id: str = "abcdefghijk") -> dict:
    return {
        "rank": 1,
        "score": score,
        "video_id": video_id,
        "chunk_index": 0,
        "start_ms": 0,
        "end_ms": 1_000,
        "vector_id": f"{video_id}#0",
    }


def _capture(dataset: list[dict]) -> dict:
    return {
        "schema_version": runner.CAPTURE_SCHEMA_VERSION,
        "metadata": {
            "dataset": runner._dataset_fingerprint(dataset),
            "retrieval": {
                "ks": [5],
                "max_k": 5,
                "production_k": 5,
                "per_video_cap": None,
            },
        },
        "candidates": {
            "absent": [_candidate(score=0.49)],
            "answerable": [_candidate(score=0.50)],
        },
    }


def test_offline_replay_exposes_the_refusal_tradeoff(
    monkeypatch,
) -> None:
    dataset = _dataset()
    monkeypatch.setattr(runner, "DATASET", FIXTURE_DATASET)
    capture = _capture(dataset)

    report = runner.replay_thresholds(
        capture,
        dataset,
        thresholds=(0.49, 0.50),
        ks=(5,),
    )

    low, selected = report["thresholds"]
    assert low["summary"]["absent_refusal_rate"] == 0.0
    assert selected["summary"]["absent_refusal_rate"] == 1.0
    assert selected["summary"]["answerable_refusal_rate"] == 0.0
    assert selected["summary"]["answerable_hit_rate"] == 1.0
    assert selected["newly_refused_ids"] == ["absent"]


def test_capture_mode_requires_explicit_live_provider_acknowledgement(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "DATASET", FIXTURE_DATASET)

    with pytest.raises(ValueError, match="allow-live-retrieval"):
        runner.main(
            [
                "--capture-candidates",
                str(FIXTURE_DATASET.parent / "would-not-write.json"),
            ]
        )
