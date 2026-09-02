import uuid

import pytest

from src.cli.ingestion_worker import main
from src.rag.ingest import IngestResult, VideoStatus
from src.services.ingestion_queue import WorkerCycleResult


def test_once_exits_cleanly_when_no_job_exists(capsys) -> None:
    result = WorkerCycleResult(
        recovered_job_ids=(),
        job_id=None,
        claim_attempt=None,
        ingest_result=None,
    )

    exit_code = main(
        ["--once"],
        run_cycle=lambda: result,
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "No runnable ingestion job found.\n"


def test_once_prints_a_processed_job(capsys) -> None:
    job_id = uuid.uuid4()
    result = WorkerCycleResult(
        recovered_job_ids=(),
        job_id=job_id,
        claim_attempt=1,
        ingest_result=IngestResult(
            video_id="dQw4w9WgXcQ",
            status=VideoStatus.OK,
            chunks=2,
            vectors=2,
        ),
    )

    exit_code = main(
        ["--once"],
        run_cycle=lambda: result,
    )

    assert exit_code == 0

    output = capsys.readouterr().out
    assert f"Processed job: {job_id}" in output
    assert "Attempt: 1" in output
    assert "Result: ok" in output


def test_once_is_required() -> None:
    with pytest.raises(SystemExit) as error:
        main([])

    assert error.value.code == 2