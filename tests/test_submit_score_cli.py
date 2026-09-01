import uuid

from src.cli.submit_source import main
from src.rag.ingest import IngestResult, VideoStatus
from src.services.source_submission import SourceSubmission


def test_cli_submits_source_without_running_jobs(capsys) -> None:
    user_id = uuid.uuid4()
    source_id = uuid.uuid4()
    video_id = uuid.uuid4()
    job_id = uuid.uuid4()

    submission = SourceSubmission(
        source_id=source_id,
        video_ids=(video_id,),
        job_ids=(job_id,),
    )

    def fake_submit_source(
        received_user_id: uuid.UUID,
        received_url: str,
    ) -> SourceSubmission:
        assert received_user_id == user_id
        assert received_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        return submission

    def fake_dispatch_jobs(_: uuid.UUID) -> list[IngestResult]:
        raise AssertionError("Dispatcher should not run without --run.")

    exit_code = main(
        [
            str(user_id),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ],
        submit_source_fn=fake_submit_source,
        dispatch_jobs_fn=fake_dispatch_jobs,
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"Source created: {source_id}" in output
    assert "Videos resolved: 1" in output
    assert "Jobs queued: 1" in output


def test_cli_runs_jobs_when_run_flag_is_provided(capsys) -> None:
    user_id = uuid.uuid4()
    source_id = uuid.uuid4()

    submission = SourceSubmission(
        source_id=source_id,
        video_ids=(uuid.uuid4(),),
        job_ids=(uuid.uuid4(),),
    )

    def fake_submit_source(
        _: uuid.UUID,
        __: str,
    ) -> SourceSubmission:
        return submission

    dispatched_source_ids: list[uuid.UUID] = []

    def fake_dispatch_jobs(received_source_id: uuid.UUID) -> list[IngestResult]:
        dispatched_source_ids.append(received_source_id)

        return [
            IngestResult(
                video_id="dQw4w9WgXcQ",
                status=VideoStatus.OK,
                chunks=3,
                vectors=3,
            )
        ]

    exit_code = main(
        [
            str(user_id),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "--run",
        ],
        submit_source_fn=fake_submit_source,
        dispatch_jobs_fn=fake_dispatch_jobs,
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert dispatched_source_ids == [source_id]
    assert "dQw4w9WgXcQ: ok" in output