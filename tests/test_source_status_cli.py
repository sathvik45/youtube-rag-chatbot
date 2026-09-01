import uuid

from src.cli.source_status import main
from src.db.models.ingestion_job import IngestionJobStatus
from src.db.models.source import SourceStatus
from src.db.models.source_video import SourceVideoStatus
from src.db.repositories.sources import SourceProgress


class FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_cli_prints_source_progress(capsys) -> None:
    source_id = uuid.uuid4()
    fake_session = FakeSession()

    video_counts = {
        status.value: 0
        for status in SourceVideoStatus
    }
    video_counts["processing"] = 1
    video_counts["ready"] = 2

    job_counts = {
        status.value: 0
        for status in IngestionJobStatus
    }
    job_counts["running"] = 1
    job_counts["succeeded"] = 2

    progress = SourceProgress(
        source_id=source_id,
        source_status=SourceStatus.PROCESSING,
        video_counts=video_counts,
        job_counts=job_counts,
    )

    received_source_ids: list[uuid.UUID] = []

    def fake_get_progress(
        received_session,
        *,
        source_id: uuid.UUID,
    ) -> SourceProgress:
        assert received_session is fake_session
        received_source_ids.append(source_id)

        return progress

    exit_code = main(
        [str(source_id)],
        get_progress_fn=fake_get_progress,
        session_factory=lambda: fake_session,
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert received_source_ids == [source_id]
    assert fake_session.closed is True

    assert f"Source: {source_id}" in output
    assert "Status: processing" in output
    assert "Videos: 2/3 completed" in output

    assert "Video states:" in output
    assert "  processing: 1" in output
    assert "  ready: 2" in output

    assert "Job states:" in output
    assert "  running: 1" in output
    assert "  succeeded: 2" in output