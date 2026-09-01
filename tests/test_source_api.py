import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.deps import (
    get_source_progress_reader,
    get_source_submitter,
)
from src.db.models.ingestion_job import IngestionJobStatus
from src.db.models.source import SourceStatus
from src.db.models.source_video import SourceVideoStatus
from src.db.repositories.sources import SourceProgress
from src.main import app
from src.services.source_submission import SourceSubmission


@pytest.fixture
def client():
    app.dependency_overrides.clear()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_create_source_returns_submission_ids(client) -> None:
    user_id = uuid.uuid4()
    source_id = uuid.uuid4()
    video_id = uuid.uuid4()
    job_id = uuid.uuid4()

    submission = SourceSubmission(
        source_id=source_id,
        video_ids=(video_id,),
        job_ids=(job_id,),
    )

    received_requests: list[tuple[uuid.UUID, str]] = []

    def fake_submit_source(
        received_user_id: uuid.UUID,
        received_url: str,
    ) -> SourceSubmission:
        received_requests.append(
            (received_user_id, received_url)
        )
        return submission

    app.dependency_overrides[get_source_submitter] = (
        lambda: fake_submit_source
    )

    response = client.post(
        "/sources",
        json={
            "user_id": str(user_id),
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        },
    )

    assert response.status_code == 201
    assert response.json() == {
        "source_id": str(source_id),
        "video_ids": [str(video_id)],
        "job_ids": [str(job_id)],
    }
    assert received_requests == [
        (
            user_id,
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
    ]


def test_get_source_status_returns_progress(client) -> None:
    source_id = uuid.uuid4()

    video_counts = {
        status.value: 0
        for status in SourceVideoStatus
    }
    video_counts["ready"] = 2
    video_counts["processing"] = 1

    job_counts = {
        status.value: 0
        for status in IngestionJobStatus
    }
    job_counts["succeeded"] = 2
    job_counts["running"] = 1

    progress = SourceProgress(
        source_id=source_id,
        source_status=SourceStatus.PROCESSING,
        video_counts=video_counts,
        job_counts=job_counts,
    )

    received_source_ids: list[uuid.UUID] = []

    def fake_read_progress(
        received_source_id: uuid.UUID,
    ) -> SourceProgress:
        received_source_ids.append(received_source_id)
        return progress

    app.dependency_overrides[get_source_progress_reader] = (
        lambda: fake_read_progress
    )

    response = client.get(f"/sources/{source_id}")

    assert response.status_code == 200
    assert response.json() == {
        "source_id": str(source_id),
        "status": "processing",
        "video_counts": video_counts,
        "job_counts": job_counts,
        "total_videos": 3,
        "completed_videos": 2,
    }
    assert received_source_ids == [source_id]


def test_create_source_rejects_blank_url_before_calling_service(client) -> None:
    def fake_submit_source(
        _: uuid.UUID,
        __: str,
    ) -> SourceSubmission:
        raise AssertionError("Service should not be called.")

    app.dependency_overrides[get_source_submitter] = (
        lambda: fake_submit_source
    )

    response = client.post(
        "/sources",
        json={
            "user_id": str(uuid.uuid4()),
            "youtube_url": "   ",
        },
    )

    assert response.status_code == 422