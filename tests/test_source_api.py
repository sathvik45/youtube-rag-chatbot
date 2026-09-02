import uuid
from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient

from src.api.deps import (
    get_current_user,
    get_source_progress_reader,
    get_source_submitter,
    get_source_list_reader,
)
from src.db.models.ingestion_job import IngestionJobStatus
from src.db.models.source import Source, SourceStatus, SourceType
from src.db.models.source_video import SourceVideoStatus
from src.db.repositories.sources import SourceProgress
from src.main import app
from src.services.auth import AuthenticatedUser
from src.services.source_submission import SourceSubmission

from src.api.cursors import decode_source_cursor

@pytest.fixture
def client():
    app.dependency_overrides.clear()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def authenticated_user() -> AuthenticatedUser:
    return AuthenticatedUser(
        id=uuid.uuid4(),
        email="api-test@example.invalid",
    )


def test_create_source_returns_submission_ids(
    client,
    authenticated_user,
) -> None:
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
        received_requests.append((received_user_id, received_url))
        return submission

    app.dependency_overrides[get_current_user] = (
        lambda: authenticated_user
    )
    app.dependency_overrides[get_source_submitter] = (
        lambda: fake_submit_source
    )

    response = client.post(
        "/sources",
        json={
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
            authenticated_user.id,
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
    ]


def test_get_source_status_returns_progress(
    client,
    authenticated_user,
) -> None:
    source_id = uuid.uuid4()

    video_counts = {status.value: 0 for status in SourceVideoStatus}
    video_counts["ready"] = 2
    video_counts["processing"] = 1

    job_counts = {status.value: 0 for status in IngestionJobStatus}
    job_counts["succeeded"] = 2
    job_counts["running"] = 1

    progress = SourceProgress(
        source_id=source_id,
        source_status=SourceStatus.PROCESSING,
        video_counts=video_counts,
        job_counts=job_counts,
    )

    received_requests: list[tuple[uuid.UUID, uuid.UUID]] = []

    def fake_read_progress(
        received_user_id: uuid.UUID,
        received_source_id: uuid.UUID,
    ) -> SourceProgress:
        received_requests.append(
            (received_user_id, received_source_id)
        )
        return progress

    app.dependency_overrides[get_current_user] = (
        lambda: authenticated_user
    )
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
    assert received_requests == [(authenticated_user.id, source_id)]


def test_create_source_rejects_blank_url_before_calling_service(
    client,
    authenticated_user,
) -> None:
    def fake_submit_source(
        _: uuid.UUID,
        __: str,
    ) -> SourceSubmission:
        raise AssertionError("Service should not be called.")

    app.dependency_overrides[get_current_user] = (
        lambda: authenticated_user
    )
    app.dependency_overrides[get_source_submitter] = (
        lambda: fake_submit_source
    )

    response = client.post(
        "/sources",
        json={"youtube_url": "   "},
    )

    assert response.status_code == 422

def test_create_source_rejects_invalid_nonblank_youtube_url(
    client,
    authenticated_user,
) -> None:
    invalid_url = "https://example.com/not-a-youtube-url"

    def fake_submit_source(
        received_user_id: uuid.UUID,
        received_url: str,
    ) -> SourceSubmission:
        assert received_user_id == authenticated_user.id
        assert received_url == invalid_url
        raise ValueError(
            "Please provide a valid YouTube video or playlist URL."
        )

    app.dependency_overrides[get_current_user] = (
        lambda: authenticated_user
    )
    app.dependency_overrides[get_source_submitter] = (
        lambda: fake_submit_source
    )

    response = client.post(
        "/sources",
        json={"youtube_url": invalid_url},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Please provide a valid YouTube video or playlist URL."
    }


def test_create_source_rejects_client_supplied_user_id(
    client,
    authenticated_user,
) -> None:
    app.dependency_overrides[get_current_user] = (
        lambda: authenticated_user
    )

    response = client.post(
        "/sources",
        json={
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "user_id": str(uuid.uuid4()),
        },
    )

    assert response.status_code == 422


def test_source_routes_require_a_bearer_token(client) -> None:
    response = client.get(f"/sources/{uuid.uuid4()}")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"

def test_list_sources_returns_a_page_and_next_cursor(
    client,
    authenticated_user,
) -> None:
    newest_created_at = datetime.now(timezone.utc)
    older_created_at = newest_created_at - timedelta(minutes=1)

    newest_source = Source(
        id=uuid.uuid4(),
        user_id=authenticated_user.id,
        source_type=SourceType.PLAYLIST,
        submitted_value="https://www.youtube.com/playlist?list=PLnewest",
        status=SourceStatus.PROCESSING,
        created_at=newest_created_at,
    )
    older_source = Source(
        id=uuid.uuid4(),
        user_id=authenticated_user.id,
        source_type=SourceType.VIDEO,
        submitted_value="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        status=SourceStatus.PENDING,
        created_at=older_created_at,
    )

    received_requests: list[
        tuple[uuid.UUID, datetime | None, uuid.UUID | None, int]
    ] = []

    def fake_read_sources(
        received_user_id: uuid.UUID,
        before_created_at: datetime | None,
        before_id: uuid.UUID | None,
        received_limit: int,
    ) -> list[Source]:
        received_requests.append(
            (
                received_user_id,
                before_created_at,
                before_id,
                received_limit,
            )
        )
        return [newest_source, older_source]

    app.dependency_overrides[get_current_user] = (
        lambda: authenticated_user
    )
    app.dependency_overrides[get_source_list_reader] = (
        lambda: fake_read_sources
    )

    response = client.get("/sources?limit=1")

    assert response.status_code == 200

    response_body = response.json()
    assert len(response_body["items"]) == 1

    item = response_body["items"][0]
    assert item["id"] == str(newest_source.id)
    assert item["submitted_value"] == newest_source.submitted_value
    assert item["source_type"] == "playlist"
    assert item["status"] == "processing"
    assert datetime.fromisoformat(item["created_at"]) == newest_created_at

    assert response_body["next_cursor"] is not None
    assert decode_source_cursor(
        response_body["next_cursor"]
    ) == (newest_created_at, newest_source.id)

    assert received_requests == [
        (
            authenticated_user.id,
            None,
            None,
            2,
        )
    ]


def test_list_sources_rejects_an_invalid_cursor(
    client,
    authenticated_user,
) -> None:
    def fake_read_sources(
        _: uuid.UUID,
        __: datetime | None,
        ___: uuid.UUID | None,
        ____: int,
    ) -> list[Source]:
        raise AssertionError("The reader should not be called.")

    app.dependency_overrides[get_current_user] = (
        lambda: authenticated_user
    )
    app.dependency_overrides[get_source_list_reader] = (
        lambda: fake_read_sources
    )

    response = client.get("/sources?cursor=not-a-valid-cursor")

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Invalid source cursor."
    }