"""Opt-in integration test for authenticated source creation."""

import os
import uuid
from functools import partial

import pytest

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from src.api.deps import (
    get_session_factory,
    get_source_submitter,
)
from src.core.security import create_access_token
from src.db.models.ingestion_job import (
    IngestionJob,
    IngestionJobStatus,
)
from src.db.models.source import Source, SourceStatus, SourceType
from src.db.models.source_video import SourceVideo, SourceVideoStatus
from src.db.models.user import User
from src.db.models.video import TranscriptStatus, Video
from src.db.session import engine
from src.main import app

from src.services.source_submission import submit_source


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against PostgreSQL.",
)


@pytest.fixture
def transactional_session_factory():
    """Allow internal transactions, then roll back all test rows."""
    connection = engine.connect()
    outer_transaction = connection.begin()

    factory = sessionmaker(
        bind=connection,
        autoflush=False,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    try:
        yield factory
    finally:
        outer_transaction.rollback()
        connection.close()

@pytest.fixture
def transactional_client(transactional_session_factory):
    """Provide an API client and clean up dependency overrides."""
    app.dependency_overrides.clear()

    app.dependency_overrides[get_session_factory] = (
        lambda: transactional_session_factory
    )
    
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def test_authenticated_user_can_create_playlist_source(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex
    playlist_id = f"PL{suffix[:12]}"
    submitted_url = (
        f"https://www.youtube.com/playlist?list={playlist_id}"
    )
    resolved_video_ids = [
        suffix[:11],
        suffix[11:22],
    ]

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"source-api-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()
            user_id = user.id

    received_urls: list[str] = []

    def fake_resolve(received_url: str) -> list[str]:
        received_urls.append(received_url)
        return resolved_video_ids

    app.dependency_overrides[get_source_submitter] = lambda: partial(
        submit_source,
        resolve=fake_resolve,
        session_factory=transactional_session_factory,
    )



    token = create_access_token(user_id)

    response = transactional_client.post(
        "/sources",
        headers={"Authorization": f"Bearer {token}"},
        json={"youtube_url": submitted_url},
    )

    assert response.status_code == 201
    assert received_urls == [submitted_url]

    response_body = response.json()
    source_id = uuid.UUID(response_body["source_id"])
    video_ids = [
        uuid.UUID(video_id)
        for video_id in response_body["video_ids"]
    ]
    job_ids = [
        uuid.UUID(job_id)
        for job_id in response_body["job_ids"]
    ]

    assert len(video_ids) == 2
    assert len(job_ids) == 2

    with transactional_session_factory() as session:
        source = session.get(Source, source_id)

        videos = session.scalars(
            select(Video).where(Video.id.in_(video_ids))
        ).all()

        source_videos = session.scalars(
            select(SourceVideo)
            .where(SourceVideo.source_id == source_id)
            .order_by(SourceVideo.position)
        ).all()

        jobs = session.scalars(
            select(IngestionJob)
            .where(IngestionJob.source_id == source_id)
            .order_by(IngestionJob.id)
        ).all()

    assert source is not None
    assert source.user_id == user_id
    assert source.source_type is SourceType.PLAYLIST
    assert source.youtube_playlist_id == playlist_id
    assert source.status is SourceStatus.PROCESSING

    assert {video.youtube_video_id for video in videos} == set(
        resolved_video_ids
    )
    assert all(
        video.transcript_status is TranscriptStatus.PENDING
        for video in videos
    )

    assert [link.position for link in source_videos] == [0, 1]
    assert {link.video_id for link in source_videos} == set(video_ids)
    assert all(
        link.status is SourceVideoStatus.PENDING
        for link in source_videos
    )

    assert {job.id for job in jobs} == set(job_ids)
    assert {job.video_id for job in jobs} == set(video_ids)
    assert all(
        job.status is IngestionJobStatus.QUEUED
        for job in jobs
    )