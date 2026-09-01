import argparse
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.models.ingestion_job import IngestionJobStatus
from src.db.models.source_video import SourceVideoStatus
from src.db.repositories.sources import SourceProgress, get_source_progress
from src.db.session import SessionLocal


ProgressReader = Callable[[Session, UUID], SourceProgress]
SessionFactory = Callable[[], Session]


def main(
    argv: list[str] | None = None,
    *,
    get_progress_fn: ProgressReader = get_source_progress,
    session_factory: SessionFactory = SessionLocal,
) -> int:
    parser = argparse.ArgumentParser(
        description="Show ingestion progress for one source."
    )
    parser.add_argument(
        "source_id",
        help="UUID of the source to inspect.",
    )
    args = parser.parse_args(argv)

    try:
        source_id = UUID(args.source_id)
    except ValueError:
        parser.error("source_id must be a valid UUID.")

    session = session_factory()
    try:
        progress = get_progress_fn(
            session,
            source_id=source_id,
        )
    except LookupError as error:
        parser.error(str(error))
    finally:
        session.close()

    print(f"Source: {progress.source_id}")
    print(f"Status: {progress.source_status.value}")
    print(
        f"Videos: {progress.completed_videos}/{progress.total_videos} completed"
    )

    print("Video states:")
    for status in SourceVideoStatus:
        print(f"  {status.value}: {progress.video_counts[status.value]}")

    print("Job states:")
    for status in IngestionJobStatus:
        print(f"  {status.value}: {progress.job_counts[status.value]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())