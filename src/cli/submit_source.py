import argparse
from collections.abc import Callable
from uuid import UUID

from src.rag.ingest import IngestResult
from src.services.ingestion_dispatcher import run_runnable_jobs_for_source
from src.services.source_submission import SourceSubmission, submit_source


SubmitSourceFunction = Callable[[UUID, str], SourceSubmission]
DispatchJobsFunction = Callable[[UUID], list[IngestResult]]


def main(
    argv: list[str] | None = None,
    *,
    submit_source_fn: SubmitSourceFunction = submit_source,
    dispatch_jobs_fn: DispatchJobsFunction = run_runnable_jobs_for_source,
) -> int:
    parser = argparse.ArgumentParser(
        description="Submit a YouTube video or playlist for ingestion."
    )
    parser.add_argument(
        "user_id",
        help="UUID of the user submitting the source.",
    )
    parser.add_argument(
        "youtube_url",
        help="YouTube video or playlist URL.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Immediately run the queued ingestion jobs.",
    )
    args = parser.parse_args(argv)

    try:
        user_id = UUID(args.user_id)
    except ValueError:
        parser.error("user_id must be a valid UUID.")

    submission = submit_source_fn(
        user_id,
        args.youtube_url,
    )

    print(f"Source created: {submission.source_id}")
    print(f"Videos resolved: {len(submission.video_ids)}")
    print(f"Jobs queued: {len(submission.job_ids)}")

    if not args.run:
        return 0

    results = dispatch_jobs_fn(submission.source_id)

    for result in results:
        print(f"{result.video_id}: {result.status.value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())