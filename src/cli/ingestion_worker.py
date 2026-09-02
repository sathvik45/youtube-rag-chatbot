import argparse
import sys
from collections.abc import Callable

from src.services.ingestion_queue import (
    WorkerCycleResult,
    run_one_ingestion_cycle,
)
from src.services.ingestion_worker import IngestionJobClaimLostError


CycleRunner = Callable[[], WorkerCycleResult]


def main(
    argv: list[str] | None = None,
    *,
    run_cycle: CycleRunner = run_one_ingestion_cycle,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run one durable ingestion-worker cycle.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        required=True,
        help="Recover stale jobs and process at most one runnable job.",
    )
    args = parser.parse_args(argv)

    try:
        cycle = run_cycle()
    except IngestionJobClaimLostError:
        print("Job claim was lost; another worker owns that job now.")
        return 0
    except Exception as error:
        print(
            f"Worker cycle failed: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1

    for job_id in cycle.recovered_job_ids:
        print(f"Recovered expired job: {job_id}")

    if cycle.job_id is None:
        print("No runnable ingestion job found.")
        return 0

    if cycle.ingest_result is None:
        print("Worker cycle ended without an ingestion result.", file=sys.stderr)
        return 1

    print(f"Processed job: {cycle.job_id}")
    print(f"Attempt: {cycle.claim_attempt}")
    print(f"Result: {cycle.ingest_result.status.value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())