import argparse
import math
import sys
import time
from collections.abc import Callable

from src.rag.ingest import IngestResult
from src.services.ingestion_queue import (
    WorkerCycleResult,
    run_one_ingestion_cycle,
)
from src.services.ingestion_worker import IngestionJobClaimLostError


CycleRunner = Callable[[], WorkerCycleResult]
Sleeper = Callable[[float], None]


def _positive_seconds(raw_value: str) -> float:
    try:
        seconds = float(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "poll interval must be a number of seconds."
        ) from error

    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(
            "poll interval must be greater than zero."
        )

    return seconds


def _run_and_report_cycle(
    run_cycle: CycleRunner,
    *,
    report_idle: bool,
) -> tuple[int, bool]:
    try:
        cycle = run_cycle()
    except IngestionJobClaimLostError:
        print("Job claim was lost; another worker owns that job now.")
        return 0, True
    except Exception as error:
        print(
            f"Worker cycle failed: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1, False

    for job_id in cycle.recovered_job_ids:
        print(f"Recovered expired job: {job_id}")

    if cycle.job_id is None:
        if report_idle:
            print("No runnable ingestion job found.")
        return 0, True

    if cycle.claim_attempt is None or cycle.ingest_result is None:
        print("Worker cycle ended without an ingestion result.", file=sys.stderr)
        return 1, False

    print(f"Processed job: {cycle.job_id}")
    print(f"Attempt: {cycle.claim_attempt}")
    print(f"Result: {cycle.ingest_result.status.value}")

    return 0, False


def main(
    argv: list[str] | None = None,
    *,
    run_cycle: CycleRunner = run_one_ingestion_cycle,
    sleep_fn: Sleeper = time.sleep,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run the durable ingestion worker.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--once",
        action="store_true",
        help="Recover stale jobs and process at most one runnable job.",
    )
    mode.add_argument(
        "--loop",
        action="store_true",
        help="Keep polling for runnable ingestion jobs until stopped.",
    )
    parser.add_argument(
        "--poll-interval",
        type=_positive_seconds,
        default=2.0,
        help="Seconds to wait after an idle worker cycle (default: 2).",
    )

    args = parser.parse_args(argv)
    idle_reported = False

    try:
        while True:
            exit_code, is_idle = _run_and_report_cycle(
                run_cycle,
                report_idle=not idle_reported,
            )

            if exit_code != 0 or args.once:
                return exit_code

            if is_idle:
                idle_reported = True
                sleep_fn(args.poll_interval)
            else:
                idle_reported = False
    except KeyboardInterrupt:
        print("Worker stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())