from datetime import timedelta

import pytest

from src.db.repositories.ingestion_jobs import retry_delay_for_attempt


@pytest.mark.parametrize(
    ("attempts", "expected_delay"),
    [
        (1, timedelta(seconds=30)),
        (2, timedelta(minutes=1)),
        (3, timedelta(minutes=2)),
        (4, timedelta(minutes=4)),
        (5, timedelta(minutes=5)),
        (6, timedelta(minutes=5)),
    ],
)
def test_retry_delay_grows_exponentially_and_is_capped(
    attempts: int,
    expected_delay: timedelta,
) -> None:
    assert retry_delay_for_attempt(attempts) == expected_delay


def test_retry_delay_rejects_non_positive_attempts() -> None:
    with pytest.raises(ValueError, match="attempts must be positive"):
        retry_delay_for_attempt(0)