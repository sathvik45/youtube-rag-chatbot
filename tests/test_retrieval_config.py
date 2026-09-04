"""Fail-fast configuration checks for retrieval evidence selection."""

import pytest
from pydantic import ValidationError

from src.core.config import Settings


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RETRIVE_K", "0"),
        ("PER_VIDEO_CAP", "0"),
        ("SCORE_THRESHOLD", "nan"),
        ("SCORE_THRESHOLD", "inf"),
    ],
)
def test_invalid_retrieval_evidence_settings_fail_at_startup(
    monkeypatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings()
