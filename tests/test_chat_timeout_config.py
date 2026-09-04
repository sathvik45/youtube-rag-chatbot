"""Configuration checks for the request-level RAG deadline."""

import pytest
from pydantic import ValidationError

from src.core.config import Settings


def test_chat_rag_timeout_uses_environment_value(monkeypatch) -> None:
    """The documented environment variable controls the default guard."""
    monkeypatch.setenv("CHAT_RAG_TIMEOUT_SECONDS", "12.5")

    assert Settings().chat_rag_timeout_seconds == 12.5


@pytest.mark.parametrize("value", ["0", "301"])
def test_chat_rag_timeout_rejects_unsafe_values(monkeypatch, value: str) -> None:
    """A missing or excessive deadline must fail fast during startup."""
    monkeypatch.setenv("CHAT_RAG_TIMEOUT_SECONDS", value)

    with pytest.raises(ValidationError):
        Settings()
