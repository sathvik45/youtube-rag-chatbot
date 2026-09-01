from datetime import datetime, timedelta, timezone
import uuid

import jwt
import pytest

from src.core import security


@pytest.fixture(autouse=True)
def test_jwt_settings(monkeypatch):
    """Keep tests independent from the real JWT secret in .env."""
    monkeypatch.setattr(
        security.settings,
        "jwt_secret_key",
        "test-secret-that-is-at-least-32-bytes-long",
    )
    monkeypatch.setattr(
        security.settings,
        "access_token_expire_minutes",
        30,
    )


def test_password_hashing_verifies_only_the_correct_password() -> None:
    password = "correct-horse-battery-staple"
    stored_hash = security.hash_password(password)

    assert stored_hash != password
    assert security.verify_password(password, stored_hash) is True
    assert security.verify_password("wrong-password", stored_hash) is False


def test_access_token_round_trip_returns_the_same_user_id() -> None:
    user_id = uuid.uuid4()

    token = security.create_access_token(user_id)

    assert security.get_user_id_from_token(token) == user_id


def test_expired_access_token_is_rejected() -> None:
    expired_token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        },
        security.settings.jwt_secret_key,
        algorithm=security.JWT_ALGORITHM,
    )

    with pytest.raises(
        security.TokenValidationError,
        match="Invalid or expired",
    ):
        security.get_user_id_from_token(expired_token)


def test_token_signed_with_a_different_secret_is_rejected() -> None:
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
        },
        "other-test-secret-that-is-at-least-32-bytes",
        algorithm=security.JWT_ALGORITHM,
    )

    with pytest.raises(
        security.TokenValidationError,
        match="Invalid or expired",
    ):
        security.get_user_id_from_token(token)