import os
import uuid

import pytest
from sqlalchemy.orm import sessionmaker

from src.core.security import verify_password
from src.db.repositories.users import get_user_by_email
from src.db.session import SessionLocal, engine
from src.services.auth import (
    EmailAlreadyRegisteredError,
    InvalidPasswordError,
    register_user,
    authenticate_user,
    InvalidCredentialsError,
)


@pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against the configured PostgreSQL database.",
)
def test_register_user_hashes_normalizes_rejects_duplicates_and_rolls_back() -> None:
    suffix = uuid.uuid4().hex
    submitted_email = f"  AUTH-{suffix}@EXAMPLE.INVALID  "
    normalized_email = f"auth-{suffix}@example.invalid"
    password = "a-long-enough-password"

    connection = engine.connect()
    outer_transaction = connection.begin()

    test_session_factory = sessionmaker(
        bind=connection,
        autoflush=False,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    try:
        registered_user = register_user(
            submitted_email,
            password,
            session_factory=test_session_factory,
        )

        assert registered_user.email == normalized_email

        with test_session_factory() as session:
            stored_user = get_user_by_email(
                session,
                email=normalized_email,
            )

            assert stored_user is not None
            assert stored_user.id == registered_user.id
            assert stored_user.password_hash != password
            assert verify_password(
                password,
                stored_user.password_hash,
            ) is True

        with pytest.raises(
            EmailAlreadyRegisteredError,
            match="already exists",
        ):
            register_user(
                normalized_email,
                password,
                session_factory=test_session_factory,
            )

    finally:
        outer_transaction.rollback()
        connection.close()

    with SessionLocal() as verification_session:
        assert get_user_by_email(
            verification_session,
            email=normalized_email,
        ) is None


def test_register_user_rejects_short_password_before_using_database() -> None:
    with pytest.raises(
        InvalidPasswordError,
        match="at least 12 characters",
    ):
        register_user(
            "learner@example.invalid",
            "too-short",
        )

@pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against the configured PostgreSQL database.",
)
def test_authenticate_user_handles_valid_wrong_unknown_and_inactive_accounts() -> None:
    suffix = uuid.uuid4().hex
    email = f"login-test-{suffix}@example.invalid"
    password = "a-long-enough-password"

    connection = engine.connect()
    outer_transaction = connection.begin()

    test_session_factory = sessionmaker(
        bind=connection,
        autoflush=False,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    try:
        registered_user = register_user(
            email,
            password,
            session_factory=test_session_factory,
        )

        authenticated_user = authenticate_user(
            email,
            password,
            session_factory=test_session_factory,
        )

        assert authenticated_user.id == registered_user.id
        assert authenticated_user.email == email

        with pytest.raises(
            InvalidCredentialsError,
            match="Incorrect email or password",
        ):
            authenticate_user(
                email,
                "wrong-password",
                session_factory=test_session_factory,
            )

        with pytest.raises(
            InvalidCredentialsError,
            match="Incorrect email or password",
        ):
            authenticate_user(
                f"unknown-{suffix}@example.invalid",
                password,
                session_factory=test_session_factory,
            )

        with test_session_factory() as session:
            with session.begin():
                user = get_user_by_email(
                    session,
                    email=email,
                )
                assert user is not None

                user.is_active = False
                session.flush()

        with pytest.raises(
            InvalidCredentialsError,
            match="Incorrect email or password",
        ):
            authenticate_user(
                email,
                password,
                session_factory=test_session_factory,
            )

    finally:
        outer_transaction.rollback()
        connection.close()