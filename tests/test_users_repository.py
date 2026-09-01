import os
import uuid

import pytest
from sqlalchemy.orm import sessionmaker

from src.db.repositories.users import (
    create_user,
    get_user_by_email,
    get_user_by_id,
)
from src.db.session import SessionLocal, engine


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against the configured PostgreSQL database.",
)


def test_user_repository_creates_reads_and_rolls_back() -> None:
    suffix = uuid.uuid4().hex
    email = f"user-repository-{suffix}@example.invalid"

    connection = engine.connect()
    outer_transaction = connection.begin()

    test_session_factory = sessionmaker(
        bind=connection,
        autoflush=False,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    try:
        with test_session_factory() as session:
            with session.begin():
                user = create_user(
                    session,
                    email=email,
                    password_hash="test-only-password-hash",
                )

                found_by_email = get_user_by_email(
                    session,
                    email=email,
                )
                found_by_id = get_user_by_id(
                    session,
                    user_id=user.id,
                )

                assert user.id is not None
                assert found_by_email is not None
                assert found_by_email.id == user.id
                assert found_by_id is not None
                assert found_by_id.email == email

    finally:
        outer_transaction.rollback()
        connection.close()

    # A fresh database session proves the outer rollback removed the row.
    with SessionLocal() as verification_session:
        assert get_user_by_email(
            verification_session,
            email=email,
        ) is None