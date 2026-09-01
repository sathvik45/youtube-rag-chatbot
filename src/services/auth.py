from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.core.security import hash_password, verify_password
from src.db.repositories.users import (
    create_user,
    get_user_by_email,
)
from src.db.session import SessionLocal


MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128

SessionFactory = Callable[[], Session]


class EmailAlreadyRegisteredError(ValueError):
    """Raised when an account already uses the submitted email."""


class InvalidEmailError(ValueError):
    """Raised when an email cannot be used for registration."""


class InvalidPasswordError(ValueError):
    """Raised when a password does not meet the basic policy."""

class InvalidCredentialsError(ValueError):
    """Raised when email/password authentication fails."""


@dataclass(frozen=True)
class RegisteredUser:
    id: UUID
    email: str

@dataclass(frozen=True)
class AuthenticatedUser:
    id: UUID
    email: str

# Used when no user exists, so login timing does not reveal account existence.
DUMMY_PASSWORD_HASH = hash_password(
    "not-a-real-user-password-for-timing-protection"
)

def normalize_email(email: str) -> str:
    """Apply one canonical representation before storing an email."""
    normalized = email.strip().lower()

    if not normalized or "@" not in normalized:
        raise InvalidEmailError("A valid email address is required.")

    return normalized


def validate_password(password: str) -> None:
    """Enforce a practical initial password-length policy."""
    if not password.strip():
        raise InvalidPasswordError("Password cannot be blank.")

    if len(password) < MIN_PASSWORD_LENGTH:
        raise InvalidPasswordError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        )

    if len(password) > MAX_PASSWORD_LENGTH:
        raise InvalidPasswordError(
            f"Password must be at most {MAX_PASSWORD_LENGTH} characters."
        )


def register_user(
    email: str,
    password: str,
    *,
    session_factory: SessionFactory = SessionLocal,
) -> RegisteredUser:
    """Create one user with a securely hashed password."""
    normalized_email = normalize_email(email)
    validate_password(password)

    # Argon2 is deliberately expensive, so hash before opening a DB transaction.
    password_hash = hash_password(password)

    session = session_factory()
    try:
        try:
            with session.begin():
                existing_user = get_user_by_email(
                    session,
                    email=normalized_email,
                )

                if existing_user is not None:
                    raise EmailAlreadyRegisteredError(
                        "An account with this email already exists."
                    )

                user = create_user(
                    session,
                    email=normalized_email,
                    password_hash=password_hash,
                )

                return RegisteredUser(
                    id=user.id,
                    email=user.email,
                )

        # The pre-check gives a friendly error, but this handles a race where
        # two registration requests check the same email simultaneously.
        except IntegrityError as error:
            raise EmailAlreadyRegisteredError(
                "An account with this email already exists."
            ) from error

    finally:
        session.close()

def authenticate_user(
    email: str,
    password: str,
    *,
    session_factory: SessionFactory = SessionLocal,
) -> AuthenticatedUser:
    """Verify credentials without revealing which part was incorrect."""
    try:
        normalized_email = normalize_email(email)
    except InvalidEmailError as error:
        verify_password(password, DUMMY_PASSWORD_HASH)
        raise InvalidCredentialsError(
            "Incorrect email or password."
        ) from error

    session = session_factory()
    try:
        user = get_user_by_email(
            session,
            email=normalized_email,
        )

        if user is None:
            verify_password(password, DUMMY_PASSWORD_HASH)
            raise InvalidCredentialsError(
                "Incorrect email or password."
            )

        password_is_correct = verify_password(
            password,
            user.password_hash,
        )

        if not password_is_correct or not user.is_active:
            raise InvalidCredentialsError(
                "Incorrect email or password."
            )

        return AuthenticatedUser(
            id=user.id,
            email=user.email,
        )

    finally:
        session.close()