from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash

from src.core.config import settings


password_hash = PasswordHash.recommended()
JWT_ALGORITHM = "HS256"


class TokenValidationError(ValueError):
    """Raised when an access token cannot identify a valid user."""


def hash_password(password: str) -> str:
    """Return a secure Argon2 password hash."""
    return password_hash.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """Return whether a plaintext password matches its stored hash."""
    return password_hash.verify(
        plain_password,
        hashed_password,
    )


def create_access_token(user_id: UUID) -> str:
    """Create an expiring signed token identifying one user."""
    if not settings.jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY must be configured.")

    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )

    return jwt.encode(
        {
            "sub": str(user_id),
            "exp": expires_at,
        },
        settings.jwt_secret_key,
        algorithm=JWT_ALGORITHM,
    )


def get_user_id_from_token(token: str) -> UUID:
    """Verify a token and return the user UUID stored in its subject claim."""
    if not settings.jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY must be configured.")

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[JWT_ALGORITHM],
        )
        subject = payload.get("sub")

        if not subject:
            raise TokenValidationError("Token has no subject.")

        return UUID(subject)

    except (InvalidTokenError, ValueError) as error:
        raise TokenValidationError("Invalid or expired access token.") from error