from collections.abc import Callable
from uuid import UUID

from src.db.repositories.sources import SourceProgress, get_source_progress
from src.db.session import SessionLocal
from src.services.source_submission import SourceSubmission, submit_source

from src.core.security import create_access_token
from src.services.auth import (
    AuthenticatedUser,
    RegisteredUser,
    authenticate_user,
    register_user,
)

SourceSubmitter = Callable[[UUID, str], SourceSubmission]
SourceProgressReader = Callable[[UUID], SourceProgress]

RegistrationService = Callable[[str, str], RegisteredUser]
AuthenticationService = Callable[[str, str], AuthenticatedUser]
TokenIssuer = Callable[[UUID], str]

def get_source_submitter() -> SourceSubmitter:
    """Provide the real source-submission service."""
    return submit_source


def get_source_progress_reader() -> SourceProgressReader:
    """Provide a read-only source-progress function."""

    def read_progress(source_id: UUID) -> SourceProgress:
        session = SessionLocal()
        try:
            return get_source_progress(
                session,
                source_id=source_id,
            )
        finally:
            session.close()

    return read_progress

def get_registration_service() -> RegistrationService:
    return register_user


def get_authentication_service() -> AuthenticationService:
    return authenticate_user


def get_token_issuer() -> TokenIssuer:
    return create_access_token