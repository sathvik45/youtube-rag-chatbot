from collections.abc import Callable
from uuid import UUID

from src.db.repositories.sources import SourceProgress, get_source_progress, list_sources_for_user
from sqlalchemy.orm import Session

from src.db.session import SessionLocal, get_db
from src.services.source_submission import SourceSubmission, submit_source
from src.services.threads import CreatedThread, create_thread_for_source

from src.core.security import create_access_token
from src.services.auth import (
    AuthenticatedUser,
    RegisteredUser,
    authenticate_user,
    register_user,
)

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.core.security import TokenValidationError, get_user_id_from_token
from src.db.repositories.users import get_user_by_id

from datetime import datetime

from src.db.models.source import Source

SourceSubmitter = Callable[[UUID, str], SourceSubmission]
SourceProgressReader = Callable[[UUID, UUID], SourceProgress]
ThreadCreator = Callable[[UUID, UUID, str], CreatedThread]

RegistrationService = Callable[[str, str], RegisteredUser]
AuthenticationService = Callable[[str, str], AuthenticatedUser]
TokenIssuer = Callable[[UUID], str]

DatabaseSessionFactory = Callable[[], Session]

SourceListReader = Callable[
    [UUID, datetime | None, UUID | None, int],
    list[Source],
]

def get_source_submitter() -> SourceSubmitter:
    """Provide the real source-submission service."""
    return submit_source

def get_session_factory() -> DatabaseSessionFactory:
    """Provide a factory for short-lived database sessions."""
    return SessionLocal


def get_thread_creator(
    session_factory: DatabaseSessionFactory = Depends(get_session_factory),
) -> ThreadCreator:
    def create(
        user_id: UUID,
        source_id: UUID,
        title: str,
    ) -> CreatedThread:
        return create_thread_for_source(
            user_id,
            source_id,
            title,
            session_factory=session_factory,
        )

    return create

def get_source_progress_reader(
    session: Session = Depends(get_db),
) -> SourceProgressReader:
    def read_progress(
        user_id: UUID,
        source_id: UUID,
    ) -> SourceProgress:
        return get_source_progress(
            session,
            source_id=source_id,
            user_id=user_id,
        )

    return read_progress

def get_registration_service() -> RegistrationService:
    return register_user


def get_authentication_service() -> AuthenticationService:
    return authenticate_user


def get_token_issuer() -> TokenIssuer:
    return create_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session_factory: DatabaseSessionFactory = Depends(
        get_session_factory
    ),
) -> AuthenticatedUser:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        user_id = get_user_id_from_token(token)
    except TokenValidationError as error:
        raise credentials_error from error

    with session_factory() as session:
        user = get_user_by_id(session, user_id=user_id)

        if user is None or not user.is_active:
            raise credentials_error

        return AuthenticatedUser(id=user.id, email=user.email)

def get_source_list_reader(
    session: Session = Depends(get_db),
) -> SourceListReader:
    def read_sources(
        user_id: UUID,
        before_created_at: datetime | None,
        before_id: UUID | None,
        limit: int,
    ) -> list[Source]:
        return list_sources_for_user(
            session,
            user_id=user_id,
            before_created_at=before_created_at,
            before_id=before_id,
            limit=limit,
        )

    return read_sources
