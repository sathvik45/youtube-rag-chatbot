import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.deps import (
    get_authentication_service,
    get_registration_service,
    get_token_issuer,
)
from src.main import app
from src.services.auth import (
    AuthenticatedUser,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    RegisteredUser,
)


@pytest.fixture
def client():
    app.dependency_overrides.clear()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_register_returns_public_user(client) -> None:
    registered_user = RegisteredUser(
        id=uuid.uuid4(),
        email="learner@example.invalid",
    )

    received_credentials: list[tuple[str, str]] = []

    def fake_register(
        email: str,
        password: str,
    ) -> RegisteredUser:
        received_credentials.append((email, password))
        return registered_user

    app.dependency_overrides[get_registration_service] = (
        lambda: fake_register
    )

    response = client.post(
        "/auth/register",
        json={
            "email": "learner@example.invalid",
            "password": "a-long-enough-password",
        },
    )

    assert response.status_code == 201
    assert response.json() == {
        "id": str(registered_user.id),
        "email": registered_user.email,
    }
    assert received_credentials == [
        (
            "learner@example.invalid",
            "a-long-enough-password",
        )
    ]


def test_register_returns_conflict_for_duplicate_email(client) -> None:
    def fake_register(
        _: str,
        __: str,
    ) -> RegisteredUser:
        raise EmailAlreadyRegisteredError(
            "An account with this email already exists."
        )

    app.dependency_overrides[get_registration_service] = (
        lambda: fake_register
    )

    response = client.post(
        "/auth/register",
        json={
            "email": "learner@example.invalid",
            "password": "a-long-enough-password",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "An account with this email already exists."
    )


def test_token_returns_bearer_token_for_valid_credentials(client) -> None:
    authenticated_user = AuthenticatedUser(
        id=uuid.uuid4(),
        email="learner@example.invalid",
    )

    def fake_authenticate(
        email: str,
        password: str,
    ) -> AuthenticatedUser:
        assert email == "learner@example.invalid"
        assert password == "a-long-enough-password"
        return authenticated_user

    def fake_issue_token(user_id: uuid.UUID) -> str:
        assert user_id == authenticated_user.id
        return "fake-access-token"

    app.dependency_overrides[get_authentication_service] = (
        lambda: fake_authenticate
    )
    app.dependency_overrides[get_token_issuer] = (
        lambda: fake_issue_token
    )

    response = client.post(
        "/auth/token",
        data={
            "username": "learner@example.invalid",
            "password": "a-long-enough-password",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "access_token": "fake-access-token",
        "token_type": "bearer",
    }


def test_token_rejects_invalid_credentials(client) -> None:
    def fake_authenticate(
        _: str,
        __: str,
    ) -> AuthenticatedUser:
        raise InvalidCredentialsError(
            "Incorrect email or password."
        )

    app.dependency_overrides[get_authentication_service] = (
        lambda: fake_authenticate
    )

    response = client.post(
        "/auth/token",
        data={
            "username": "learner@example.invalid",
            "password": "wrong-password",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password."
    assert response.headers["www-authenticate"] == "Bearer"