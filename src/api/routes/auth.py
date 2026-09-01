from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from src.api.deps import (
    AuthenticationService,
    RegistrationService,
    TokenIssuer,
    get_authentication_service,
    get_registration_service,
    get_token_issuer,
)
from src.schemas.auth import (
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from src.services.auth import (
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    InvalidEmailError,
    InvalidPasswordError,
)


router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: RegisterRequest,
    registration_service: RegistrationService = Depends(
        get_registration_service
    ),
) -> UserResponse:
    try:
        user = registration_service(
            request.email,
            request.password,
        )
    except EmailAlreadyRegisteredError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except (InvalidEmailError, InvalidPasswordError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return UserResponse(
        id=user.id,
        email=user.email,
    )


@router.post(
    "/token",
    response_model=TokenResponse,
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    authentication_service: AuthenticationService = Depends(
        get_authentication_service
    ),
    token_issuer: TokenIssuer = Depends(
        get_token_issuer
    ),
) -> TokenResponse:
    try:
        user = authentication_service(
            form_data.username,
            form_data.password,
        )
    except InvalidCredentialsError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error

    return TokenResponse(
        access_token=token_issuer(user.id),
    )