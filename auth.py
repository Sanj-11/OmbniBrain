from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from datetime import timedelta # Keep timedelta for access_token_expires calculation
from typing import Optional

import crud
import models
import schemas
from database import get_db
from config import settings # Keep settings for SECRET_KEY in get_current_user
import security # New import for password hashing/token creation


router = APIRouter(tags=["Authentication"])


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if token is None:
        raise credentials_exception

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=["HS256"]
        )

        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception

        token_data = schemas.TokenData(username=username)

    except JWTError:
        raise credentials_exception

    user = crud.get_user_by_username(
        db,
        username=token_data.username
    )

    if user is None:
        raise credentials_exception

    return user


@router.post("/register", response_model=schemas.User)
def register_user(
    user: schemas.UserCreate,
    db: Session = Depends(get_db)
):
    db_user = crud.get_user_by_username(
        db,
        username=user.username
    )

    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )

    # The password hashing is now handled by security.get_password_hash in crud.py
    return crud.create_user(db=db, user_data=user.model_dump())


@router.post("/token", response_model=schemas.Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = crud.get_user_by_username(
        db,
        username=form_data.username
    )

    if not user or not security.verify_password( # Use security.verify_password
        form_data.password,
        user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}, # Corrected indentation
        )

    access_token_expires = timedelta(
        minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES # Use security.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    access_token = security.create_access_token( # Use security.create_access_token
        data={"sub": user.username},
        expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.get("/users/me", response_model=schemas.User)
async def read_users_me(
    current_user: models.User = Depends(get_current_user)
):
    return current_user