"""JWT-based authentication for the enterprise RAG API."""

from __future__ import annotations

from datetime import datetime, timedelta

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.core.config import get_settings

_bearer_scheme = HTTPBearer(auto_error=False)


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


# Demo users (replace with database in production)
_DEMO_USERS: dict[str, dict] = {
    "admin": {
        "hashed_password": _hash_password("admin"),
        "role": "admin",
    },
    "viewer": {
        "hashed_password": _hash_password("viewer"),
        "role": "viewer",
    },
}


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def authenticate_user(username: str, password: str) -> dict | None:
    user = _DEMO_USERS.get(username)
    if user and verify_password(password, user["hashed_password"]):
        return {"username": username, **user}
    return None


def create_access_token(data: dict) -> str:
    settings = get_settings().auth
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {**data, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> dict:
    settings = get_settings().auth
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict | None:
    """Dependency that enforces auth when enabled, otherwise returns None."""
    settings = get_settings().auth
    if not settings.enabled:
        return None

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        payload = decode_token(credentials.credentials)
        username: str | None = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return {"username": username, "role": payload.get("role", "viewer")}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
