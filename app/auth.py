"""
auth.py
-------
Password hashing and session-cookie helpers for the SignalCare AI dashboard app.

Sessions are stored via Starlette's signed-cookie SessionMiddleware (configured
in app/main.py with app.config.SECRET_KEY) -- the cookie holds only the
logged-in user's id, never the password.

Usage:
    from app.auth import hash_password, verify_password, login_user, logout_user, get_user_id
"""

from passlib.context import CryptContext
from starlette.requests import Request

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ==========================================
# 1. Password hashing
# ==========================================

def hash_password(password: str) -> str:
    """
    Hashes a plaintext password for storage.

    Args:
        password: Plaintext password.

    Returns:
        Bcrypt hash string.
    """
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Checks a plaintext password against a stored hash.

    Args:
        password: Plaintext password to check.
        hashed_password: Stored bcrypt hash.

    Returns:
        True if the password matches.
    """
    return pwd_context.verify(password, hashed_password)


# ==========================================
# 2. Session helpers
# ==========================================

def login_user(request: Request, user_id: int) -> None:
    """Stores the logged-in user's id in the signed session cookie."""
    request.session["user_id"] = user_id


def logout_user(request: Request) -> None:
    """Clears the session cookie."""
    request.session.clear()


def get_user_id(request: Request) -> int | None:
    """
    Reads the logged-in user's id from the session cookie, if present.

    Args:
        request: The current request.

    Returns:
        The user id, or None if not logged in.
    """
    return request.session.get("user_id")
