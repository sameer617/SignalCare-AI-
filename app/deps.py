"""
deps.py
-------
FastAPI dependency functions: database session and current-user resolution.

Usage:
    from app.deps import get_db, get_current_user, require_user
"""

from typing import Generator

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.auth import get_user_id
from app.database import SessionLocal
from app.models import User


# ==========================================
# 1. Database session
# ==========================================

def get_db() -> Generator[Session, None, None]:
    """Yields a SQLAlchemy session, closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================
# 2. Current user
# ==========================================

def get_current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    """
    Resolves the logged-in user from the session cookie, if any.

    Args:
        request: The current request.
        db: Database session.

    Returns:
        The User, or None if not logged in / user no longer exists.
    """
    user_id = get_user_id(request)
    if user_id is None:
        return None
    return db.get(User, user_id)


def require_user(user: User | None = Depends(get_current_user)) -> User:
    """
    Resolves the logged-in user, raising 401 if not logged in.

    Args:
        user: Result of get_current_user.

    Returns:
        The logged-in User.

    Raises:
        HTTPException: 401 if not logged in.
    """
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")
    return user
