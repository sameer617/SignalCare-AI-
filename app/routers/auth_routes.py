"""
auth_routes.py
--------------
Registration, login, and logout routes for the SignalCare AI dashboard app.

Usage:
    Mounted in app/main.py via app.include_router(auth_routes.router).
"""

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth import hash_password, login_user, logout_user, verify_password
from app.deps import get_db
from app.models import User

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


# ==========================================
# 1. Register
# ==========================================

@router.get("/register", response_class=HTMLResponse)
def register_form(request: Request) -> HTMLResponse:
    """Renders the registration form."""
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@router.post("/register", response_class=HTMLResponse)
def register_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Creates a new user account and logs them in."""
    existing = db.query(User).filter(User.email == email).first()
    if existing is not None:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "An account with that email already exists."},
            status_code=400,
        )

    user = User(email=email, hashed_password=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)

    login_user(request, user.id)
    return RedirectResponse(url="/", status_code=303)


# ==========================================
# 2. Login / Logout
# ==========================================

@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request) -> HTMLResponse:
    """Renders the login form."""
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@router.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Verifies credentials and logs the user in."""
    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid email or password."},
            status_code=401,
        )

    login_user(request, user.id)
    return RedirectResponse(url="/", status_code=303)


@router.get("/logout")
def logout(request: Request) -> RedirectResponse:
    """Clears the session and redirects to the login page."""
    logout_user(request)
    return RedirectResponse(url="/login", status_code=303)
