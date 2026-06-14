"""
main.py
-------
FastAPI application entry point for the SignalCare AI dashboard.

Usage:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.config import SECRET_KEY
from app.database import init_db
from app.routers import auth_routes, dashboard

app = FastAPI(title="SignalCare AI")

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(auth_routes.router)
app.include_router(dashboard.router)


@app.on_event("startup")
def on_startup() -> None:
    """Creates database tables if they don't exist yet."""
    init_db()
