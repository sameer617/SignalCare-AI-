"""
database.py
------------
SQLAlchemy engine/session setup for the SignalCare AI dashboard app.

Usage:
    from app.database import Base, SessionLocal, engine, init_db
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DB_PATH

# ==========================================
# 1. Engine + session factory
# ==========================================

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ==========================================
# 2. Init
# ==========================================

def init_db() -> None:
    """Creates all tables that don't exist yet."""
    from app import models  # noqa: F401 (ensure models are registered on Base)

    Base.metadata.create_all(bind=engine)
