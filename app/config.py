"""
config.py
---------
Paths and settings for the SignalCare AI dashboard app.

Reads secrets (OPENAI_API_KEY, SECRET_KEY) from the repo-root .env via
python-dotenv, matching the pattern used in src/risk_taxonomy.py and
src/distress_signal_inference.py.

Usage:
    from app.config import MEDIA_DIR, DB_PATH, SECRET_KEY
"""

import os

from dotenv import load_dotenv

# ==========================================
# 1. Paths
# ==========================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")

MEDIA_DIR = os.path.join(REPO_ROOT, "media")
DB_PATH = os.path.join(REPO_ROOT, "signalcare.db")

load_dotenv(os.path.join(REPO_ROOT, ".env"))


# ==========================================
# 2. Settings
# ==========================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Used to sign session cookies (Starlette SessionMiddleware). Falls back to a
# fixed dev value if not set in .env, but a real value should be added there.
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-insecure-secret-key-change-me")

os.makedirs(MEDIA_DIR, exist_ok=True)
