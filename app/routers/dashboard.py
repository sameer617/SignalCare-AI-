"""
dashboard.py
------------
Landing page, video upload, and session view routes for the SignalCare AI
dashboard app.

Usage:
    Mounted in app/main.py via app.include_router(dashboard.router).
"""

import json
import os
import shutil

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import MEDIA_DIR, REPO_ROOT
from app.database import SessionLocal
from app.deps import get_current_user, get_db, require_user
from app.models import TherapySession, User
from app.pipeline.run_video_pipeline import run_pipeline
from app.pipeline.youtube_download import download_youtube_video

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

TAXONOMY_PATH = os.path.join(REPO_ROOT, "taxonomy.json")


# ==========================================
# 1. Landing page / session list
# ==========================================

@router.get("/", response_class=HTMLResponse)
def index(request: Request, user: User | None = Depends(get_current_user), db: Session = Depends(get_db)) -> HTMLResponse:
    """Shows the logged-in user's sessions and an upload form, or redirects to login."""
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    sessions = (
        db.query(TherapySession)
        .filter(TherapySession.user_id == user.id)
        .order_by(TherapySession.created_at.desc())
        .all()
    )
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "user": user, "sessions": sessions}
    )


# ==========================================
# 2. Upload
# ==========================================

@router.post("/upload")
def upload_video(
    title: str = Form(...),
    video: UploadFile = None,
    youtube_url: str = Form(""),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """Saves an uploaded video (file or YouTube link) and creates a session
    row with status='uploaded'."""
    youtube_url = youtube_url.strip()
    has_file = video is not None and bool(video.filename)

    if not has_file and not youtube_url:
        raise HTTPException(status_code=400, detail="Provide a video file or a YouTube link.")
    if has_file and youtube_url:
        raise HTTPException(status_code=400, detail="Provide either a video file or a YouTube link, not both.")

    session = TherapySession(
        user_id=user.id,
        title=title,
        video_filename="",
        status="uploaded",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    session_dir = os.path.join(MEDIA_DIR, str(user.id), str(session.id))
    os.makedirs(session_dir, exist_ok=True)

    if has_file:
        ext = os.path.splitext(video.filename)[1] or ".mp4"
        video_filename = f"original{ext}"
        dest_path = os.path.join(session_dir, video_filename)
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(video.file, out_file)
    else:
        try:
            video_filename = download_youtube_video(youtube_url, session_dir)
        except Exception as exc:  # pylint: disable=broad-except
            db.delete(session)
            db.commit()
            raise HTTPException(status_code=400, detail=f"Could not download YouTube video: {exc}") from exc

    session.video_filename = video_filename
    db.commit()

    return RedirectResponse(url=f"/sessions/{session.id}", status_code=303)


# ==========================================
# 3. Session view
# ==========================================

def _get_owned_session(session_id: int, user: User, db: Session) -> TherapySession:
    """Fetches a session by id, raising 404 if missing or not owned by user."""
    session = db.get(TherapySession, session_id)
    if session is None or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def _load_taxonomy() -> dict:
    """Loads taxonomy.json and returns a dict mapping signal code -> {name, definition}."""
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        sig["code"]: {"name": sig["name"], "definition": sig["definition"]}
        for sig in data["signals"]
    }


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
def session_view(
    session_id: int,
    request: Request,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Shows the video player + analysis dashboard for a session."""
    session = _get_owned_session(session_id, user, db)
    taxonomy = _load_taxonomy()

    return templates.TemplateResponse(
        "session.html",
        {
            "request": request,
            "user": user,
            "session": session,
            "taxonomy": taxonomy,
        },
    )


# ==========================================
# 4. Start analysis (background pipeline)
# ==========================================

def _run_analysis_job(session_id: int, video_path: str, output_path: str) -> None:
    """Runs the analysis pipeline and updates the session's status (background task)."""
    db = SessionLocal()
    try:
        session = db.get(TherapySession, session_id)
        try:
            run_pipeline(video_path=video_path, output_path=output_path)
            session.status = "ready"
        except Exception as exc:  # pylint: disable=broad-except
            session.status = "failed"
            session.error_message = str(exc)
        db.commit()
    finally:
        db.close()


@router.post("/sessions/{session_id}/start-analysis")
def start_analysis(
    session_id: int,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """Kicks off the analysis pipeline for a session's video in the background."""
    session = _get_owned_session(session_id, user, db)

    if session.status == "uploaded":
        session.status = "processing"
        db.commit()

        session_dir = os.path.join(MEDIA_DIR, str(user.id), str(session.id))
        video_path = os.path.join(session_dir, session.video_filename)
        output_path = os.path.join(session_dir, "analysis.json")

        background_tasks.add_task(_run_analysis_job, session.id, video_path, output_path)

    return RedirectResponse(url=f"/sessions/{session.id}", status_code=303)


# ==========================================
# 5. Status + analysis JSON endpoints
# ==========================================

@router.get("/sessions/{session_id}/status")
def session_status(
    session_id: int,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Returns the session's current processing status (for polling)."""
    session = _get_owned_session(session_id, user, db)
    return JSONResponse({"status": session.status, "error_message": session.error_message})


@router.get("/sessions/{session_id}/analysis.json")
def session_analysis_json(
    session_id: int,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Returns the precomputed per-turn analysis for a ready session."""
    session = _get_owned_session(session_id, user, db)
    if session.status != "ready":
        raise HTTPException(status_code=409, detail="Analysis is not ready yet.")

    analysis_path = os.path.join(MEDIA_DIR, str(user.id), str(session.id), "analysis.json")
    return FileResponse(analysis_path, media_type="application/json")


@router.get("/sessions/{session_id}/video")
def session_video(
    session_id: int,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Serves the uploaded video file for playback."""
    session = _get_owned_session(session_id, user, db)
    video_path = os.path.join(MEDIA_DIR, str(user.id), str(session.id), session.video_filename)
    return FileResponse(video_path)
