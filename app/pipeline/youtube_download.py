"""
youtube_download.py
--------------------
Downloads a YouTube video to a session's media directory so it can be
processed by the same analysis pipeline as a directly uploaded file
(app.pipeline.run_video_pipeline).

Usage:
    from app.pipeline.youtube_download import download_youtube_video

    filename = download_youtube_video("https://youtu.be/...", session_dir)
    # filename -> "original.mp4"
"""

import os

import yt_dlp


# ==========================================
# 1. Download
# ==========================================

def download_youtube_video(url: str, session_dir: str) -> str:
    """
    Downloads a YouTube video as an mp4 file named "original.mp4" inside
    session_dir, merging separate audio/video streams with ffmpeg if needed.

    Args:
        url: The YouTube video URL.
        session_dir: Directory to download the video into (must exist).

    Returns:
        The filename ("original.mp4") of the downloaded video, relative to
        session_dir.

    Raises:
        yt_dlp.utils.DownloadError: If the video cannot be downloaded.
    """
    output_template = os.path.join(session_dir, "original.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return "original.mp4"
