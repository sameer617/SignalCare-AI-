"""
youtube_download.py
--------------------
Downloads a YouTube video to a session's media directory so it can be
processed by the same analysis pipeline as a directly uploaded file
(app.pipeline.run_video_pipeline).

Uses pytubefix to download YouTube videos — this avoids the bot-check /
sign-in errors that yt-dlp encounters on cloud/datacenter IPs (AWS EC2 etc).
Audio and video streams are merged with ffmpeg when downloaded separately.

Usage:
    from app.pipeline.youtube_download import download_youtube_video

    filename = download_youtube_video("https://youtu.be/...", session_dir)
    # filename -> "original.mp4"
"""

import os
import subprocess

from pytubefix import YouTube


# ==========================================
# 1. Download
# ==========================================

def download_youtube_video(url: str, session_dir: str) -> str:
    """
    Downloads a YouTube video as original.mp4 inside session_dir.
    Tries the highest-resolution progressive stream first (video+audio in one
    file). If none is available, downloads the best video-only and audio-only
    streams separately and merges them with ffmpeg.

    Args:
        url: The YouTube video URL.
        session_dir: Directory to download the video into (must exist).

    Returns:
        The filename ("original.mp4") of the downloaded video, relative to
        session_dir.

    Raises:
        Exception: If the video cannot be downloaded or merged.
    """
    yt = YouTube(url)
    output_path = os.path.join(session_dir, "original.mp4")

    # Try progressive stream first (video + audio combined, no ffmpeg needed)
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").last()

    if stream:
        stream.download(output_path=session_dir, filename="original.mp4")
        return "original.mp4"

    # Fall back to separate video + audio streams merged with ffmpeg
    video_stream = yt.streams.filter(adaptive=True, file_extension="mp4", only_video=True).order_by("resolution").last()
    audio_stream = yt.streams.filter(adaptive=True, file_extension="mp4", only_audio=True).order_by("abr").last()

    if not video_stream or not audio_stream:
        raise RuntimeError("No suitable YouTube streams found for this video.")

    video_path = os.path.join(session_dir, "_video.mp4")
    audio_path = os.path.join(session_dir, "_audio.mp4")

    video_stream.download(output_path=session_dir, filename="_video.mp4")
    audio_stream.download(output_path=session_dir, filename="_audio.mp4")

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c", "copy", output_path],
        check=True,
        capture_output=True,
    )

    os.remove(video_path)
    os.remove(audio_path)

    return "original.mp4"
