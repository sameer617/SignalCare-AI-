"""
transcribe.py
-------------
Audio extraction + timestamped transcription for uploaded session videos.

Extracts the audio track from a video file with ffmpeg, then transcribes it
with the OpenAI Whisper API (response_format="verbose_json") to get
segment-level timestamps.

Known limitation: Whisper's API does not perform speaker diarization. v1
treats every transcript segment as a patient turn (see
run_video_pipeline.py docstring) -- true speaker separation is deferred.

Usage:
    from app.pipeline.transcribe import transcribe_video

    segments = transcribe_video("media/1/3/original.mp4")
    # segments -> [TranscriptSegment(start=0.0, end=4.2, text="..."), ...]
"""

import os
import tempfile

import ffmpeg
from openai import OpenAI
from pydantic import BaseModel

from app.config import OPENAI_API_KEY


# ==========================================
# 1. Schema
# ==========================================

class TranscriptSegment(BaseModel):
    """A single timestamped segment from Whisper."""
    start: float
    end: float
    text: str


# ==========================================
# 2. Audio extraction
# ==========================================

def extract_audio(video_path: str, audio_path: str) -> None:
    """
    Extracts the audio track from a video file as a mono 16kHz WAV file.

    Args:
        video_path: Path to the input video file.
        audio_path: Path to write the output WAV file.
    """
    (
        ffmpeg.input(video_path)
        .output(audio_path, ac=1, ar="16000", format="wav")
        .overwrite_output()
        .run(quiet=True)
    )


# ==========================================
# 3. Transcription
# ==========================================

def transcribe_video(video_path: str) -> list[TranscriptSegment]:
    """
    Extracts audio from a video and transcribes it with timestamps.

    Args:
        video_path: Path to the input video file.

    Returns:
        List of TranscriptSegment, in chronological order.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = os.path.join(tmp_dir, "audio.wav")
        extract_audio(video_path, audio_path)

        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
            )

    return [
        TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
        for seg in response.segments
        if seg.text.strip()
    ]
