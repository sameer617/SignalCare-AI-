"""
run_video_pipeline.py
----------------------
Full analysis pipeline for an uploaded session video.

For a single video file, this:
  1. Transcribes the video with timestamps (app.pipeline.transcribe).
  2. Redacts PII from each segment's text (src/preprocess.redact_pii).
  3. Runs the v1 sentiment MLP on each segment (src/run_temporal_on_transcript).
  4. Classifies distress signals S01-S18 for each segment
     (src/distress_signal_inference.classify_utterances).
  5. Feeds both sequences through a fresh SessionAnalyzer
     (src/session_analysis) to get S05/S08/S09/S12 per turn.
  6. Writes one JSON record per segment to analysis.json, including the
     segment's start/end time so the dashboard can replay the analysis in
     sync with video playback.

Known limitation: Whisper does not perform speaker diarization, so every
transcript segment is currently treated as a patient turn. True
patient/therapist separation is deferred (see CLAUDE.md Section 2).

This is decision-support only (CLAUDE.md Rule 5): outputs are signals +
confidences for a human counselor to review, never a diagnosis or crisis
verdict.

Usage:
    from app.pipeline.run_video_pipeline import run_pipeline

    run_pipeline(video_path="media/1/3/original.mp4",
                  output_path="media/1/3/analysis.json")
"""

import json
import sys

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from app.config import SRC_DIR
from app.pipeline.transcribe import transcribe_video

sys.path.insert(0, SRC_DIR)
from distress_signal_inference import classify_utterances, DistressSignalResult  # noqa: E402
from preprocess import redact_pii  # noqa: E402
from run_temporal_on_transcript import (  # noqa: E402
    build_embedding_model,
    load_sentiment_mlp,
    predict_sentiment_turns,
)
from session_analysis import DistressTurn, SessionAnalyzer  # noqa: E402


DISTRESS_BATCH_SIZE = 12


# ==========================================
# 1. Pipeline
# ==========================================

def run_pipeline(video_path: str, output_path: str) -> None:
    """
    Runs the full analysis pipeline for a single video and writes the
    per-turn results to output_path as JSON.

    Args:
        video_path: Path to the uploaded video file.
        output_path: Path to write analysis.json.
    """
    segments = transcribe_video(video_path)
    if not segments:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return

    analyzer_engine = AnalyzerEngine()
    anonymizer_engine = AnonymizerEngine()
    redacted_texts = [
        redact_pii(seg.text, analyzer_engine, anonymizer_engine) for seg in segments
    ]

    embedder = build_embedding_model()
    mlp = load_sentiment_mlp()
    sentiment_turns = predict_sentiment_turns(embedder, mlp, redacted_texts)

    distress_results: list[DistressSignalResult] = classify_utterances(
        redacted_texts, batch_size=DISTRESS_BATCH_SIZE
    )
    distress_turns = [DistressTurn(signals=r.signals) for r in distress_results]

    analyzer = SessionAnalyzer()
    records = []

    for seg, text, sent_turn, dist_turn, dist_result in zip(
        segments, redacted_texts, sentiment_turns, distress_turns, distress_results
    ):
        signals = analyzer.update(sent_turn, dist_turn)

        records.append({
            "turn_id": len(records) + 1,
            "start_time": seg.start,
            "end_time": seg.end,
            "text": text,
            "sentiment": sent_turn.sentiment,
            "sentiment_confidence": sent_turn.confidence,
            "distress_signals": [s.value for s in dist_turn.signals],
            "distress_signal_confidence": dist_result.signal_confidence,
            "drift_score": signals.drift_score,
            "drift_fired": signals.drift_fired,
            "drift_confidence": signals.drift_confidence,
            "volatility_score": signals.volatility_score,
            "volatility_fired": signals.volatility_fired,
            "volatility_confidence": signals.volatility_confidence,
            "rumination_fired": signals.rumination_fired,
            "rumination_confidence": signals.rumination_confidence,
            "rumination_codes": [s.value for s in signals.rumination_codes],
            "escalation_fired": signals.escalation_fired,
            "escalation_confidence": signals.escalation_confidence,
            "escalation_codes": [s.value for s in signals.escalation_codes],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
