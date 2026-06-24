"""
run_session_pipeline.py
------------------------
Week 3: Per-split session aggregation pipeline.

For every transcript in a split, this script:
  1. Loads patient turns from processed/<SPLIT>/utterances.jsonl.
  2. Encodes each turn's text with TWO embedders (v1 raw bert-base-uncased +
     v2 TSDAE domain-adapted bert-base-uncased, concatenated to 1536-dim) and
     runs the final sentiment MLP (models/sentiment-v4-ensemble/) to get a
     SentimentTurn per turn (S05/S08 input) -- reuses build_embedding_model /
     build_tsdae_embedding_model / load_sentiment_mlp / predict_sentiment_turns
     from run_temporal_on_transcript.py.
  3. Calls distress_signal_inference.classify_utterances() to get a
     DistressTurn per turn (S09/S12 input). Unlike
     run_temporal_on_transcript.py (which reads utterance_labels.jsonl for
     one-off validation), this is the production path: per CLAUDE.md Rule 6,
     utterance_labels.jsonl is LLM-labeled reference data, not a production
     dependency.
  4. Feeds both sequences through SessionAnalyzer.update() per turn.
  5. Writes one JSON object per patient turn to
     processed/<SPLIT>/session_signals.jsonl.

This is decision-support only (CLAUDE.md Rule 5): outputs are signals +
confidences for a human counselor to review, never a diagnosis or crisis
verdict.

Usage:
    python src/run_session_pipeline.py --split Validation
    python src/run_session_pipeline.py --split Validation --limit 2
"""

import os
import sys
import json
import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distress_signal_inference import classify_utterances, DistressSignalResult
from session_analysis import SessionAnalyzer, DistressTurn
from run_temporal_on_transcript import (
    build_embedding_model,
    build_tsdae_embedding_model,
    load_sentiment_mlp,
    predict_sentiment_turns,
)


# ==========================================
# 1. Constants
# ==========================================

REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "processed")

DISTRESS_BATCH_SIZE = 12


# ==========================================
# 2. Loading
# ==========================================

def load_split_turns(split: str) -> "OrderedDict[str, List[Tuple[int, str]]]":
    """
    Loads all patient turns for a split, grouped by transcript.

    Args:
        split: One of 'Train', 'Validation', 'Test'.

    Returns:
        Ordered dict mapping transcript_id -> list of (patient_turn_id, text)
        tuples, in file order. Transcript order matches first appearance in
        utterances.jsonl.
    """
    path = os.path.join(PROCESSED_DIR, split, "utterances.jsonl")
    by_transcript: "OrderedDict[str, List[Tuple[int, str]]]" = OrderedDict()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            transcript_id = record["transcript_id"]
            by_transcript.setdefault(transcript_id, []).append(
                (record["patient_turn_id"], record["text"])
            )

    return by_transcript


# ==========================================
# 3. Pipeline
# ==========================================

def run_pipeline(split: str, limit: int = None) -> None:
    """
    Runs the full session aggregation pipeline for a split and writes
    processed/<SPLIT>/session_signals.jsonl.

    Args:
        split: One of 'Train', 'Validation', 'Test'.
        limit: If set, only process the first `limit` transcripts (for
            quick testing).
    """
    print(f"Loading {split} patient turns from utterances.jsonl...")
    by_transcript = load_split_turns(split)
    transcript_ids = list(by_transcript.keys())
    if limit is not None:
        transcript_ids = transcript_ids[:limit]
    print(f"  {len(transcript_ids)} transcript(s) to process.")

    print(f"\nLoading {build_embedding_model.__module__}'s bert-base-uncased embedder (v1)...")
    embedder = build_embedding_model()

    print("Loading TSDAE domain-adapted embedder (v2)...")
    tsdae_embedder = build_tsdae_embedding_model()

    print("Loading sentiment MLP (models/sentiment-v4-ensemble/)...")
    mlp = load_sentiment_mlp()

    out_path = os.path.join(PROCESSED_DIR, split, "session_signals.jsonl")
    fired_counts = {"S05": 0, "S08": 0, "S09": 0, "S12": 0}
    total_turns = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for transcript_id in tqdm(transcript_ids, desc=f"{split} transcripts"):
            turns = by_transcript[transcript_id]
            turn_ids, texts = zip(*turns)
            texts = list(texts)

            sentiment_turns = predict_sentiment_turns(embedder, mlp, texts, tsdae_embedder)
            distress_results: List[DistressSignalResult] = classify_utterances(
                texts, batch_size=DISTRESS_BATCH_SIZE
            )
            distress_turns = [
                DistressTurn(signals=r.signals) for r in distress_results
            ]

            analyzer = SessionAnalyzer()
            for turn_id, sent_turn, dist_turn, dist_result in zip(
                turn_ids, sentiment_turns, distress_turns, distress_results
            ):
                signals = analyzer.update(sent_turn, dist_turn)
                total_turns += 1

                if signals.drift_fired:
                    fired_counts["S08"] += 1
                if signals.volatility_fired:
                    fired_counts["S05"] += 1
                if signals.rumination_fired:
                    fired_counts["S09"] += 1
                if signals.escalation_fired:
                    fired_counts["S12"] += 1

                record: Dict = {
                    "transcript_id": transcript_id,
                    "patient_turn_id": turn_id,
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
                }
                out_f.write(json.dumps(record) + "\n")

    print(f"\nWrote {total_turns} turn-level records to {out_path}")
    for code, count in fired_counts.items():
        pct = 100 * count / total_turns if total_turns else 0.0
        print(f"  {code} fired on {count}/{total_turns} turns ({pct:.1f}%)")


# ==========================================
# 4. CLI
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the session aggregation pipeline (S05/S08/S09/S12) over every transcript in a split."
    )
    parser.add_argument("--split", type=str, default="Validation", choices=["Train", "Validation", "Test"])
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N transcripts (for testing).")
    args = parser.parse_args()

    run_pipeline(args.split, limit=args.limit)
