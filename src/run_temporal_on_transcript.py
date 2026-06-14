"""
run_temporal_on_transcript.py
------------------------------
Week 3: Validates session_analysis.py (S05/S08/S09/S12) against a real HOPE
transcript.

Pipeline for a single transcript:
  1. Load patient turns from processed/<SPLIT>/utterances.jsonl.
  2. Encode each turn's text with bert-base-uncased + mean pooling (v1
     embeddings, matching train_sentiment_v1.py / train_sentiment_mlp.py).
  3. Run the final sentiment MLP (models/sentiment-v1-mlp-d05-wd3/) to get a
     sentiment label + confidence per turn -> SentimentTurn (S05/S08 input).
  4. Load the matching turns' distress signal labels from
     processed/<SPLIT>/utterance_labels.jsonl (LLM-generated via label.py,
     same approach as distress_signal_inference.py) -> DistressTurn
     (S09/S12 input).
  5. Feed both sequences through SessionAnalyzer and print a per-turn table.

This is a one-off validation script (Week 3), not part of the production
pipeline — session-level aggregation will supersede it.

Usage:
  python src/run_temporal_on_transcript.py Transcript_13 --split Validation
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Transformer, Pooling

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_taxonomy import RiskSignal
from session_analysis import SessionAnalyzer, DistressTurn
from temporal_analysis import SentimentTurn
from train_sentiment_mlp import SentimentMLP, SENTIMENT_LABELS, HIDDEN_DIMS, DROPOUT


# ==========================================
# 1. Constants
# ==========================================

REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "processed")
MODEL_DIR     = os.path.join(REPO_ROOT, "models", "sentiment-v1-mlp-d05-wd3")
BASE_MODEL    = "bert-base-uncased"
EMBEDDING_DIM = 768


# ==========================================
# 2. Loading
# ==========================================

def load_transcript_turns(split: str, transcript_id: str) -> List[Tuple[int, str]]:
    """
    Loads a single transcript's patient turns from utterances.jsonl.

    Args:
        split: One of 'Train', 'Validation', 'Test'.
        transcript_id: e.g. 'Transcript_13'.

    Returns:
        List of (patient_turn_id, text) tuples, in file order.
    """
    path = os.path.join(PROCESSED_DIR, split, "utterances.jsonl")
    turns: List[Tuple[int, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            if record["transcript_id"] == transcript_id:
                turns.append((record["patient_turn_id"], record["text"]))

    return turns


def load_distress_turns(split: str, transcript_id: str) -> Dict[int, DistressTurn]:
    """
    Loads a single transcript's distress signal labels from
    utterance_labels.jsonl (LLM-generated, see label.py and CLAUDE.md Rule 6).

    Args:
        split: One of 'Train', 'Validation', 'Test'.
        transcript_id: e.g. 'Transcript_13'.

    Returns:
        Dict mapping patient_turn_id -> DistressTurn.
    """
    path = os.path.join(PROCESSED_DIR, split, "utterance_labels.jsonl")
    by_turn_id: Dict[int, DistressTurn] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            if record["transcript_id"] == transcript_id:
                signals = [RiskSignal(s) for s in record.get("distress_signals", [])]
                by_turn_id[record["patient_turn_id"]] = DistressTurn(signals=signals)

    return by_turn_id


def build_embedding_model() -> SentenceTransformer:
    """
    Constructs the v1 (raw bert-base-uncased + mean pooling) embedding model,
    matching train_sentiment_v1.py.

    Returns:
        SentenceTransformer ready for inference.
    """
    word_embedding_model = Transformer(BASE_MODEL)
    pooling_model = Pooling(
        word_embedding_model.get_embedding_dimension(),
        pooling_mode="mean",
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def load_sentiment_mlp() -> SentimentMLP:
    """
    Loads the finalized sentiment classifier head (v1 + MLP[128], dropout=0.5,
    weight_decay=1e-3) from models/sentiment-v1-mlp-d05-wd3/.

    Returns:
        SentimentMLP in eval mode.
    """
    model = SentimentMLP(
        input_dim=EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        num_classes=len(SENTIMENT_LABELS),
        dropout=DROPOUT,
    )
    state_dict = torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ==========================================
# 3. Inference
# ==========================================

def predict_sentiment_turns(
    embedder: SentenceTransformer, mlp: SentimentMLP, texts: List[str]
) -> List[SentimentTurn]:
    """
    Encodes texts and runs the sentiment MLP to produce SentimentTurn records.

    Args:
        embedder: v1 embedding model.
        mlp: Trained sentiment MLP classifier head.
        texts: Patient turn texts, in chronological order.

    Returns:
        List of SentimentTurn, one per input text, in order.
    """
    embeddings = embedder.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True)

    with torch.no_grad():
        logits = mlp(torch.tensor(embeddings, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)
        confidences, pred_idx = probs.max(dim=1)

    return [
        SentimentTurn(sentiment=SENTIMENT_LABELS[idx.item()], confidence=conf.item())
        for idx, conf in zip(pred_idx, confidences)
    ]


# ==========================================
# 4. CLI
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run session_analysis.py on a real HOPE transcript.")
    parser.add_argument("transcript_id", type=str, help="e.g. Transcript_13")
    parser.add_argument("--split", type=str, default="Validation", choices=["Train", "Validation", "Test"])
    parser.add_argument("--window-size", type=int, default=None, help="Override SessionAnalyzer window size.")
    args = parser.parse_args()

    print(f"Loading {args.split}/{args.transcript_id} patient turns...")
    turns = load_transcript_turns(args.split, args.transcript_id)
    print(f"  {len(turns)} patient turns found.")
    if not turns:
        print("No turns found for this transcript_id. Exiting.")
        sys.exit(1)

    turn_ids, texts = zip(*turns)

    print(f"Loading distress signal labels (utterance_labels.jsonl)...")
    distress_by_turn_id = load_distress_turns(args.split, args.transcript_id)

    print(f"\nLoading {BASE_MODEL} embedder (v1, no domain adaptation)...")
    embedder = build_embedding_model()

    print("Loading sentiment MLP (models/sentiment-v1-mlp-d05-wd3/)...")
    mlp = load_sentiment_mlp()

    print("\nEncoding + classifying patient turns...")
    sentiment_turns = predict_sentiment_turns(embedder, mlp, list(texts))
    distress_turns = [distress_by_turn_id.get(tid, DistressTurn(signals=[])) for tid in turn_ids]

    print("\nRunning SessionAnalyzer over the session...")
    analyzer_kwargs = {} if args.window_size is None else {"window_size": args.window_size}
    analyzer = SessionAnalyzer(**analyzer_kwargs)

    print(f"\n=== {args.split}/{args.transcript_id} ({len(turns)} patient turns) ===")
    print(f"{'turn_id':>7} {'sentiment':>9} {'conf':>5} {'score':>6} {'S08':>5} {'S05':>5} "
          f"{'S09':>5} {'S12':>5}  {'distress codes'}")

    fired_counts = {"S05": 0, "S08": 0, "S09": 0, "S12": 0}

    for turn_id, sent_turn, dist_turn in zip(turn_ids, sentiment_turns, distress_turns):
        signals = analyzer.update(sent_turn, dist_turn)

        if signals.drift_fired:
            fired_counts["S08"] += 1
        if signals.volatility_fired:
            fired_counts["S05"] += 1
        if signals.rumination_fired:
            fired_counts["S09"] += 1
        if signals.escalation_fired:
            fired_counts["S12"] += 1

        codes = ", ".join(s.value.split("_")[0] for s in dist_turn.signals) or "-"
        print(
            f"{turn_id:>7} {sent_turn.sentiment:>9} {sent_turn.confidence:>5.2f} "
            f"{sent_turn.score():>+6.2f} "
            f"{'YES' if signals.drift_fired else '.':>5} {'YES' if signals.volatility_fired else '.':>5} "
            f"{'YES' if signals.rumination_fired else '.':>5} {'YES' if signals.escalation_fired else '.':>5}  "
            f"{codes}"
        )

    print(f"\nS08 (negative sentiment trend) fired on {fired_counts['S08']}/{len(turns)} turns.")
    print(f"S05 (emotional volatility)     fired on {fired_counts['S05']}/{len(turns)} turns.")
    print(f"S09 (rumination patterns)      fired on {fired_counts['S09']}/{len(turns)} turns.")
    print(f"S12 (escalation composite)     fired on {fired_counts['S12']}/{len(turns)} turns.")
