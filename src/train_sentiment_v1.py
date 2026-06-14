"""
train_sentiment_v1.py
----------------------
Week 2, v1: utterance-level sentiment classifier on raw (non-domain-adapted)
bert-base-uncased embeddings.

Pipeline:
  1. Load processed/<SPLIT>/utterance_sentiment.jsonl (LLM-generated sentiment
     labels, see sentiment_labeling.py and CLAUDE.md Section 7 Rule 6 — these
     are not gold labels).
  2. Encode utterance text with bert-base-uncased + mean pooling, via the same
     SentenceTransformer construction used in domain_adaptation.py, so v1 and
     v2 (TSDAE-adapted) embeddings are directly comparable.
  3. Train a class-weighted logistic regression head on Train embeddings.
  4. Evaluate on Validation (target: >=80% accuracy) and Test.
  5. Save the classifier head and a metrics report.

Output:
  models/sentiment-v1/classifier.joblib
  models/sentiment-v1/metrics.json

Usage:
  python src/train_sentiment_v1.py
  python src/train_sentiment_v1.py --batch-size 16
"""

import os
import json
import argparse
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Transformer, Pooling


# ==========================================
# 1. Constants
# ==========================================

REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "processed")
OUTPUT_DIR    = os.path.join(REPO_ROOT, "models", "sentiment-v1")
EMBEDDING_CACHE_DIR = os.path.join(REPO_ROOT, "processed", "embeddings_cache")

BASE_MODEL  = "bert-base-uncased"
BATCH_SIZE  = 32
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


# ==========================================
# 2. Load labeled utterances
# ==========================================

def load_sentiment_split(split: str) -> Tuple[List[str], List[str]]:
    """
    Reads processed/<SPLIT>/utterance_sentiment.jsonl and returns texts + labels.

    Args:
        split: One of 'Train', 'Validation', 'Test'.

    Returns:
        Tuple of (texts, sentiment_labels), aligned by index.
    """
    path = os.path.join(PROCESSED_DIR, split, "utterance_sentiment.jsonl")
    texts: List[str] = []
    labels: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            texts.append(record["text"])
            labels.append(record["sentiment"])

    return texts, labels


# ==========================================
# 3. Embedding model
# ==========================================

def build_embedding_model(base_model: str) -> SentenceTransformer:
    """
    Constructs a SentenceTransformer with a Transformer encoder + mean pooling,
    matching the architecture used in domain_adaptation.py so v1 and v2
    embeddings are directly comparable.

    Args:
        base_model: HuggingFace model name or local path.

    Returns:
        SentenceTransformer ready for inference (no domain adaptation).
    """
    word_embedding_model = Transformer(base_model)
    pooling_model = Pooling(
        word_embedding_model.get_embedding_dimension(),
        pooling_mode="mean"
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    """
    Encodes a list of texts into embeddings using the given SentenceTransformer.

    Args:
        model: SentenceTransformer to use for encoding.
        texts: List of input strings.
        batch_size: Encoding batch size.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim).
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def encode_texts_cached(
    model: SentenceTransformer, texts: List[str], batch_size: int, cache_path: str
) -> np.ndarray:
    """
    Encodes texts into embeddings, caching the result to disk as .npy.

    BERT encoding on CPU is the most expensive step in this pipeline, so
    repeated runs (e.g. classifier tuning) reuse the cached array instead
    of re-encoding.

    Args:
        model: SentenceTransformer to use for encoding.
        texts: List of input strings.
        batch_size: Encoding batch size.
        cache_path: Path to the .npy cache file.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim).
    """
    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    embeddings = encode_texts(model, texts, batch_size)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


# ==========================================
# 4. Classifier head
# ==========================================

def train_classifier(X_train: np.ndarray, y_train: List[str]) -> LogisticRegression:
    """
    Trains a class-weighted multinomial logistic regression on embeddings.

    Class weighting accounts for the severe class imbalance in the sentiment
    labels (neutral dominates, positive is the minority class).

    Args:
        X_train: Training embeddings, shape (n_samples, embedding_dim).
        y_train: Training sentiment labels.

    Returns:
        Fitted LogisticRegression classifier.
    """
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf: LogisticRegression, X: np.ndarray, y_true: List[str], split_name: str) -> dict:
    """
    Evaluates the classifier and returns accuracy + per-class metrics.

    Plain accuracy alone is misleading given the class imbalance (Rule 9),
    so a full classification report and confusion matrix are also computed.

    Args:
        clf: Fitted classifier.
        X: Embeddings to evaluate on.
        y_true: Ground-truth (LLM-generated) sentiment labels.
        split_name: Name of the split, for logging.

    Returns:
        Dict with accuracy, classification report, and confusion matrix.
    """
    y_pred = clf.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=SENTIMENT_LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_LABELS)

    print(f"\n--- {split_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, labels=SENTIMENT_LABELS, zero_division=0))
    print(f"Confusion matrix (rows=true, cols=pred, order={SENTIMENT_LABELS}):")
    print(cm)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


# ==========================================
# 5. Run
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train v1 (no domain adaptation) sentiment classifier.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Embedding batch size (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to save the classifier and metrics (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    print(f"Loading {BASE_MODEL} (no domain adaptation)...")
    embedder = build_embedding_model(BASE_MODEL)

    print("\nLoading labeled splits...")
    train_texts, train_labels = load_sentiment_split("Train")
    val_texts, val_labels = load_sentiment_split("Validation")
    test_texts, test_labels = load_sentiment_split("Test")
    print(f"  Train: {len(train_texts)} | Validation: {len(val_texts)} | Test: {len(test_texts)}")

    print("\nEncoding Train embeddings...")
    X_train = encode_texts_cached(
        embedder, train_texts, args.batch_size,
        os.path.join(EMBEDDING_CACHE_DIR, "v1_train.npy"),
    )
    print("Encoding Validation embeddings...")
    X_val = encode_texts_cached(
        embedder, val_texts, args.batch_size,
        os.path.join(EMBEDDING_CACHE_DIR, "v1_validation.npy"),
    )
    print("Encoding Test embeddings...")
    X_test = encode_texts_cached(
        embedder, test_texts, args.batch_size,
        os.path.join(EMBEDDING_CACHE_DIR, "v1_test.npy"),
    )

    print(f"\nEmbedding shape: {X_train.shape}")

    print("\nTraining logistic regression classifier head...")
    clf = train_classifier(X_train, train_labels)

    results = {
        "base_model": BASE_MODEL,
        "domain_adapted": False,
        "embedding_dim": X_train.shape[1],
        "train": evaluate(clf, X_train, train_labels, "Train"),
        "validation": evaluate(clf, X_val, val_labels, "Validation"),
        "test": evaluate(clf, X_test, test_labels, "Test"),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    dump(clf, os.path.join(args.output_dir, "classifier.joblib"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    val_acc = results["validation"]["accuracy"]
    print(f"\n{'='*50}")
    print(f"Validation accuracy: {val_acc:.4f} (target: >=0.80)")
    print(f"Saved classifier + metrics to: {args.output_dir}")
