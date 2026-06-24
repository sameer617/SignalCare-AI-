"""
train_sentiment_v3_roberta.py
-------------------------------
Week 2 follow-up, Experiment Track C: v3 sentiment classifier head on
TSDAE domain-adapted roberta-base embeddings (models/tsdae-adapted-roberta/),
to compare against the v2 (TSDAE bert-base-uncased) ceiling of ~0.7846
Validation accuracy (see train_sentiment_v2_experiments.py).

Pipeline:
  1. Load processed/<SPLIT>/utterance_sentiment.jsonl sentiment labels
     (LLM-generated, see CLAUDE.md Rule 6 - not gold labels).
  2. Encode utterance text with the TSDAE domain-adapted roberta-base model
     at models/tsdae-adapted-roberta/, caching embeddings to
     processed/embeddings_cache/v3_{train,validation,test}.npy.
  3. Train a class-weighted 1-layer MLP head (MLP[128], dropout=0.5,
     weight_decay=1e-3) - the same architecture as the final v1/v2
     production config (models/sentiment-v1-mlp-d05-wd3/).
  4. Evaluate on Train/Validation/Test, log to MLflow (experiment
     "signalcare-sentiment-v2") with confusion matrix plot artifacts.
  5. Save the model + metrics to models/sentiment-v3-roberta-tsdae/.

Output:
  processed/embeddings_cache/v3_{train,validation,test}.npy
  models/sentiment-v3-roberta-tsdae/model.pt
  models/sentiment-v3-roberta-tsdae/metrics.json

Usage:
  python src/train_sentiment_v3_roberta.py
"""

import os
import json

import numpy as np
import mlflow
from sentence_transformers import SentenceTransformer

from train_sentiment_v2_experiments import (
    SentimentMLP,
    train_mlp,
    evaluate,
    plot_confusion_matrix,
    SENTIMENT_LABELS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from train_sentiment_v2 import load_sentiment_split, encode_texts_cached


# ==========================================
# 1. Constants
# ==========================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "processed")
EMBEDDING_CACHE_DIR = os.path.join(REPO_ROOT, "processed", "embeddings_cache")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

VARIANT = "v3"
TSDAE_ROBERTA_MODEL_DIR = os.path.join(MODELS_DIR, "tsdae-adapted-roberta")
BATCH_SIZE = 32
OUTPUT_DIR = os.path.join(MODELS_DIR, "sentiment-v3-roberta-tsdae")

RUN_NAME = "v3_roberta_tsdae_mlp128_d0.5_wd1e-3"
HIDDEN_DIMS = [128]
DROPOUT = 0.5
WEIGHT_DECAY = 1e-3
EPOCHS = 50
BATCH_SIZE_TRAIN = 64
LR = 1e-3
PATIENCE = 10
TARGET_VAL_ACCURACY = 0.80


# ==========================================
# 2. Run
# ==========================================

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(f"Loading TSDAE domain-adapted roberta-base model from {TSDAE_ROBERTA_MODEL_DIR}...")
    embedder = SentenceTransformer(TSDAE_ROBERTA_MODEL_DIR)

    print("\nLoading labeled splits...")
    train_texts, train_labels_str = load_sentiment_split("Train")
    val_texts, val_labels_str = load_sentiment_split("Validation")
    test_texts, test_labels_str = load_sentiment_split("Test")
    print(f"  Train: {len(train_texts)} | Validation: {len(val_texts)} | Test: {len(test_texts)}")

    label_to_idx = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
    train_labels = np.array([label_to_idx[l] for l in train_labels_str])
    val_labels = np.array([label_to_idx[l] for l in val_labels_str])
    test_labels = np.array([label_to_idx[l] for l in test_labels_str])

    print("\nEncoding Train embeddings...")
    X_train = encode_texts_cached(
        embedder, train_texts, BATCH_SIZE,
        os.path.join(EMBEDDING_CACHE_DIR, "v3_train.npy"),
    )
    print("Encoding Validation embeddings...")
    X_val = encode_texts_cached(
        embedder, val_texts, BATCH_SIZE,
        os.path.join(EMBEDDING_CACHE_DIR, "v3_validation.npy"),
    )
    print("Encoding Test embeddings...")
    X_test = encode_texts_cached(
        embedder, test_texts, BATCH_SIZE,
        os.path.join(EMBEDDING_CACHE_DIR, "v3_test.npy"),
    )
    print(f"\nEmbedding shape: {X_train.shape}")

    params = dict(
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE_TRAIN,
        lr=LR,
        patience=PATIENCE,
        use_batchnorm=False,
        label_smoothing=0.0,
        use_class_weights=True,
        resample_method=None,
        base_model="roberta-base",
        domain_adapted_model_dir=TSDAE_ROBERTA_MODEL_DIR,
    )

    print(f"\n{'='*60}")
    print(f"Run: {RUN_NAME}")
    print(f"Params: {params}")
    print(f"{'='*60}")

    model, history = train_mlp(
        X_train, train_labels, X_val, val_labels,
        hidden_dims=HIDDEN_DIMS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE_TRAIN,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        dropout=DROPOUT,
        use_batchnorm=False,
        label_smoothing=0.0,
        use_class_weights=True,
    )

    train_metrics = evaluate(model, X_train, train_labels, "Train")
    val_metrics = evaluate(model, X_val, val_labels, "Validation")
    test_metrics = evaluate(model, X_test, test_labels, "Test")

    results = {
        "run_name": RUN_NAME,
        "variant": VARIANT,
        "base_model": "roberta-base",
        "domain_adapted": True,
        "domain_adapted_model_dir": TSDAE_ROBERTA_MODEL_DIR,
        "embedding_dim": X_train.shape[1],
        "params": params,
        "history": history,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metric("train_accuracy", train_metrics["accuracy"])
        mlflow.log_metric("val_accuracy", val_metrics["accuracy"])
        mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
        mlflow.log_metric("val_macro_f1", val_metrics["classification_report"]["macro avg"]["f1-score"])
        mlflow.log_metric("val_positive_f1", val_metrics["classification_report"]["positive"]["f1-score"])
        mlflow.log_metric("val_positive_recall", val_metrics["classification_report"]["positive"]["recall"])

        for epoch, (loss, acc) in enumerate(zip(history["train_loss"], history["val_accuracy"])):
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_accuracy_curve", acc, step=epoch)

        run_plot_dir = os.path.join(MODELS_DIR, "sentiment-v3-roberta-tsdae", "plots", RUN_NAME)
        os.makedirs(run_plot_dir, exist_ok=True)
        for split_name, split_metrics in [("train", train_metrics), ("validation", val_metrics), ("test", test_metrics)]:
            plot_path = os.path.join(run_plot_dir, f"confusion_matrix_{split_name}.png")
            plot_confusion_matrix(
                split_metrics["confusion_matrix"],
                title=f"{RUN_NAME} - {split_name} (acc={split_metrics['accuracy']:.4f})",
                save_path=plot_path,
            )
            mlflow.log_artifact(plot_path, artifact_path="confusion_matrices")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import torch
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    val_acc = val_metrics["accuracy"]
    test_acc = test_metrics["accuracy"]
    flag = "PASS" if val_acc >= TARGET_VAL_ACCURACY else "below target"
    print(f"\n{'='*60}")
    print(f"{RUN_NAME}")
    print(f"Validation accuracy: {val_acc:.4f} (target >= {TARGET_VAL_ACCURACY}) [{flag}]")
    print(f"Test accuracy:       {test_acc:.4f}")
    print(f"Saved model + metrics to: {OUTPUT_DIR}")
