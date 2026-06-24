"""
train_sentiment_v4_ensemble.py
--------------------------------
Week 2 follow-up, Experiment Track D: v4 sentiment classifier head on a
concatenated v1 + v2 embedding ensemble (raw bert-base-uncased + TSDAE
domain-adapted bert-base-uncased, 768 + 768 = 1536 dims per utterance).

Rationale: v1 (raw BERT) and v2 (TSDAE BERT) embeddings come from the same
base architecture but different training objectives, so they may encode
complementary information. Both are already cached
(processed/embeddings_cache/v1_*.npy, v2_*.npy), so this experiment is
essentially free to run.

Pipeline:
  1. Load processed/<SPLIT>/utterance_sentiment.jsonl sentiment labels.
  2. Load cached v1 + v2 embeddings, concatenate along the feature axis.
  3. Sweep a small set of MLP head configs (same architectures already
     explored in train_sentiment_v2_experiments.py) on the 1536-dim input.
  4. Evaluate on Train/Validation/Test, log to MLflow (experiment
     "signalcare-sentiment-v2") with confusion matrix plot artifacts.
  5. Save the best model + metrics to models/sentiment-v4-ensemble/.

Output:
  models/sentiment-v4-ensemble/model.pt
  models/sentiment-v4-ensemble/metrics.json

Usage:
  python src/train_sentiment_v4_ensemble.py
"""

import os
import json

import numpy as np
import torch
import mlflow

from train_sentiment_v2_experiments import (
    train_mlp,
    evaluate,
    plot_confusion_matrix,
    load_sentiment_labels,
    load_cached_embeddings,
    SENTIMENT_LABELS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)


# ==========================================
# 1. Constants
# ==========================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
OUTPUT_DIR = os.path.join(MODELS_DIR, "sentiment-v4-ensemble")

VARIANT = "v4"
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 10
TARGET_VAL_ACCURACY = 0.80


# ==========================================
# 2. Run
# ==========================================

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Loading labeled splits...")
    label_to_idx = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
    train_labels = np.array([label_to_idx[l] for l in load_sentiment_labels("Train")])
    val_labels = np.array([label_to_idx[l] for l in load_sentiment_labels("Validation")])
    test_labels = np.array([label_to_idx[l] for l in load_sentiment_labels("Test")])
    print(f"  Train: {len(train_labels)} | Validation: {len(val_labels)} | Test: {len(test_labels)}")

    print("\nLoading + concatenating cached v1 + v2 embeddings...")
    X_train = np.concatenate(
        [load_cached_embeddings("v1", "Train"), load_cached_embeddings("v2", "Train")], axis=1
    )
    X_val = np.concatenate(
        [load_cached_embeddings("v1", "Validation"), load_cached_embeddings("v2", "Validation")], axis=1
    )
    X_test = np.concatenate(
        [load_cached_embeddings("v1", "Test"), load_cached_embeddings("v2", "Test")], axis=1
    )
    print(f"Ensemble embedding shape: {X_train.shape}")

    base_params = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE)

    configs = [
        # Same head as the v1/v2 production config, on the 1536-dim ensemble input.
        ("v4_ensemble_mlp128_d0.5_wd1e-3", dict(
            base_params, hidden_dims=[128], dropout=0.5, weight_decay=1e-3,
            use_batchnorm=False, label_smoothing=0.0, use_class_weights=True,
        )),
        # Same architecture as the best v2 "deep" config.
        ("v4_ensemble_mlp256-64_d0.5_wd0.001_ls0.1", dict(
            base_params, hidden_dims=[256, 64], dropout=0.5, weight_decay=1e-3,
            use_batchnorm=True, label_smoothing=0.1, use_class_weights=True,
        )),
        # Heavier dropout to counter the doubled input dimensionality / overfitting risk.
        ("v4_ensemble_mlp128_d0.6_wd1e-3", dict(
            base_params, hidden_dims=[128], dropout=0.6, weight_decay=1e-3,
            use_batchnorm=False, label_smoothing=0.0, use_class_weights=True,
        )),
    ]

    print(f"\nRunning {len(configs)} experiment(s)...")

    best_val_acc = -1.0
    best_results = None
    best_model = None

    for run_name, params in configs:
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"Params: {params}")
        print(f"{'='*60}")

        model, history = train_mlp(
            X_train, train_labels, X_val, val_labels,
            hidden_dims=params["hidden_dims"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            patience=params["patience"],
            dropout=params["dropout"],
            use_batchnorm=params["use_batchnorm"],
            label_smoothing=params["label_smoothing"],
            use_class_weights=params["use_class_weights"],
        )

        train_metrics = evaluate(model, X_train, train_labels, "Train")
        val_metrics = evaluate(model, X_val, val_labels, "Validation")
        test_metrics = evaluate(model, X_test, test_labels, "Test")

        results = {
            "run_name": run_name,
            "variant": VARIANT,
            "ensemble_of": ["v1", "v2"],
            "embedding_dim": X_train.shape[1],
            "params": params,
            "history": history,
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        }

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_param("ensemble_of", "v1+v2")
            mlflow.log_metric("train_accuracy", train_metrics["accuracy"])
            mlflow.log_metric("val_accuracy", val_metrics["accuracy"])
            mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
            mlflow.log_metric("val_macro_f1", val_metrics["classification_report"]["macro avg"]["f1-score"])
            mlflow.log_metric("val_positive_f1", val_metrics["classification_report"]["positive"]["f1-score"])
            mlflow.log_metric("val_positive_recall", val_metrics["classification_report"]["positive"]["recall"])

            for epoch, (loss, acc) in enumerate(zip(history["train_loss"], history["val_accuracy"])):
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("val_accuracy_curve", acc, step=epoch)

            run_plot_dir = os.path.join(OUTPUT_DIR, "plots", run_name)
            os.makedirs(run_plot_dir, exist_ok=True)
            for split_name, split_metrics in [("train", train_metrics), ("validation", val_metrics), ("test", test_metrics)]:
                plot_path = os.path.join(run_plot_dir, f"confusion_matrix_{split_name}.png")
                plot_confusion_matrix(
                    split_metrics["confusion_matrix"],
                    title=f"{run_name} - {split_name} (acc={split_metrics['accuracy']:.4f})",
                    save_path=plot_path,
                )
                mlflow.log_artifact(plot_path, artifact_path="confusion_matrices")

        val_acc = val_metrics["accuracy"]
        flag = "PASS" if val_acc >= TARGET_VAL_ACCURACY else ""
        print(f"\n{run_name}: validation accuracy = {val_acc:.4f} (target >= {TARGET_VAL_ACCURACY}) {flag}")

        all_results_path = os.path.join(OUTPUT_DIR, f"{run_name}.json")
        os.makedirs(os.path.dirname(all_results_path), exist_ok=True)
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_results = results
            best_model = model

    print(f"\n{'='*60}")
    print(f"Best run: {best_results['run_name']} | Validation accuracy = {best_val_acc:.4f}")
    print(f"Test accuracy = {best_results['test']['accuracy']:.4f}")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=2)
    print(f"Saved best model + metrics to: {OUTPUT_DIR}")
