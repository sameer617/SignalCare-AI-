"""
train_sentiment_v2_experiments.py
-----------------------------------
Week 2 follow-up: experiments to push the v2 (TSDAE domain-adapted
bert-base-uncased) sentiment classifier head toward >=0.80 Validation
accuracy (current best v2 head: 0.7601, see models/sentiment-v2-mlp/).

Reuses the cached v2 embeddings from processed/embeddings_cache/
(produced by train_sentiment_v2.py) — no re-encoding needed.

Experiment tracks (selectable via --track):
  - "oversample": RandomOverSampler / SMOTE on the Train embeddings to
    rebalance the minority "positive" class before training a 1-layer MLP
    (same architecture as the current best v1/v2 MLP heads).
  - "deep": 2-layer MLP architectures with BatchNorm and label smoothing,
    which were previously avoided due to overfitting on this small dataset.
  - "all": runs both tracks as a grid of experiments.

Each run is logged to MLflow (params, metrics, confusion matrix plot
artifact for Train/Validation/Test) under the experiment
"signalcare-sentiment-v2". The best-performing checkpoint (by Validation
accuracy) across all runs is saved to models/sentiment-v2-best/.

Output:
  mlruns/                          (MLflow tracking data, gitignored)
  models/sentiment-v2-best/model.pt
  models/sentiment-v2-best/metrics.json

Usage:
  python src/train_sentiment_v2_experiments.py --track all
  python src/train_sentiment_v2_experiments.py --track oversample
  python src/train_sentiment_v2_experiments.py --track deep
"""

import os
import json
import argparse
import itertools
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

from sentiment_model import SentimentMLP, SENTIMENT_LABELS


# ==========================================
# 1. Constants
# ==========================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "processed")
EMBEDDING_CACHE_DIR = os.path.join(REPO_ROOT, "processed", "embeddings_cache")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

VARIANT = "v2"

EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 10
SEED = 42
TARGET_VAL_ACCURACY = 0.80

MLFLOW_EXPERIMENT_NAME = "signalcare-sentiment-v2"
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(REPO_ROOT, 'mlflow.db').replace(os.sep, '/')}"
BEST_MODEL_DIR = os.path.join(MODELS_DIR, "sentiment-v2-best")


# ==========================================
# 2. Load labels + cached embeddings
# ==========================================

def load_sentiment_labels(split: str) -> List[str]:
    """
    Reads processed/<SPLIT>/utterance_sentiment.jsonl and returns sentiment labels.

    Args:
        split: One of 'Train', 'Validation', 'Test'.

    Returns:
        List of sentiment labels, in file order (matches cached embedding order).
    """
    path = os.path.join(PROCESSED_DIR, split, "utterance_sentiment.jsonl")
    labels: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            labels.append(record["sentiment"])

    return labels


def load_cached_embeddings(variant: str, split: str) -> np.ndarray:
    """
    Loads cached embeddings produced by train_sentiment_v1.py / train_sentiment_v2.py.

    Args:
        variant: 'v1' (raw bert-base-uncased) or 'v2' (TSDAE domain-adapted).
        split: One of 'Train', 'Validation', 'Test'.

    Returns:
        np.ndarray of shape (n_samples, embedding_dim).

    Raises:
        FileNotFoundError: If the cache file does not exist.
    """
    cache_path = os.path.join(EMBEDDING_CACHE_DIR, f"{variant}_{split.lower()}.npy")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cached embeddings not found at {cache_path}. "
            f"Run train_sentiment_{variant}.py first."
        )
    return np.load(cache_path)


# ==========================================
# 3. Resampling
# ==========================================

def resample_train(
    X_train: np.ndarray, y_train: np.ndarray, method: Optional[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rebalances the training set to address severe class imbalance (Rule 9),
    particularly the minority "positive" class.

    Args:
        X_train: Training embeddings, shape (n_train, embedding_dim).
        y_train: Integer-encoded training labels, shape (n_train,).
        method: One of None, "random_oversample", or "smote".

    Returns:
        Tuple of (resampled X_train, resampled y_train).
    """
    if method is None:
        return X_train, y_train

    if method == "random_oversample":
        from imblearn.over_sampling import RandomOverSampler

        sampler = RandomOverSampler(random_state=SEED)
    elif method == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=SEED, k_neighbors=5)
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res


# ==========================================
# 5. Training
# ==========================================

def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Computes inverse-frequency class weights for cross-entropy loss, to
    account for residual class imbalance in the training labels.

    Args:
        y_train: Integer-encoded training labels, shape (n_samples,).
        num_classes: Total number of classes.

    Returns:
        Tensor of per-class weights, shape (num_classes,).
    """
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = len(y_train) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    dropout: float,
    use_batchnorm: bool,
    label_smoothing: float,
    use_class_weights: bool,
) -> Tuple[SentimentMLP, dict]:
    """
    Trains the MLP with early stopping on Validation accuracy.

    Args:
        X_train: Training embeddings, shape (n_train, embedding_dim).
        y_train: Integer-encoded training labels, shape (n_train,).
        X_val: Validation embeddings, shape (n_val, embedding_dim).
        y_val: Integer-encoded validation labels, shape (n_val,).
        hidden_dims: Sizes of each hidden layer, in order.
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate for Adam.
        weight_decay: L2 weight decay for Adam.
        patience: Epochs without Validation accuracy improvement before stopping.
        dropout: Dropout probability applied after each hidden layer activation.
        use_batchnorm: If True, applies BatchNorm1d after each hidden Linear layer.
        label_smoothing: Label smoothing factor for cross-entropy loss.
        use_class_weights: If True, weights the loss by inverse class frequency.

    Returns:
        Tuple of (best model, training history dict).
    """
    torch.manual_seed(SEED)

    input_dim = X_train.shape[1]
    num_classes = len(SENTIMENT_LABELS)

    model = SentimentMLP(input_dim, hidden_dims, num_classes, dropout, use_batchnorm)
    class_weights = compute_class_weights(y_train, num_classes) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        history["train_loss"].append(epoch_loss)
        history["val_accuracy"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def predict(model: SentimentMLP, X: np.ndarray) -> np.ndarray:
    """
    Runs inference and returns predicted class indices.

    Args:
        model: Trained SentimentMLP.
        X: Embeddings, shape (n_samples, embedding_dim).

    Returns:
        np.ndarray of predicted class indices, shape (n_samples,).
    """
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return logits.argmax(dim=1).numpy()


def evaluate(model: SentimentMLP, X: np.ndarray, y_true_idx: np.ndarray, split_name: str) -> dict:
    """
    Evaluates the model and returns accuracy + per-class metrics.

    Plain accuracy alone is misleading given the class imbalance (Rule 9),
    so a full classification report and confusion matrix are also computed.

    Args:
        model: Trained SentimentMLP.
        X: Embeddings to evaluate on.
        y_true_idx: Integer-encoded ground-truth labels.
        split_name: Name of the split, for logging.

    Returns:
        Dict with accuracy, classification report, and confusion matrix.
    """
    y_pred_idx = predict(model, X)
    y_true = [SENTIMENT_LABELS[i] for i in y_true_idx]
    y_pred = [SENTIMENT_LABELS[i] for i in y_pred_idx]

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
# 6. Confusion matrix plotting
# ==========================================

def plot_confusion_matrix(cm: List[List[int]], title: str, save_path: str) -> None:
    """
    Renders a confusion matrix as a heatmap with annotated cell counts and
    saves it to disk.

    Args:
        cm: Confusion matrix as a nested list (rows=true, cols=pred).
        title: Plot title (e.g. "run_name - Validation").
        save_path: File path to save the PNG to.
    """
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")

    ax.set_xticks(range(len(SENTIMENT_LABELS)))
    ax.set_yticks(range(len(SENTIMENT_LABELS)))
    ax.set_xticklabels(SENTIMENT_LABELS)
    ax.set_yticklabels(SENTIMENT_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm_arr.max() / 2.0
    for i, j in itertools.product(range(cm_arr.shape[0]), range(cm_arr.shape[1])):
        ax.text(
            j, i, format(cm_arr[i, j], "d"),
            ha="center", va="center",
            color="white" if cm_arr[i, j] > thresh else "black",
        )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ==========================================
# 7. Experiment runner
# ==========================================

def run_experiment(
    run_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict,
) -> dict:
    """
    Runs a single training experiment, evaluates it, logs everything to
    MLflow (params, metrics, confusion matrix plots), and returns the results.

    Args:
        run_name: MLflow run name / identifier for this configuration.
        X_train: (Resampled) training embeddings.
        y_train: (Resampled) integer-encoded training labels.
        X_val: Validation embeddings.
        y_val: Integer-encoded validation labels.
        X_test: Test embeddings.
        y_test: Integer-encoded test labels.
        params: Dict of hyperparameters for this run. Expected keys:
            hidden_dims, dropout, weight_decay, use_batchnorm,
            label_smoothing, use_class_weights, resample_method,
            lr, epochs, batch_size, patience.

    Returns:
        Dict with the full results (params, history, train/val/test metrics).
    """
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Params: {params}")
    print(f"{'='*60}")

    model, history = train_mlp(
        X_train, y_train, X_val, y_val,
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

    train_metrics = evaluate(model, X_train, y_train, "Train")
    val_metrics = evaluate(model, X_val, y_val, "Validation")
    test_metrics = evaluate(model, X_test, y_test, "Test")

    results = {
        "run_name": run_name,
        "variant": VARIANT,
        "params": params,
        "history": history,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    with mlflow.start_run(run_name=run_name):
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

        run_plot_dir = os.path.join(MODELS_DIR, "sentiment-v2-experiments", "plots", run_name)
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

    return results, model


# ==========================================
# 8. Run
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiments to push the v2 (TSDAE) sentiment classifier head past 0.80 Validation accuracy."
    )
    parser.add_argument(
        "--track", type=str, default="all", choices=["oversample", "deep", "round2", "all"],
        help="Which experiment track to run (default: all).",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Loading labeled splits...")
    label_to_idx = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
    train_labels = np.array([label_to_idx[l] for l in load_sentiment_labels("Train")])
    val_labels = np.array([label_to_idx[l] for l in load_sentiment_labels("Validation")])
    test_labels = np.array([label_to_idx[l] for l in load_sentiment_labels("Test")])
    print(f"  Train: {len(train_labels)} | Validation: {len(val_labels)} | Test: {len(test_labels)}")

    print("\nLoading cached v2 (TSDAE) embeddings...")
    X_train_full = load_cached_embeddings(VARIANT, "Train")
    X_val = load_cached_embeddings(VARIANT, "Validation")
    X_test = load_cached_embeddings(VARIANT, "Test")
    print(f"Embedding shape: {X_train_full.shape}")

    base_params = dict(
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE,
    )

    configs = []  # list of (run_name, resample_method, params)

    if args.track in ("oversample", "all"):
        for resample_method in ["random_oversample", "smote"]:
            for dropout, weight_decay in [(0.5, 1e-3), (0.4, 1e-3)]:
                run_name = f"v2_oversample-{resample_method}_mlp128_d{dropout}_wd{weight_decay}"
                params = dict(
                    base_params,
                    hidden_dims=[128],
                    dropout=dropout,
                    weight_decay=weight_decay,
                    use_batchnorm=False,
                    label_smoothing=0.0,
                    use_class_weights=False,  # resampling already balances classes
                    resample_method=resample_method,
                )
                configs.append((run_name, resample_method, params))

    if args.track in ("deep", "all"):
        for hidden_dims in [[256, 64], [128, 64]]:
            for dropout, weight_decay, label_smoothing in [(0.5, 1e-3, 0.0), (0.5, 1e-3, 0.1)]:
                run_name = f"v2_deep_mlp{'-'.join(map(str, hidden_dims))}_d{dropout}_wd{weight_decay}_ls{label_smoothing}"
                params = dict(
                    base_params,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    weight_decay=weight_decay,
                    use_batchnorm=True,
                    label_smoothing=label_smoothing,
                    use_class_weights=True,
                    resample_method=None,
                )
                configs.append((run_name, None, params))

    if args.track == "round2":
        # Refine around the best "deep" result so far
        # (v2_deep_mlp256-64_d0.5_wd0.001_ls0.1 -> Validation accuracy 0.7846):
        # sweep dropout/weight_decay/label_smoothing, and also combine the
        # winning architecture with oversampling.
        for dropout, weight_decay, label_smoothing in [
            (0.4, 1e-3, 0.1),
            (0.5, 1e-4, 0.1),
            (0.5, 5e-3, 0.1),
            (0.5, 1e-3, 0.2),
            (0.6, 1e-3, 0.1),
        ]:
            run_name = f"v2_round2_mlp256-64_d{dropout}_wd{weight_decay}_ls{label_smoothing}"
            params = dict(
                base_params,
                hidden_dims=[256, 64],
                dropout=dropout,
                weight_decay=weight_decay,
                use_batchnorm=True,
                label_smoothing=label_smoothing,
                use_class_weights=True,
                resample_method=None,
            )
            configs.append((run_name, None, params))

        for resample_method in ["random_oversample", "smote"]:
            run_name = f"v2_round2_oversample-{resample_method}_mlp256-64_d0.5_wd0.001_ls0.1"
            params = dict(
                base_params,
                hidden_dims=[256, 64],
                dropout=0.5,
                weight_decay=1e-3,
                use_batchnorm=True,
                label_smoothing=0.1,
                use_class_weights=False,  # resampling already balances classes
                resample_method=resample_method,
            )
            configs.append((run_name, resample_method, params))

    print(f"\nRunning {len(configs)} experiment(s)...")

    best_val_acc = -1.0
    best_results = None
    best_model = None

    for run_name, resample_method, params in configs:
        X_tr, y_tr = resample_train(X_train_full, train_labels, resample_method)
        if resample_method is not None:
            print(f"\n[{run_name}] Resampled Train with {resample_method}: "
                  f"{X_train_full.shape[0]} -> {X_tr.shape[0]} samples")

        results, model = run_experiment(
            run_name, X_tr, y_tr, X_val, val_labels, X_test, test_labels, params,
        )

        all_results_path = os.path.join(MODELS_DIR, "sentiment-v2-experiments", f"{run_name}.json")
        os.makedirs(os.path.dirname(all_results_path), exist_ok=True)
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        val_acc = results["validation"]["accuracy"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_results = results
            best_model = model

    print(f"\n{'='*60}")
    print(f"Best run: {best_results['run_name']} | Validation accuracy = {best_val_acc:.4f}")
    print(f"{'='*60}")

    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(BEST_MODEL_DIR, "model.pt"))
    with open(os.path.join(BEST_MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=2)
    print(f"Saved best model + metrics to: {BEST_MODEL_DIR}")
