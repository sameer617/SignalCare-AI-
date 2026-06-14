"""
train_sentiment_mlp.py
------------------------
Week 2: PyTorch MLP classifier head variant for the v1/v2 sentiment
classifiers, for comparison against the logistic regression heads in
train_sentiment_v1.py / train_sentiment_v2.py and the random forest heads
in train_sentiment_rf.py.

Reuses the cached embeddings from processed/embeddings_cache/ (produced by
train_sentiment_v1.py and train_sentiment_v2.py) — no re-encoding needed.

Pipeline:
  1. Load cached v1 (raw bert-base-uncased) and v2 (TSDAE domain-adapted)
     embeddings + sentiment labels for Train/Validation/Test.
  2. Train a small feedforward network (768 -> hidden -> 3) with dropout and
     class-weighted cross-entropy loss on each embedding set, tracking the
     best Validation accuracy checkpoint (early stopping).
  3. Evaluate the best checkpoint on Validation (target: >=80% accuracy) and Test.
  4. Save each model's weights and a metrics report.

Output:
  models/sentiment-v1-mlp/model.pt
  models/sentiment-v1-mlp/metrics.json
  models/sentiment-v2-mlp/model.pt
  models/sentiment-v2-mlp/metrics.json

Usage:
  python src/train_sentiment_mlp.py
  python src/train_sentiment_mlp.py --epochs 100 --hidden-dim 256
"""

import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. Constants
# ==========================================

REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "processed")
EMBEDDING_CACHE_DIR = os.path.join(REPO_ROOT, "processed", "embeddings_cache")
MODELS_DIR    = os.path.join(REPO_ROOT, "models")

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
VARIANTS = ["v1", "v2"]

HIDDEN_DIMS  = [128]
DROPOUT      = 0.3
EPOCHS       = 50
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 10  # epochs without Validation accuracy improvement before stopping
SEED         = 42


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
        FileNotFoundError: If the cache file does not exist (run the
            corresponding train_sentiment_v*.py first).
    """
    cache_path = os.path.join(EMBEDDING_CACHE_DIR, f"{variant}_{split.lower()}.npy")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cached embeddings not found at {cache_path}. "
            f"Run train_sentiment_{variant}.py first."
        )
    return np.load(cache_path)


# ==========================================
# 3. Model
# ==========================================

class SentimentMLP(nn.Module):
    """Feedforward classifier head: embedding_dim -> hidden_dims[...] -> num_classes."""

    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float):
        """
        Args:
            input_dim: Dimensionality of the input embeddings.
            hidden_dims: Sizes of each hidden layer, in order.
            num_classes: Number of output classes.
            dropout: Dropout probability applied after each hidden layer activation.
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings, shape (batch_size, input_dim).

        Returns:
            Class logits, shape (batch_size, num_classes).
        """
        return self.net(x)


# ==========================================
# 4. Training
# ==========================================

def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Computes inverse-frequency class weights for cross-entropy loss, to
    account for the severe class imbalance in the sentiment labels.

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
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    hidden_dims: List[int], epochs: int, batch_size: int, lr: float,
    weight_decay: float, patience: int, dropout: float,
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

    Returns:
        Tuple of (best model, training history dict).
    """
    torch.manual_seed(SEED)

    input_dim = X_train.shape[1]
    num_classes = len(SENTIMENT_LABELS)

    model = SentimentMLP(input_dim, hidden_dims, num_classes, dropout)
    class_weights = compute_class_weights(y_train, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        print(f"  Epoch {epoch+1:3d}/{epochs} | train_loss={epoch_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
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
# 5. Run
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch MLP sentiment classifier heads on cached v1/v2 embeddings.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=HIDDEN_DIMS, help=f"Hidden layer sizes, in order (default: {HIDDEN_DIMS}).")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Max training epochs (default: {EPOCHS}).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Training batch size (default: {BATCH_SIZE}).")
    parser.add_argument("--lr", type=float, default=LR, help=f"Learning rate (default: {LR}).")
    parser.add_argument("--patience", type=int, default=PATIENCE, help=f"Early stopping patience (default: {PATIENCE}).")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help=f"Dropout probability (default: {DROPOUT}).")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help=f"L2 weight decay for Adam (default: {WEIGHT_DECAY}).")
    parser.add_argument("--variants", type=str, nargs="+", default=VARIANTS, choices=VARIANTS, help=f"Which embedding variants to train (default: {VARIANTS}).")
    parser.add_argument("--output-suffix", type=str, default="mlp", help="Suffix for the output directory name: models/sentiment-<variant>-<suffix> (default: mlp).")
    args = parser.parse_args()

    print("Loading labeled splits...")
    label_to_idx = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
    train_labels = np.array([label_to_idx[l] for l in load_sentiment_labels("Train")])
    val_labels = np.array([label_to_idx[l] for l in load_sentiment_labels("Validation")])
    test_labels = np.array([label_to_idx[l] for l in load_sentiment_labels("Test")])
    print(f"  Train: {len(train_labels)} | Validation: {len(val_labels)} | Test: {len(test_labels)}")

    for variant in args.variants:
        print(f"\n{'='*50}")
        print(f"Variant: {variant} (PyTorch MLP head, hidden_dims={args.hidden_dims})")
        print(f"{'='*50}")

        X_train = load_cached_embeddings(variant, "Train")
        X_val = load_cached_embeddings(variant, "Validation")
        X_test = load_cached_embeddings(variant, "Test")
        print(f"Embedding shape: {X_train.shape}")

        print(f"\nTraining MLP (hidden_dims={args.hidden_dims}, dropout={args.dropout}, weight_decay={args.weight_decay})...")
        model, history = train_mlp(
            X_train, train_labels, X_val, val_labels,
            hidden_dims=args.hidden_dims, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay, patience=args.patience, dropout=args.dropout,
        )

        results = {
            "variant": variant,
            "classifier": "PyTorch MLP",
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "embedding_dim": X_train.shape[1],
            "history": history,
            "train": evaluate(model, X_train, train_labels, "Train"),
            "validation": evaluate(model, X_val, val_labels, "Validation"),
            "test": evaluate(model, X_test, test_labels, "Test"),
        }

        output_dir = os.path.join(MODELS_DIR, f"sentiment-{variant}-{args.output_suffix}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        val_acc = results["validation"]["accuracy"]
        print(f"\n{variant} validation accuracy: {val_acc:.4f} (target: >=0.80)")
        print(f"Saved model + metrics to: {output_dir}")
