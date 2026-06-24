"""
sentiment_model.py
-------------------
Shared definition of the sentiment classifier head (`SentimentMLP`) and the
sentiment label set (`SENTIMENT_LABELS`), used by both the training/experiment
scripts (train_sentiment_v2_experiments.py, v3, v4) and the live inference
path (run_temporal_on_transcript.py).

Kept dependency-light (torch only) so the inference path does not need to
import training-only dependencies (mlflow, matplotlib) via
train_sentiment_v2_experiments.py.
"""

from typing import List

import torch
import torch.nn as nn


# ==========================================
# 1. Constants
# ==========================================

SENTIMENT_LABELS = ["negative", "neutral", "positive"]


# ==========================================
# 2. Model
# ==========================================

class SentimentMLP(nn.Module):
    """Feedforward classifier head: embedding_dim -> hidden_dims[...] -> num_classes.

    Optionally applies BatchNorm after each hidden Linear layer, which helps
    stabilize deeper (2-layer) heads on this small dataset.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float,
        use_batchnorm: bool = False,
    ):
        """
        Args:
            input_dim: Dimensionality of the input embeddings.
            hidden_dims: Sizes of each hidden layer, in order.
            num_classes: Number of output classes.
            dropout: Dropout probability applied after each hidden layer activation.
            use_batchnorm: If True, applies BatchNorm1d after each hidden Linear layer.
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
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
