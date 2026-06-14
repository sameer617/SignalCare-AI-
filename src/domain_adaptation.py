"""
domain_adaptation.py
--------------------
Adapts a pretrained sentence-transformer to the HOPE therapy corpus
using TSDAE (Transformer-based Sequential Denoising Auto-Encoder).

TSDAE corrupts each sentence by randomly deleting tokens, then trains
the encoder-decoder pair to reconstruct the original. This unsupervised
objective pulls the model's embedding space toward the target domain
without requiring any labels.

Input:  processed/<SPLIT>/utterances.jsonl  (all splits, patient turns only)
Output: models/tsdae-adapted/  (sentence-transformers compatible directory)

Usage:
  python src/domain_adaptation.py
  python src/domain_adaptation.py --epochs 3 --batch-size 8
  python src/domain_adaptation.py --no-val-test   # use Train utterances only
"""

import os
import json
import argparse
from typing import List

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.sentence_transformer.modules import Transformer, Pooling
from sentence_transformers.sentence_transformer.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.sentence_transformer.losses import DenoisingAutoEncoderLoss

import logging

# ==========================================
# 1. Constants
# ==========================================

REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR  = os.path.join(REPO_ROOT, "processed")
OUTPUT_DIR     = os.path.join(REPO_ROOT, "models", "tsdae-adapted")
ALL_SPLITS     = ["Train", "Validation", "Test"]

BASE_MODEL     = "bert-base-uncased"
EPOCHS         = 1        # TSDAE authors use 1 epoch; increase only if loss hasn't converged
BATCH_SIZE     = 8        # safe default; raise to 16 if GPU VRAM allows
DELETION_RATIO = 0.6      # fraction of tokens deleted per sentence (TSDAE paper default)
SCHEDULER      = "constantlr"
WARMUP_STEPS   = 500
WEIGHT_DECAY   = 1e-5
LR             = 3e-5


# ==========================================
# 2. Load corpus
# ==========================================

def load_utterances(splits: List[str]) -> List[str]:
    """
    Reads utterance text from processed/<SPLIT>/utterances.jsonl for each split.

    Args:
        splits: List of split names, e.g. ['Train', 'Validation', 'Test'].

    Returns:
        Flat list of utterance strings.
    """
    sentences: List[str] = []
    for split in splits:
        jsonl_path = os.path.join(PROCESSED_DIR, split, "utterances.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"[WARN] utterances.jsonl not found for split '{split}' — skipping.")
            continue
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                text = record.get("text", "").strip()
                if text:
                    sentences.append(text)
        print(f"  Loaded {split}: {len(sentences)} sentences total so far.")
    return sentences


# ==========================================
# 3. Build model
# ==========================================

def build_model(base_model: str) -> SentenceTransformer:
    """
    Constructs a SentenceTransformer with a Transformer encoder + mean pooling.
    Using explicit component construction so the Pooling layer config is correct
    for TSDAE's reconstruction head.

    Args:
        base_model: HuggingFace model name or local path.

    Returns:
        SentenceTransformer ready for TSDAE training.
    """
    word_embedding_model = Transformer(base_model)
    pooling_model = Pooling(
        word_embedding_model.get_embedding_dimension(),
        pooling_mode="mean"
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


# ==========================================
# 4. TSDAE training
# ==========================================

def run_tsdae(
    sentences: List[str],
    output_dir: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    deletion_ratio: float = DELETION_RATIO,
) -> None:
    """
    Trains TSDAE domain adaptation on the given sentences and saves the model.

    Args:
        sentences:      List of training sentences.
        output_dir:     Directory to save the adapted model.
        epochs:         Number of training epochs.
        batch_size:     Per-device training batch size.
        deletion_ratio: Fraction of tokens to delete for the noise function.
    """
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    print(f"\nBuilding model from: {BASE_MODEL}")
    model = build_model(BASE_MODEL)

    print(f"Corpus size: {len(sentences):,} sentences")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Deletion ratio: {deletion_ratio}")

    # DenoisingAutoEncoderDataset wraps raw sentences with the noise function
    train_dataset = DenoisingAutoEncoderDataset(sentences)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    train_loss = DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=BASE_MODEL,
        tie_encoder_decoder=False,
    )

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs

    print(f"Steps per epoch: {steps_per_epoch:,} | Total steps: {total_steps:,}")
    print(f"Output directory: {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=WEIGHT_DECAY,
        scheduler=SCHEDULER,
        optimizer_params={"lr": LR},
        warmup_steps=WARMUP_STEPS,
        show_progress_bar=True,
        output_path=output_dir,
    )

    print(f"\nDomain-adapted model saved to: {output_dir}")


# ==========================================
# 5. Quick embedding sanity check
# ==========================================

def sanity_check(model_dir: str) -> None:
    """
    Loads the saved model and prints cosine similarities between sample
    therapy-domain sentence pairs to confirm the model is functional.

    Args:
        model_dir: Path to the saved SentenceTransformer model.
    """
    from sentence_transformers import util

    model = SentenceTransformer(model_dir)

    pairs = [
        ("I feel completely hopeless about everything.", "Nothing will ever get better for me."),
        ("I feel completely hopeless about everything.", "The weather is nice today."),
    ]

    print("\n--- Sanity check: cosine similarity ---")
    for a, b in pairs:
        emb_a = model.encode(a, convert_to_tensor=True)
        emb_b = model.encode(b, convert_to_tensor=True)
        sim = util.cos_sim(emb_a, emb_b).item()
        print(f"  {sim:.4f}  |  '{a[:50]}' vs '{b[:50]}'")


# ==========================================
# 6. Run
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSDAE domain adaptation on HOPE corpus.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--deletion-ratio",
        type=float,
        default=DELETION_RATIO,
        help=f"Token deletion ratio for TSDAE noise (default: {DELETION_RATIO}).",
    )
    parser.add_argument(
        "--no-val-test",
        action="store_true",
        help="Use Train utterances only (avoids any Val/Test text in the encoder).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to save the adapted model (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    splits = ["Train"] if args.no_val_test else ALL_SPLITS

    print(f"Loading utterances from splits: {splits}")
    sentences = load_utterances(splits)

    if not sentences:
        print("[ERROR] No sentences loaded. Check that preprocess.py has been run.")
        raise SystemExit(1)

    run_tsdae(
        sentences=sentences,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        deletion_ratio=args.deletion_ratio,
    )

    sanity_check(args.output_dir)
