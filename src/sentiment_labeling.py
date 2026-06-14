"""
sentiment_labeling.py
----------------------
Stage 3b of the SignalCare AI pipeline (runs alongside label.py).

Reads preprocessed utterances from processed/<SPLIT>/utterances.jsonl and
labels each utterance with a sentiment class (positive / negative / neutral)
plus a model-reported confidence score, using GPT-4o-mini via batched calls.

These labels are NOT ground truth — they are an LLM-generated proxy used to
train and evaluate the embedding-based sentiment classifiers (v1: base
embeddings, v2: domain-adapted embeddings). See CLAUDE.md Section 7, Rule 6.

Batching strategy mirrors label.py:
  - BATCH_SIZE utterances per API call
  - Exponential backoff retries on RateLimitError
  - Progress written once per split on completion

Output: processed/<SPLIT>/utterance_sentiment.jsonl
Each line:
  {
    "transcript_id": "Transcript_1",
    "patient_turn_id": 1,
    "text": "...",
    "sentiment": "negative",
    "sentiment_confidence": 0.9
  }

Usage:
  python src/sentiment_labeling.py --split Train
  python src/sentiment_labeling.py --split Validation
  python src/sentiment_labeling.py --split Test
"""

import os
import json
import time
import argparse
from enum import Enum
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ==========================================
# 1. Constants
# ==========================================

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_SPLITS = ["Train", "Validation", "Test"]

BATCH_SIZE   = 12   # utterances per API call
MAX_RETRIES  = 5    # max retry attempts on rate limit
BACKOFF_BASE = 2    # seconds; doubles each retry
BATCH_SLEEP  = 0.5  # seconds between batches to stay under 200K TPM


# ==========================================
# 2. LLM Setup
# ==========================================

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class UtteranceSentiment(BaseModel):
    """Sentiment label and confidence for a single utterance."""
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)


class BatchSentimentOutput(BaseModel):
    """Structured output for a batch of utterances."""
    results: List[UtteranceSentiment] = Field(
        description="List of sentiment results, one per utterance, in the same order as input."
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(BatchSentimentOutput)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert annotation assistant labeling the emotional sentiment of
patient utterances from therapy session transcripts.

You will receive a numbered list of patient utterances. For each utterance:
- Assign exactly one sentiment label: "positive", "negative", or "neutral".
- "positive" means the utterance expresses relief, hope, satisfaction, or other
  pleasant emotion.
- "negative" means the utterance expresses distress, sadness, frustration, fear,
  anger, or other unpleasant emotion.
- "neutral" means the utterance is factual, descriptive, or does not clearly
  express positive or negative emotion.
- Also provide a confidence score between 0 and 1 reflecting how clear-cut the
  sentiment is.

Return one result per utterance, in the SAME ORDER as the input.
"""),
    ("human", """
Utterances to label:
{utterances}

Return results as a list of sentiment + confidence pairs, one per utterance in order.
""")
])

chain = prompt | structured_llm


# ==========================================
# 3. Batch labeling with backoff
# ==========================================

def label_batch(utterances: List[str]) -> List[UtteranceSentiment]:
    """
    Sends a batch of utterances to GPT-4o-mini and returns sentiment labels.
    Retries with exponential backoff on RateLimitError.

    Args:
        utterances: List of utterance texts.

    Returns:
        List of UtteranceSentiment, one per utterance.
    """
    numbered = "\n".join([f"{i+1}. {u}" for i, u in enumerate(utterances)])
    fallback = [UtteranceSentiment(sentiment=Sentiment.NEUTRAL, confidence=0.0) for _ in utterances]

    for attempt in range(MAX_RETRIES):
        try:
            result = chain.invoke({"utterances": numbered})
            results = result.results
            while len(results) < len(utterances):
                results.append(UtteranceSentiment(sentiment=Sentiment.NEUTRAL, confidence=0.0))
            return results[:len(utterances)]
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = BACKOFF_BASE ** attempt
                print(f"\n  [RateLimit] Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                print(f"\n  [Error] {e}")
                return fallback

    print(f"\n  [Failed] Max retries exceeded for batch, returning neutral labels.")
    return fallback


# ==========================================
# 4. Main labeling pipeline
# ==========================================

def label_split(split: str, limit: int = None) -> None:
    """
    Labels all utterances for a given split and writes utterance_sentiment.jsonl.

    Args:
        split: One of 'Train', 'Validation', 'Test'.
        limit: If set, only label the first `limit` utterances and write to a
            separate `utterance_sentiment.sample.jsonl` file for sanity checks.
    """
    input_path  = os.path.join(REPO_ROOT, "processed", split, "utterances.jsonl")
    output_name = "utterance_sentiment.sample.jsonl" if limit else "utterance_sentiment.jsonl"
    output_path = os.path.join(REPO_ROOT, "processed", split, output_name)

    if not os.path.exists(input_path):
        print(f"[ERROR] utterances.jsonl not found at {input_path}")
        print(f"        Run preprocess.py --split {split} first.")
        return

    # Load utterances
    utterances = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            utterances.append(json.loads(line.strip()))

    if limit:
        utterances = utterances[:limit]

    print(f"\nLoaded {len(utterances)} utterances from {split} split.")
    print(f"Batch size: {BATCH_SIZE} | Estimated API calls: {len(utterances) // BATCH_SIZE + 1}")

    # Process in batches
    all_records = []
    batches = [utterances[i:i+BATCH_SIZE] for i in range(0, len(utterances), BATCH_SIZE)]

    for batch in tqdm(batches, desc=f"Sentiment labeling [{split}]"):
        texts = [u["text"] for u in batch]
        sentiments = label_batch(texts)

        for utt, sent in zip(batch, sentiments):
            all_records.append({
                "transcript_id"       : utt["transcript_id"],
                "patient_turn_id"     : utt["patient_turn_id"],
                "text"                : utt["text"],
                "sentiment"           : sent.sentiment.value,
                "sentiment_confidence": sent.confidence,
            })

        time.sleep(BATCH_SLEEP)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    counts = {s.value: 0 for s in Sentiment}
    for r in all_records:
        counts[r["sentiment"]] += 1

    print(f"\nDone.")
    print(f"  Total utterances labeled : {len(all_records)}")
    for label, count in counts.items():
        print(f"  {label:<10}: {count} ({count/len(all_records)*100:.1f}%)")
    print(f"  Output saved to          : {output_path}")


# ==========================================
# 5. Run
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label utterances with sentiment + confidence.")
    parser.add_argument(
        "--split",
        choices=ALL_SPLITS,
        default=None,
        help="Which split to label. Omit to label all splits."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only label the first N utterances (writes a .sample.jsonl file)."
    )
    args = parser.parse_args()

    splits = [args.split] if args.split else ALL_SPLITS

    for split in splits:
        print(f"\n{'='*50}")
        print(f"Sentiment labeling split: {split}")
        print(f"{'='*50}")
        label_split(split, limit=args.limit)
