"""
label.py
--------
Stage 3 of the SignalCare AI pipeline.

Reads preprocessed utterances from processed/<SPLIT>/utterances.jsonl and
labels each utterance with risk signals using GPT-4o-mini via batched calls.

Batching strategy:
  - BATCH_SIZE utterances are sent per API call (reduces requests ~10x)
  - Exponential backoff retries on RateLimitError
  - Progress saved incrementally so a failed run can be resumed

Output: processed/<SPLIT>/utterance_labels.jsonl
Each line:
  {
    "transcript_id": "Transcript_1",
    "patient_turn_id": 1,
    "text": "...",
    "distress_signals": ["S01_hopelessness_escalation", ...],
    "signal_confidence": {"S01_hopelessness_escalation": 0.8, ...}
  }

Usage:
  python src/label.py --split Train
  python src/label.py --split Validation
  python src/label.py --split Test
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict

from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Add src/ to path so risk_taxonomy imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_taxonomy import RiskSignal

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

class BatchClassificationOutput(BaseModel):
    """Structured output for a batch of utterances."""
    results: List[List[RiskSignal]] = Field(
        description="List of risk signal lists, one per utterance, in the same order as input."
    )


VALID_LABELS = "\n".join([f"- {s.value}" for s in RiskSignal])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(BatchClassificationOutput)

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
You are an expert clinical-text annotation assistant labeling therapy utterances for linguistic and emotional distress signals.

You will receive a numbered list of patient utterances. For each utterance:
- Assign risk signal labels ONLY based on what the patient says in that utterance.
- Do NOT diagnose the patient.
- Do NOT infer beyond the text.
- Prefer precision over recall: only label signals with clear or moderate evidence.
- If no signal applies, return an empty list for that utterance.

Return one list of labels per utterance, in the SAME ORDER as the input.
You MUST only use labels exactly as listed below:

{VALID_LABELS}
"""),
    ("human", """
Utterances to label:
{utterances}

Return results as a list of lists, one per utterance in order.
""")
])

chain = prompt | structured_llm


# ==========================================
# 3. Batch labeling with backoff
# ==========================================

def label_batch(utterances: List[str]) -> List[List[str]]:
    """
    Sends a batch of utterances to GPT-4o-mini and returns signal lists.
    Retries with exponential backoff on RateLimitError.

    Args:
        utterances: List of utterance texts.

    Returns:
        List of signal lists (one per utterance).
    """
    numbered = "\n".join([f"{i+1}. {u}" for i, u in enumerate(utterances)])

    for attempt in range(MAX_RETRIES):
        try:
            result = chain.invoke({"utterances": numbered})
            # Pad with empty lists if model returns fewer results than expected
            labels = result.results
            while len(labels) < len(utterances):
                labels.append([])
            return [[sig.value for sig in sig_list] for sig_list in labels[:len(utterances)]]
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = BACKOFF_BASE ** attempt
                print(f"\n  [RateLimit] Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                print(f"\n  [Error] {e}")
                return [[] for _ in utterances]

    print(f"\n  [Failed] Max retries exceeded for batch, returning empty labels.")
    return [[] for _ in utterances]


# ==========================================
# 4. Main labeling pipeline
# ==========================================

def label_split(split: str) -> None:
    """
    Labels all utterances for a given split and writes utterance_labels.jsonl.

    Args:
        split: One of 'Train', 'Validation', 'Test'.
    """
    input_path  = os.path.join(REPO_ROOT, "processed", split, "utterances.jsonl")
    output_path = os.path.join(REPO_ROOT, "processed", split, "utterance_labels.jsonl")

    if not os.path.exists(input_path):
        print(f"[ERROR] utterances.jsonl not found at {input_path}")
        print(f"        Run preprocess.py --split {split} first.")
        return

    # Load utterances
    utterances = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            utterances.append(json.loads(line.strip()))

    print(f"\nLoaded {len(utterances)} utterances from {split} split.")
    print(f"Batch size: {BATCH_SIZE} | Estimated API calls: {len(utterances) // BATCH_SIZE + 1}")

    # Process in batches
    all_records = []
    batches = [utterances[i:i+BATCH_SIZE] for i in range(0, len(utterances), BATCH_SIZE)]

    for batch in tqdm(batches, desc=f"Labeling [{split}]"):
        texts = [u["text"] for u in batch]
        signal_lists = label_batch(texts)

        for utt, signals in zip(batch, signal_lists):
            all_records.append({
                "transcript_id"  : utt["transcript_id"],
                "patient_turn_id": utt["patient_turn_id"],
                "text"           : utt["text"],
                "distress_signals": signals,
                "signal_confidence": {s: 0.8 for s in signals}
            })

        time.sleep(BATCH_SLEEP)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    labeled = sum(1 for r in all_records if r["distress_signals"])
    print(f"\nDone.")
    print(f"  Total utterances labeled : {len(all_records)}")
    print(f"  Utterances with signals  : {labeled} ({labeled/len(all_records)*100:.1f}%)")
    print(f"  Output saved to          : {output_path}")


# ==========================================
# 5. Run
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label utterances with distress signals.")
    parser.add_argument(
        "--split",
        choices=ALL_SPLITS,
        default=None,
        help="Which split to label. Omit to label all splits."
    )
    args = parser.parse_args()

    splits = [args.split] if args.split else ALL_SPLITS

    for split in splits:
        print(f"\n{'='*50}")
        print(f"Labeling split: {split}")
        print(f"{'='*50}")
        label_split(split)
