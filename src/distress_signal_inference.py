"""
distress_signal_inference.py
------------------------------
Reusable inference module for utterance-level distress signal (S01-S18)
classification via GPT-4o-mini structured output.

This is the inference-time counterpart to label.py: label.py batch-labels the
static HOPE dataset splits to produce LLM-generated reference labels
(processed/<SPLIT>/utterance_labels.jsonl, see CLAUDE.md Section 7 Rule 6 --
these are not gold labels). This module exposes the same classification as an
importable function for live use (e.g. a counselor-facing demo/prototype),
operating on arbitrary patient utterance text rather than the fixed dataset.

Unlike label.py (which hardcodes signal_confidence=0.8), this module asks the
model for a real per-signal confidence score, since a live demo benefits from
being able to threshold or rank signals by confidence.

This is decision-support only (CLAUDE.md Rule 5): output is a list of
signals + confidences for a human counselor to review, never a diagnosis or
crisis verdict.

Usage:
    from distress_signal_inference import classify_utterance, classify_utterances

    result = classify_utterance("I just don't see the point in trying anymore.")
    # result.signals -> [RiskSignal.S01_HOPELESSNESS_ESCALATION, ...]
    # result.signal_confidence -> {"S01_hopelessness_escalation": 0.85, ...}

    results = classify_utterances([text1, text2, ...])  # batched

CLI:
    python src/distress_signal_inference.py "I feel like nothing I do matters."
"""

import os
import sys
import time
from typing import Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Add src/ to path so risk_taxonomy imports work when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_taxonomy import RiskSignal

load_dotenv()


# ==========================================
# 1. Constants
# ==========================================

MODEL_NAME   = "gpt-4o-mini"
TEMPERATURE  = 0
MAX_RETRIES  = 5    # max retry attempts on rate limit
BACKOFF_BASE = 2    # seconds; doubles each retry
BATCH_SLEEP  = 0.5  # seconds between batches to stay under rate limits

VALID_LABELS = "\n".join([f"- {s.value}" for s in RiskSignal])


# ==========================================
# 2. Schema + LLM setup
# ==========================================

class SignalScore(BaseModel):
    """A single distress signal with its confidence score."""
    signal: RiskSignal
    confidence: float = Field(ge=0.0, le=1.0)


class UtteranceDistressSignals(BaseModel):
    """Distress signal classification result for a single utterance."""
    signals: List[SignalScore] = Field(
        default_factory=list,
        description="Distress signals detected in the utterance, each with a confidence score. "
                    "Empty if no signal applies.",
    )


class BatchDistressOutput(BaseModel):
    """Structured output for a batch of utterances."""
    results: List[UtteranceDistressSignals] = Field(
        description="List of distress signal results, one per utterance, in the same order as input."
    )


class DistressSignalResult(BaseModel):
    """Convenience view of UtteranceDistressSignals as a signal -> confidence map."""
    signals: List[RiskSignal]
    signal_confidence: Dict[str, float]


_llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
_structured_llm_single = _llm.with_structured_output(UtteranceDistressSignals)
_structured_llm_batch = _llm.with_structured_output(BatchDistressOutput)

_SYSTEM_PROMPT = f"""
You are an expert clinical-text annotation assistant labeling therapy utterances for linguistic and emotional distress signals.

For each patient utterance:
- Assign risk signal labels ONLY based on what the patient says in that utterance.
- Do NOT diagnose the patient.
- Do NOT infer beyond the text.
- Prefer precision over recall: only label signals with clear or moderate evidence.
- For each signal you assign, also provide a confidence score between 0 and 1
  reflecting how clear-cut the evidence is.
- If no signal applies, return an empty list of signals.

You MUST only use signals exactly as listed below:

{VALID_LABELS}
"""

_single_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", """
Patient utterance:
{utterance}

Classify the applicable distress signals with confidence scores.
"""),
])

_batch_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT + "\nYou will receive a numbered list of patient utterances. "
               "Return one result per utterance, in the SAME ORDER as the input."),
    ("human", """
Utterances to classify:
{utterances}

Return results as a list of distress signal classifications, one per utterance in order.
"""),
])

_single_chain = _single_prompt | _structured_llm_single
_batch_chain = _batch_prompt | _structured_llm_batch


# ==========================================
# 3. Single-utterance inference
# ==========================================

def classify_utterance(text: str) -> DistressSignalResult:
    """
    Classifies distress signals for a single patient utterance.

    Args:
        text: Patient utterance text. Should already be PII-redacted
            (CLAUDE.md Rule 4) before being sent here.

    Returns:
        DistressSignalResult with the detected signals and their confidences.
        Returns an empty result on repeated API failure.
    """
    fallback = DistressSignalResult(signals=[], signal_confidence={})

    for attempt in range(MAX_RETRIES):
        try:
            result = _single_chain.invoke({"utterance": text})
            return DistressSignalResult(
                signals=[s.signal for s in result.signals],
                signal_confidence={s.signal.value: s.confidence for s in result.signals},
            )
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = BACKOFF_BASE ** attempt
                print(f"\n  [RateLimit] Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                print(f"\n  [Error] {e}")
                return fallback

    print(f"\n  [Failed] Max retries exceeded, returning empty result.")
    return fallback


# ==========================================
# 4. Batch inference
# ==========================================

def classify_utterances(texts: List[str], batch_size: int = 12) -> List[DistressSignalResult]:
    """
    Classifies distress signals for a list of patient utterances, batched to
    reduce API calls.

    Args:
        texts: List of patient utterance texts. Should already be
            PII-redacted (CLAUDE.md Rule 4) before being sent here.
        batch_size: Number of utterances per API call.

    Returns:
        List of DistressSignalResult, one per input utterance, in order.
    """
    results: List[DistressSignalResult] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results.extend(_classify_batch(batch))
        if i + batch_size < len(texts):
            time.sleep(BATCH_SLEEP)

    return results


def _classify_batch(texts: List[str]) -> List[DistressSignalResult]:
    """
    Classifies a single batch of utterances. Retries with exponential backoff
    on RateLimitError.

    Args:
        texts: Batch of patient utterance texts.

    Returns:
        List of DistressSignalResult, one per input utterance, in order.
    """
    numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    fallback = [DistressSignalResult(signals=[], signal_confidence={}) for _ in texts]

    for attempt in range(MAX_RETRIES):
        try:
            output = _batch_chain.invoke({"utterances": numbered})
            items = output.results
            while len(items) < len(texts):
                items.append(UtteranceDistressSignals(signals=[]))
            return [
                DistressSignalResult(
                    signals=[s.signal for s in item.signals],
                    signal_confidence={s.signal.value: s.confidence for s in item.signals},
                )
                for item in items[:len(texts)]
            ]
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = BACKOFF_BASE ** attempt
                print(f"\n  [RateLimit] Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                print(f"\n  [Error] {e}")
                return fallback

    print(f"\n  [Failed] Max retries exceeded for batch, returning empty results.")
    return fallback


# ==========================================
# 5. CLI
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/distress_signal_inference.py \"<utterance text>\"")
        sys.exit(1)

    utterance_text = " ".join(sys.argv[1:])
    classification = classify_utterance(utterance_text)

    print(f"\nUtterance: {utterance_text}")
    if not classification.signals:
        print("No distress signals detected.")
    else:
        print("Detected signals:")
        for signal in classification.signals:
            confidence = classification.signal_confidence[signal.value]
            print(f"  - {signal.value}: {confidence:.2f}")
