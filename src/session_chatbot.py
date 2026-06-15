"""
session_chatbot.py
-------------------
Conversational Q&A over a single therapy session's analysis results.

Builds a per-session context string from a session's analysis.json records
(per-turn redacted transcript, sentiment, and S05/S08/S09/S12 signals) plus
the taxonomy.json signal definitions, and answers free-form counselor
questions about that session via GPT-4o-mini, with multi-turn conversation
history.

This is decision-support only (CLAUDE.md Rule 5): the assistant answers
questions using the session's signals + transcript, and must never produce a
clinical diagnosis or crisis verdict -- it defers that judgment to the human
counselor.

Usage:
    from session_chatbot import build_session_context, answer_question

    context = build_session_context(records, session_title, taxonomy)
    reply = answer_question(context, history, "When did volatility spike?")
    # history is a list of (role, content) tuples, role in {"user", "assistant"}
"""

import os
import sys
import time
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# ==========================================
# 1. Constants
# ==========================================

MODEL_NAME   = "gpt-4o-mini"
TEMPERATURE  = 0
MAX_RETRIES  = 5    # max retry attempts on rate limit
BACKOFF_BASE = 2    # seconds; doubles each retry

SESSION_SIGNAL_CODES = ["S05", "S08", "S09", "S12"]

FALLBACK_REPLY = (
    "Sorry, I couldn't process that question right now. Please try again."
)


# ==========================================
# 2. LLM setup
# ==========================================

_llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

_SYSTEM_PROMPT = """
You are an assistant embedded in SignalCare AI, a decision-support dashboard
that helps counselors review a therapy session's transcript and automatically
detected distress signals.

You will be given the full redacted transcript of one therapy session, along
with per-turn sentiment scores and any distress signals (S01-S18) or
session-level signals (S05, S08, S09, S12) that were detected, plus the
taxonomy definitions for those signal codes.

Answer the counselor's questions using ONLY the session data provided below.

Rules:
- This is decision support, not diagnosis. NEVER state or imply a clinical
  diagnosis, and NEVER issue a crisis verdict ("this patient is at risk of...").
  Describe what the data shows (signals, sentiment, transcript content) and
  defer clinical judgment to the counselor.
- If asked something the session data doesn't cover, say so rather than
  guessing.
- Reference turn numbers when pointing to specific moments in the session.
- Keep answers concise and grounded in the provided data.

--- SESSION DATA ---
{session_data}
--- END SESSION DATA ---
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])

_chain = _prompt | _llm


# ==========================================
# 3. Context building
# ==========================================

def build_session_context(records: List[dict], session_title: str, taxonomy: dict) -> str:
    """
    Builds the session-data block for the chatbot's system prompt from a
    session's analysis.json records and the signal taxonomy.

    Args:
        records: Per-turn analysis records, as written by
            app.pipeline.run_video_pipeline.run_pipeline (analysis.json).
        session_title: The session's title, for context.
        taxonomy: Mapping of signal code -> {name, definition, severity},
            as returned by app.routers.dashboard._load_taxonomy.

    Returns:
        A string describing the session's transcript, sentiment, and fired
        signals, followed by a legend of the relevant taxonomy definitions.
    """
    lines = [f"Session title: {session_title}", "", "Transcript (patient turns):"]

    all_codes = set()
    for record in records:
        turn_id = record["turn_id"]
        text = record["text"]
        sentiment = record["sentiment"]
        confidence = record["sentiment_confidence"]

        turn_line = f"Turn {turn_id}: \"{text}\" [sentiment: {sentiment} ({confidence:.2f})]"

        fired_codes = []
        for code in record.get("distress_signals", []):
            short_code = code.split("_")[0]
            fired_codes.append(short_code)
            all_codes.add(short_code)

        if record.get("volatility_fired"):
            fired_codes.append("S05")
            all_codes.add("S05")
        if record.get("drift_fired"):
            fired_codes.append("S08")
            all_codes.add("S08")
        if record.get("rumination_fired"):
            fired_codes.append("S09")
            all_codes.add("S09")
        if record.get("escalation_fired"):
            fired_codes.append("S12")
            all_codes.add("S12")

        if fired_codes:
            turn_line += f" [signals: {', '.join(fired_codes)}]"

        lines.append(turn_line)

    lines.append("")
    lines.append("Signal definitions:")
    for code in sorted(all_codes):
        entry = taxonomy.get(code)
        if entry:
            lines.append(f"{code} ({entry['name']}, severity {entry['severity']}): {entry['definition']}")

    return "\n".join(lines)


# ==========================================
# 4. Question answering
# ==========================================

def answer_question(context: str, history: List[Tuple[str, str]], question: str) -> str:
    """
    Answers a counselor's question about a session, using the session's
    context and prior conversation history.

    Args:
        context: Session-data block from build_session_context.
        history: Prior messages as (role, content) tuples, role in
            {"user", "assistant"}, oldest first.
        question: The counselor's new question.

    Returns:
        The assistant's reply text. Returns FALLBACK_REPLY on repeated
        API failure.
    """
    history_messages = [(role, content) for role, content in history]

    for attempt in range(MAX_RETRIES):
        try:
            result = _chain.invoke({
                "session_data": context,
                "history": history_messages,
                "question": question,
            })
            return result.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = BACKOFF_BASE ** attempt
                print(f"\n  [RateLimit] Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                print(f"\n  [Error] {e}")
                return FALLBACK_REPLY

    print(f"\n  [Failed] Max retries exceeded, returning fallback reply.")
    return FALLBACK_REPLY
