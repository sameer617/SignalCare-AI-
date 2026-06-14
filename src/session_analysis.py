"""
session_analysis.py
------------------------
Week 3: Session-level aggregation for S09 (rumination patterns) and S12
(escalation risk composite), building on temporal_analysis.py (S05/S08).

Per CLAUDE.md Section 2, SessionAnalyzer exposes an incremental .update()
interface so it can be called turn-by-turn as live transcript lines arrive,
in addition to batch replay over a HOPE transcript.

Input per patient turn:
  - SentimentTurn (sentiment label + confidence) -> feeds TemporalAnalyzer
    (S05 emotional volatility, S08 negative sentiment trend).
  - DistressTurn (list of S01-S18 RiskSignal codes detected for that turn,
    e.g. from distress_signal_inference.classify_utterance) -> feeds S09
    and S12 below.

Signals computed per turn (SessionSignals):
  - S09_rumination_patterns: fires when the SAME distress signal code
    recurs across multiple turns within the rolling window without the
    window going clean (a stuck cognitive loop, per taxonomy.json notes).
  - S12_escalation_risk_composite: fires when multiple DISTINCT
    High/Medium-severity signals (from the distress turns, plus S05/S08
    from TemporalAnalyzer) co-occur within the rolling window. Per
    taxonomy.json notes, S12 never fires from a single signal alone.

This is decision-support only (CLAUDE.md Rule 5): outputs are signals +
confidences for a human counselor to review, never a diagnosis or crisis
verdict.

Usage:
    from session_analysis import SessionAnalyzer, SentimentTurn, DistressTurn

    analyzer = SessionAnalyzer()
    for sentiment_turn, distress_turn in zip(sentiment_turns, distress_turns):
        signals = analyzer.update(sentiment_turn, distress_turn)
        # signals.rumination_fired, signals.escalation_fired, ...

CLI (runs synthetic scenarios as a benchmark):
    python src/session_analysis.py
"""

import os
import sys
import json
from collections import deque
from typing import Deque, Dict, List

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_taxonomy import RiskSignal
from temporal_analysis import (
    TemporalAnalyzer,
    TemporalSignals,
    SentimentTurn,
    WINDOW_SIZE,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==========================================
# 1. Constants
# ==========================================

# S09: a signal code recurring in at least this many turns within the
# rolling window suggests a stuck cognitive loop (rumination).
RUMINATION_MIN_RECURRENCES = 3

# S12: at least this many DISTINCT High/Medium-severity signals
# co-occurring within the rolling window triggers the escalation composite.
# Per taxonomy.json notes, S12 must never fire from a single signal alone,
# so this must be >= 2. Set to 4 (rather than the taxonomy's "three or more"
# example) because Medium severity is the most common tier in the taxonomy
# (8 of 18 signals) -- with a 5-turn window, 3 distinct Medium-severity
# codes accumulate too easily for S12 to function as a rare "elevated
# overall risk" flag. 4 brought real-transcript fire rates from 25%/32% to
# 8%/2% on Validation/Transcript_13 and Transcript_9 (see CLAUDE.md Section 9,
# 2026-06-14 tuning), while transcripts with little co-occurring risk
# (Transcript_17, Transcript_6) stayed near 0% either way.
ESCALATION_MIN_SIGNALS = 4

# Severities that count toward S12. Low-severity signals (e.g. S03, S06,
# S09, S10, S14) do not contribute to the escalation composite on their own.
ESCALATION_SEVERITIES = {"High", "Medium"}


# ==========================================
# 2. Severity map (loaded from taxonomy.json, Rule 3: stays in sync with RiskSignal)
# ==========================================

def _load_severity_map() -> Dict[RiskSignal, str]:
    """
    Loads the severity for each signal from taxonomy.json.

    Returns:
        Dict mapping RiskSignal -> severity string ('Low', 'Medium', 'High').
    """
    taxonomy_path = os.path.join(REPO_ROOT, "taxonomy.json")
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    severity_map: Dict[RiskSignal, str] = {}
    for entry in data["signals"]:
        signal = RiskSignal(entry["enum_value"])
        severity_map[signal] = entry["severity"]
    return severity_map


SEVERITY_MAP = _load_severity_map()


# ==========================================
# 3. Schemas
# ==========================================

class DistressTurn(BaseModel):
    """A single patient turn's distress signal classifications (S01-S18)."""
    signals: List[RiskSignal] = Field(default_factory=list)


class SessionSignals(TemporalSignals):
    """Per-turn session-level signals: S05/S08 (inherited) + S09 + S12."""

    rumination_signal: RiskSignal = Field(default=RiskSignal.S09_RUMINATION_PATTERNS)
    rumination_fired: bool = Field(default=False)
    rumination_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rumination_codes: List[RiskSignal] = Field(
        default_factory=list, description="Signal codes recurring across turns in the window."
    )

    escalation_signal: RiskSignal = Field(default=RiskSignal.S12_ESCALATION_RISK_COMPOSITE)
    escalation_fired: bool = Field(default=False)
    escalation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    escalation_codes: List[RiskSignal] = Field(
        default_factory=list, description="Distinct High/Medium-severity signals co-occurring in the window."
    )


# ==========================================
# 4. Incremental analyzer
# ==========================================

class SessionAnalyzer:
    """
    Wraps TemporalAnalyzer (S05/S08) and adds S09 (rumination) + S12
    (escalation composite) over a rolling window of per-turn distress
    signal codes.

    Designed to be called turn-by-turn (CLAUDE.md Section 2): construct one
    instance per session, then call .update() once per new patient turn
    (live transcript line, or replayed from a static transcript).
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        """
        Args:
            window_size: Number of most recent patient turns to consider
                for the rolling window (default: WINDOW_SIZE).
        """
        self.window_size = window_size
        self._temporal = TemporalAnalyzer(window_size=window_size)
        self._distress_window: Deque[List[RiskSignal]] = deque(maxlen=window_size)

    def update(self, sentiment_turn: SentimentTurn, distress_turn: DistressTurn) -> SessionSignals:
        """
        Adds a new patient turn and recomputes all session-level signals.

        Args:
            sentiment_turn: The new patient turn's sentiment classification,
                for S05/S08 (see temporal_analysis.py).
            distress_turn: The new patient turn's S01-S18 distress signal
                classifications, for S09/S12.

        Returns:
            SessionSignals with S05, S08, S09, and S12 scores, fired flags,
            confidences, and (for S09/S12) the contributing signal codes.
        """
        temporal_result = self._temporal.update(sentiment_turn)
        self._distress_window.append(distress_turn.signals)

        result = SessionSignals(**temporal_result.model_dump())

        # --- S09: rumination patterns ---
        recurrence_counts: Dict[RiskSignal, int] = {}
        for turn_signals in self._distress_window:
            for signal in set(turn_signals):
                recurrence_counts[signal] = recurrence_counts.get(signal, 0) + 1

        recurring = [
            signal for signal, count in recurrence_counts.items()
            if count >= RUMINATION_MIN_RECURRENCES
        ]
        if recurring:
            result.rumination_fired = True
            result.rumination_codes = recurring
            max_count = max(recurrence_counts[s] for s in recurring)
            result.rumination_confidence = min(1.0, max_count / len(self._distress_window))

        # --- S12: escalation risk composite ---
        co_occurring: set = set()
        for turn_signals in self._distress_window:
            for signal in turn_signals:
                if SEVERITY_MAP.get(signal) in ESCALATION_SEVERITIES:
                    co_occurring.add(signal)

        # S05/S08 are Medium severity (taxonomy.json) and count toward the
        # composite when they fire on the current turn.
        if temporal_result.drift_fired:
            co_occurring.add(temporal_result.drift_signal)
        if temporal_result.volatility_fired:
            co_occurring.add(temporal_result.volatility_signal)

        if len(co_occurring) >= ESCALATION_MIN_SIGNALS:
            result.escalation_fired = True
            result.escalation_codes = sorted(co_occurring, key=lambda s: s.value)
            result.escalation_confidence = min(1.0, len(co_occurring) / (ESCALATION_MIN_SIGNALS + 2))

        return result

    def reset(self) -> None:
        """Clears all rolling state, starting a fresh session."""
        self._temporal.reset()
        self._distress_window.clear()


# ==========================================
# 5. Batch helper
# ==========================================

def analyze_session(
    sentiment_turns: List[SentimentTurn],
    distress_turns: List[DistressTurn],
    window_size: int = WINDOW_SIZE,
) -> List[SessionSignals]:
    """
    Runs the rolling-window session analyzer over a full session.

    Args:
        sentiment_turns: Ordered per-turn sentiment classifications.
        distress_turns: Ordered per-turn distress signal classifications,
            same length and order as sentiment_turns.
        window_size: Number of most recent turns to consider (default: WINDOW_SIZE).

    Returns:
        List of SessionSignals, one per turn, in the same order as input.
    """
    analyzer = SessionAnalyzer(window_size=window_size)
    return [
        analyzer.update(s_turn, d_turn)
        for s_turn, d_turn in zip(sentiment_turns, distress_turns)
    ]


# ==========================================
# 6. CLI: synthetic benchmark
# ==========================================

def _print_session(name: str, sentiment_turns: List[SentimentTurn], distress_turns: List[DistressTurn]) -> List[SessionSignals]:
    """
    Runs analyze_session and prints a per-turn summary table.

    Args:
        name: Scenario name, for display.
        sentiment_turns: Ordered per-turn sentiment classifications.
        distress_turns: Ordered per-turn distress signal classifications.

    Returns:
        The list of SessionSignals produced by analyze_session.
    """
    print(f"\n=== {name} ===")
    print(f"{'turn':>4} {'distress codes':<45} {'S09':>5} {'S12':>5}")

    results = analyze_session(sentiment_turns, distress_turns)
    for i, (d_turn, signals) in enumerate(zip(distress_turns, results)):
        codes = ", ".join(s.value.split("_")[0] for s in d_turn.signals) or "-"
        print(f"{i:>4} {codes:<45} {'YES' if signals.rumination_fired else '.':>5} {'YES' if signals.escalation_fired else '.':>5}")

    return results


if __name__ == "__main__":
    # Neutral sentiment turns throughout (isolate S09/S12 from S05/S08).
    neutral_turns = [SentimentTurn(sentiment="neutral", confidence=0.5) for _ in range(8)]

    # Scenario 1: rumination -- patient repeatedly returns to hopelessness
    # (S01) across turns, with unrelated content in between.
    rumination_turns = [
        DistressTurn(signals=[RiskSignal.S01_HOPELESSNESS_ESCALATION]),
        DistressTurn(signals=[]),
        DistressTurn(signals=[RiskSignal.S01_HOPELESSNESS_ESCALATION]),
        DistressTurn(signals=[RiskSignal.S06_SOCIAL_WITHDRAWAL_LANGUAGE]),
        DistressTurn(signals=[RiskSignal.S01_HOPELESSNESS_ESCALATION]),
        DistressTurn(signals=[]),
        DistressTurn(signals=[]),
        DistressTurn(signals=[]),
    ]

    # Scenario 2: escalation composite -- multiple distinct high/medium
    # severity signals co-occur within the window (no single repeated code).
    # Needs >= ESCALATION_MIN_SIGNALS (4) distinct High/Medium signals.
    escalation_turns = [
        DistressTurn(signals=[]),
        DistressTurn(signals=[RiskSignal.S01_HOPELESSNESS_ESCALATION]),
        DistressTurn(signals=[RiskSignal.S04_SELF_BLAME_AMPLIFICATION]),
        DistressTurn(signals=[RiskSignal.S07_HELPLESSNESS_LOSS_OF_AGENCY]),
        DistressTurn(signals=[RiskSignal.S16_TRAUMA_OR_ABUSE]),
        DistressTurn(signals=[]),
        DistressTurn(signals=[]),
        DistressTurn(signals=[]),
    ]

    # Scenario 3: isolated, non-recurring, non-co-occurring signals --
    # neither S09 nor S12 should fire.
    isolated_turns = [
        DistressTurn(signals=[RiskSignal.S14_SUBSTANCE_ABUSE]),
        DistressTurn(signals=[]),
        DistressTurn(signals=[]),
        DistressTurn(signals=[RiskSignal.S06_SOCIAL_WITHDRAWAL_LANGUAGE]),
        DistressTurn(signals=[]),
        DistressTurn(signals=[]),
        DistressTurn(signals=[RiskSignal.S03_ALL_OR_NOTHING_THINKING]),
        DistressTurn(signals=[]),
    ]

    rum_results = _print_session("Scenario 1: Rumination (expect S09)", neutral_turns, rumination_turns)
    esc_results = _print_session("Scenario 2: Escalation composite (expect S12)", neutral_turns, escalation_turns)
    iso_results = _print_session("Scenario 3: Isolated signals (expect neither)", neutral_turns, isolated_turns)

    print("\n=== Benchmark summary ===")
    print(f"Scenario 1 (rumination):  S09 fired = {any(r.rumination_fired for r in rum_results)} "
          f"(expected True), S12 fired = {any(r.escalation_fired for r in rum_results)} (expected False)")
    print(f"Scenario 2 (escalation):  S12 fired = {any(r.escalation_fired for r in esc_results)} "
          f"(expected True), S09 fired = {any(r.rumination_fired for r in esc_results)} (expected False)")
    print(f"Scenario 3 (isolated):    S09 fired = {any(r.rumination_fired for r in iso_results)} "
          f"(expected False), S12 fired = {any(r.escalation_fired for r in iso_results)} (expected False)")
