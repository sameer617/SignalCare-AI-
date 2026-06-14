"""
temporal_analysis.py
------------------------
Week 3: Rolling-window temporal analysis module.

Implements the "Temporal Signal Modeling" deliverable from the project brief:
  - Rolling sentiment window
  - Drift detection logic -> S05_emotional_volatility... (S08, see below)
  - Volatility metric -> S05_emotional_volatility

Per CLAUDE.md Section 2 (End Goal), this module exposes an incremental
streaming interface (TemporalAnalyzer.update()) so it can later be called
turn-by-turn as live transcript lines arrive, in addition to being run in
batch over a full HOPE transcript.

Input: a sequence of per-patient-turn sentiment classifications (label +
confidence), e.g. from processed/<SPLIT>/utterance_sentiment.jsonl or the
Week 2 sentiment classifier (models/sentiment-v1-mlp-d05-wd3/).

Output per turn (TemporalSignals):
  - S08_negative_sentiment_trend: drift score (regression slope of the
    rolling window) + fired flag + confidence
  - S05_emotional_volatility: volatility score (std dev of turn-to-turn
    changes in the rolling window) + fired flag + confidence

This is decision-support only (CLAUDE.md Rule 5): outputs are signals +
confidences for a human counselor to review, never a diagnosis or crisis
verdict.

Usage:
    from temporal_analysis import TemporalAnalyzer, SentimentTurn

    analyzer = TemporalAnalyzer()
    for turn in sentiment_turns:
        signals = analyzer.update(turn)
        # signals.drift_score, signals.drift_fired, ...

CLI (runs synthetic escalation scenarios as a benchmark):
    python src/temporal_analysis.py
"""

import os
import sys
from collections import deque
from typing import Deque, List, Optional

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_taxonomy import RiskSignal


# ==========================================
# 1. Constants
# ==========================================

# Maps sentiment labels to a numeric score in [-1, 1]. Multiplied by the
# classifier's confidence to get a per-turn signed score.
SENTIMENT_SCORE_MAP = {
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
}

WINDOW_SIZE = 5         # number of recent patient turns considered
MIN_TURNS_FOR_SIGNAL = 3  # need at least this many turns before scoring

# S08: a window-average slope at or below this value indicates a worsening
# (more negative) sentiment trend.
DRIFT_THRESHOLD = -0.15

# S05: a standard deviation of turn-to-turn deltas at or above this value
# indicates emotional volatility. Set near the 80th percentile of
# volatility scores observed across real HOPE Validation transcripts
# (median ~0.55, p80 ~0.90) so S05 flags the most volatile ~20% of turns
# rather than ~half (see CLAUDE.md Section 9, 2026-06-14 tuning).
VOLATILITY_THRESHOLD = 0.9


# ==========================================
# 2. Schemas
# ==========================================

class SentimentTurn(BaseModel):
    """A single patient turn's sentiment classification."""
    sentiment: str = Field(description="One of 'negative', 'neutral', 'positive'.")
    confidence: float = Field(ge=0.0, le=1.0)

    def score(self) -> float:
        """
        Converts this turn's sentiment + confidence into a signed score.

        Returns:
            float in [-1, 1]: negative sentiment -> negative score,
            positive sentiment -> positive score, scaled by confidence.
        """
        return SENTIMENT_SCORE_MAP[self.sentiment] * self.confidence


class TemporalSignals(BaseModel):
    """Rolling-window temporal signals computed after a turn is added."""
    turn_index: int = Field(description="0-based index of the turn just processed.")
    window_size: int = Field(description="Number of turns actually used in the window (<= WINDOW_SIZE).")

    drift_score: Optional[float] = Field(
        default=None, description="Linear regression slope of sentiment scores over the window."
    )
    drift_signal: RiskSignal = Field(default=RiskSignal.S08_NEGATIVE_SENTIMENT_TREND)
    drift_fired: bool = Field(default=False)
    drift_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    volatility_score: Optional[float] = Field(
        default=None, description="Standard deviation of turn-to-turn sentiment score deltas over the window."
    )
    volatility_signal: RiskSignal = Field(default=RiskSignal.S05_EMOTIONAL_VOLATILITY)
    volatility_fired: bool = Field(default=False)
    volatility_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ==========================================
# 3. Rolling-window math
# ==========================================

def _linear_slope(values: List[float]) -> float:
    """
    Computes the slope of the best-fit line through (index, value) pairs.

    Args:
        values: Sequence of numeric values, in chronological order.

    Returns:
        The slope (rate of change per turn). 0.0 if fewer than 2 values
        or all x-values are identical (always false here since x = 0..n-1).
    """
    n = len(values)
    if n < 2:
        return 0.0

    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    denominator = sum((x - mean_x) ** 2 for x in xs)

    return numerator / denominator if denominator != 0 else 0.0


def _stdev(values: List[float]) -> float:
    """
    Computes the population standard deviation of a sequence of values.

    Args:
        values: Sequence of numeric values.

    Returns:
        Standard deviation. 0.0 if fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return 0.0

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return variance ** 0.5


# ==========================================
# 4. Incremental analyzer
# ==========================================

class TemporalAnalyzer:
    """
    Maintains a rolling window of per-turn sentiment scores for a single
    session and computes drift (S08) and volatility (S05) signals
    incrementally as new turns arrive.

    Designed to be called turn-by-turn (CLAUDE.md Section 2): construct one
    instance per session, then call .update() once per new patient turn as
    it becomes available (live transcript line, or replayed from a static
    transcript for batch evaluation).
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        """
        Args:
            window_size: Number of most recent patient turns to consider
                for the rolling window (default: WINDOW_SIZE).
        """
        self.window_size = window_size
        self._window: Deque[float] = deque(maxlen=window_size)
        self._turn_count = 0

    def update(self, turn: SentimentTurn) -> TemporalSignals:
        """
        Adds a new patient turn and recomputes the rolling-window signals.

        Args:
            turn: The new patient turn's sentiment classification.

        Returns:
            TemporalSignals with the updated drift (S08) and volatility
            (S05) scores, fired flags, and confidences.
        """
        self._window.append(turn.score())
        window_values = list(self._window)
        result = TemporalSignals(turn_index=self._turn_count, window_size=len(window_values))
        self._turn_count += 1

        if len(window_values) < MIN_TURNS_FOR_SIGNAL:
            return result

        # --- S05: emotional volatility ---
        deltas = [window_values[i] - window_values[i - 1] for i in range(1, len(window_values))]
        volatility = _stdev(deltas) if deltas else 0.0
        result.volatility_score = volatility
        if volatility >= VOLATILITY_THRESHOLD:
            result.volatility_fired = True
            result.volatility_confidence = min(1.0, volatility / VOLATILITY_THRESHOLD)

        # --- S08: negative sentiment trend (drift) ---
        # A sustained downward trend is distinct from volatile swings: an
        # oscillating window can produce a spuriously negative best-fit
        # slope from a single sharp dip, so drift only fires when the
        # window is not also flagged as volatile.
        slope = _linear_slope(window_values)
        result.drift_score = slope
        if slope <= DRIFT_THRESHOLD and not result.volatility_fired:
            result.drift_fired = True
            # Scale confidence by how far past the threshold the slope is,
            # capped at 1.0.
            result.drift_confidence = min(1.0, slope / DRIFT_THRESHOLD)

        return result

    def reset(self) -> None:
        """Clears the rolling window, starting a fresh session."""
        self._window.clear()
        self._turn_count = 0


# ==========================================
# 5. Batch helper
# ==========================================

def analyze_session(turns: List[SentimentTurn], window_size: int = WINDOW_SIZE) -> List[TemporalSignals]:
    """
    Runs the rolling-window analyzer over a full sequence of sentiment turns.

    Args:
        turns: Ordered list of per-patient-turn sentiment classifications
            for one session.
        window_size: Number of most recent turns to consider (default: WINDOW_SIZE).

    Returns:
        List of TemporalSignals, one per turn, in the same order as input.
    """
    analyzer = TemporalAnalyzer(window_size=window_size)
    return [analyzer.update(turn) for turn in turns]


# ==========================================
# 6. CLI: synthetic escalation benchmark
# ==========================================

def _print_session(name: str, turns: List[SentimentTurn]) -> List[TemporalSignals]:
    """
    Runs analyze_session and prints a per-turn summary table.

    Args:
        name: Scenario name, for display.
        turns: Ordered list of sentiment turns for the synthetic session.

    Returns:
        The list of TemporalSignals produced by analyze_session.
    """
    print(f"\n=== {name} ===")
    print(f"{'turn':>4} {'sentiment':>9} {'score':>6} {'drift':>7} {'S08':>5} {'volatility':>10} {'S05':>5}")

    results = analyze_session(turns)
    for turn, signals in zip(turns, results):
        drift_str = f"{signals.drift_score:+.2f}" if signals.drift_score is not None else "   -  "
        vol_str = f"{signals.volatility_score:.2f}" if signals.volatility_score is not None else "    -    "
        print(
            f"{signals.turn_index:>4} {turn.sentiment:>9} {turn.score():>+6.2f} "
            f"{drift_str:>7} {'YES' if signals.drift_fired else '.':>5} "
            f"{vol_str:>10} {'YES' if signals.volatility_fired else '.':>5}"
        )
    return results


if __name__ == "__main__":
    # Scenario 1: steady escalation (S08 should fire, S05 should not).
    escalation_turns = [
        SentimentTurn(sentiment="neutral", confidence=0.6),
        SentimentTurn(sentiment="neutral", confidence=0.5),
        SentimentTurn(sentiment="negative", confidence=0.6),
        SentimentTurn(sentiment="negative", confidence=0.7),
        SentimentTurn(sentiment="negative", confidence=0.8),
        SentimentTurn(sentiment="negative", confidence=0.9),
        SentimentTurn(sentiment="negative", confidence=0.95),
    ]

    # Scenario 2: volatile oscillation (S05 should fire, S08 should not).
    volatile_turns = [
        SentimentTurn(sentiment="positive", confidence=0.8),
        SentimentTurn(sentiment="negative", confidence=0.9),
        SentimentTurn(sentiment="positive", confidence=0.7),
        SentimentTurn(sentiment="negative", confidence=0.95),
        SentimentTurn(sentiment="positive", confidence=0.8),
        SentimentTurn(sentiment="negative", confidence=0.9),
    ]

    # Scenario 3: stable session (neither should fire).
    stable_turns = [
        SentimentTurn(sentiment="neutral", confidence=0.6),
        SentimentTurn(sentiment="neutral", confidence=0.55),
        SentimentTurn(sentiment="positive", confidence=0.5),
        SentimentTurn(sentiment="neutral", confidence=0.6),
        SentimentTurn(sentiment="neutral", confidence=0.5),
        SentimentTurn(sentiment="positive", confidence=0.55),
    ]

    esc_results = _print_session("Scenario 1: Steady escalation (expect S08)", escalation_turns)
    vol_results = _print_session("Scenario 2: Volatile oscillation (expect S05)", volatile_turns)
    stable_results = _print_session("Scenario 3: Stable session (expect neither)", stable_turns)

    print("\n=== Benchmark summary ===")
    print(f"Scenario 1 (escalation): S08 fired = {any(r.drift_fired for r in esc_results)} "
          f"(expected True), S05 fired = {any(r.volatility_fired for r in esc_results)} (expected False)")
    print(f"Scenario 2 (volatile):   S05 fired = {any(r.volatility_fired for r in vol_results)} "
          f"(expected True), S08 fired = {any(r.drift_fired for r in vol_results)} (expected False)")
    print(f"Scenario 3 (stable):     S08 fired = {any(r.drift_fired for r in stable_results)} "
          f"(expected False), S05 fired = {any(r.volatility_fired for r in stable_results)} (expected False)")
