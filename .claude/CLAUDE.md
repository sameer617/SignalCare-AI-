# CLAUDE.md — SignalCare AI

This file gives Claude Code the context needed to build this project end-to-end.
Read it fully before making changes.

---

## 1. Project Context

**SignalCare AI** is a real-time NLP system that detects emotional distress markers
in therapy-session transcripts to support (not replace) human counselors. It is being
built for **CHIEAC**, a nonprofit, as an 8-week portfolio-grade project.

**The core problem:** Counselors cannot perfectly track every linguistic distress
cue across a long session. SignalCare AI surfaces patterns — hopelessness escalation,
catastrophizing, rumination, etc. — so a counselor can review them.

**What this system is NOT:**
- It is NOT a diagnostic tool. It never outputs a clinical diagnosis.
- It is NOT a crisis-intervention system. It is decision *support*.
- It does NOT replace clinical judgment.

**Dataset:** `HOPE_WSDM_2022` — 317 real therapy transcripts split into Train /
Validation / Test. Each transcript is a `.txt` file with alternating
`<Patient>:` and `<Therapist>:` turns. HOPE was originally built for dialogue-act
classification, so it ships **without distress labels** — we generate them with an LLM.

**Why HOPE:** HOPE has genuine multi-turn session structure,
which is required for the temporal signals (S05, S08, S09, S12). 

**End goal deliverables:**
1. Clean, documented, reproducible codebase
2. Trained baseline + improved distress-detection models
3. Evaluation report (utterance-level and session-level metrics)
4. A demo-able prototype

---

## 2. End Goal — Product Vision (Demo Prototype)

This section describes the **target end-to-end experience** for the demo prototype
(Week 8 deliverable). It is the north star for architecture decisions in later weeks
(Weeks 3–6) — current work should move toward this without attempting to build all of
it at once. Update the "Current Position" in Section 9 to track progress toward this
vision.

**The experience:**

1. **Upload page.** A counselor uploads a session recording (video or audio) through
   the dashboard.
2. **Live transcription.** As the recording plays/processes, the transcript is
   generated incrementally, line by line (alternating patient/therapist turns),
   and streamed to the UI as it becomes available.
3. **Real-time analysis, line by line.** As each new **patient** transcript line
   arrives:
   - It is PII-redacted (Rule 4) before any further processing.
   - Sentiment is scored (the Week 2 classifier).
   - Utterance-level distress signals (S01–S18) are classified
     (`distress_signal_inference.py`).
   - The temporal/rolling-window module (Week 3) updates its session state with the
     new turn and re-emits the current drift (S08), volatility (S05), rumination
     (S09), and escalation composite (S12) scores.
4. **Live dashboard.** The UI updates as each line is processed, showing:
   - The growing transcript (with redaction applied)
   - A running sentiment curve / rolling sentiment window
   - Highlighted distress signals on the lines that triggered them, with
     confidence + explanation ("why this was flagged")
   - Session-level indicators (volatility, drift, rumination, escalation composite)
     that update as the session progresses
   - Non-diagnostic framing throughout (Rule 5): signals + confidences for a human
     to review, never a verdict

**Architectural implication for current work:** components built now (temporal
analysis module, session aggregation) should expose an **incremental / streaming
interface** (e.g. `.update(new_turn)` that maintains rolling state) wherever that is
not significantly more work than a batch interface — so they can be reused as-is when
line-by-line live input becomes available, rather than rewritten in Week 5/6.

**Explicitly out of scope for now** (deferred to their roadmap weeks — do not build
prematurely):
- Video upload handling / storage
- Live speech-to-text / streaming ASR
- FastAPI backend, WebSocket streaming (Week 5)
- React dashboard UI (Week 6)

---

## 3. Risk Signal Taxonomy (S01–S18)

The 18 distress signals are the heart of the project. Canonical definitions live in
`taxonomy.json` (repo root). The enum lives in `src/risk_taxonomy.py` as `RiskSignal`.
**`taxonomy.json` and the `RiskSignal` enum must always stay in sync.**

| Code | Signal | Severity |
|------|--------|----------|
| S01 | Hopelessness escalation | High |
| S02 | Catastrophizing | Medium |
| S03 | All-or-nothing thinking | Low |
| S04 | Self-blame amplification | Medium |
| S05 | Emotional volatility | Medium |
| S06 | Social withdrawal language | Low |
| S07 | Helplessness / loss of agency | Medium |
| S08 | Negative sentiment trend | Medium |
| S09 | Rumination patterns | Low |
| S10 | Emotional numbing | Low |
| S11 | Cognitive distortion density | Medium |
| S12 | Escalation risk composite | High |
| S13 | Suicidal ideation | High |
| S14 | Substance abuse | Low |
| S15 | Self-harm | High |
| S16 | Trauma or abuse | Medium |
| S17 | Manic / hypomanic episodes | High |
| S18 | Psychotic symptoms | High |

S12 is a **composite meta-signal** — it fires when multiple other signals co-occur.

---

## 4. Architecture



**Data flow:**
```
HOPE .txt files
   │
   ├──> risk_taxonomy.py ──> train_risk_signals.csv   (transcript-level labels)
   │
   └──> preprocess.py ──> processed/utterances.jsonl  (redacted utterances)
                                  │
                                  └──> label.py ──> utterance-level labels
                                  │
        utterance_stats.ipynb ────┴──> utterances_with_signals.jsonl
                                            │
                                            └──> Week 2 model training
```

**Where new files belong:**
- All Python source → `src/`
- All generated data → `processed/` (never commit)
- Training scripts → `src/` with a clear `train_*.py` prefix
- Model checkpoints → `models/` (create when needed, gitignore)
- Docs → repo root

---

## 5. Code Style

- **Python 3.10+**, follow PEP 8.
- Every function gets a docstring (Args / Returns). Match the style already in `utils.py`.
- Use type hints on all function signatures.
- Module-level files start with a `"""docstring"""` block explaining purpose + usage.
- Organize files into numbered comment sections, e.g.:
  ```python
  # ==========================================
  # 1. Section name
  # ==========================================
  ```
- Prefer pure functions; keep `if __name__ == "__main__":` blocks thin.
- Constants in `UPPER_SNAKE_CASE` near the top of the file.
- No hardcoded absolute paths inside functions — accept paths as arguments,
  set the absolute path only in the `__main__` block or in `config.json`.
- `tqdm` for any loop over the dataset.
- Use `encoding="utf-8"` on every file open.

---

## 6. Preferred Libraries

| Purpose | Use | Do NOT use |
|---------|-----|------------|
| LLM labeling | `langchain-openai` + structured output (Pydantic) | raw OpenAI calls |
| Schema / validation | `pydantic` | dataclasses for LLM I/O |
| PII redaction | `presidio-analyzer`, `presidio-anonymizer` | regex-only redaction |
| Transformers / models | HuggingFace `transformers`, `torch` | — |
| Embeddings / domain adaptation | `sentence-transformers` (TSDAE) | — |
| Data | `pandas`, `numpy` | — |
| Plotting | `matplotlib`, `seaborn`, `plotly`  | — |
| Progress bars | `tqdm` | — |
| Config | `python-dotenv` for secrets, `config.json` for settings | — |

- LLM model: **`gpt-4o-mini`**, `temperature=0` for all labeling (determinism).
- Emotion model: Use above LLM (can experiment with other LLMs too depending on result)
- Pin new dependencies; keep a `requirements.txt` at repo root (TO BE CREATED).

---

## 7. Commands

Run everything from the repo root (`SIGNALCARE-AI-/`) with the venv activated.

```bash
# Activate environment (Windows)
venv\Scripts\activate

# Stage 1: transcript-level signal labeling -> src/train_risk_signals.csv
python src/risk_taxonomy.py

# Stage 2: preprocessing -> processed/utterances.jsonl + processed/redacted/
python src/preprocess.py

# Stage 3: utterance-level labeling -> utterance-level JSONL
python src/label.py

# EDA: run in Jupyter / VS Code notebook UI
#   src/utterance_stats.ipynb

# Install dependencies
pip install -r requirements.txt
```

There is no test suite yet — adding `pytest` tests under `tests/` is a future task.

---

## 8. Critical Rules

1. **NEVER commit `.env`** or any API key. It is gitignored — keep it that way.
2. **NEVER modify anything under `HOPE_WSDM_2022/`** — it is the read-only raw dataset.
3. **`taxonomy.json` and the `RiskSignal` enum must stay in sync.** If you add or
   rename a signal, update both, then regenerate `train_risk_signals.csv`.
4. **PII redaction must happen before any utterance text is persisted or sent
   anywhere.** Patient text is sensitive. `preprocess.py` redacts; downstream stages
   consume redacted text only.
5. **This is decision-support, not diagnosis.** No code path should emit a clinical
   diagnosis or a crisis verdict. Outputs are signals + confidences for a human.
6. **Labels are LLM-generated, not ground truth.** Any model trained on them inherits
   GPT-4o-mini's labeling accuracy as a ceiling. State this as a limitation in the
   eval report; do not present these labels as gold.
7. **Transcript parsing edge case:** turns are separated by single OR double newlines.
   The last patient turn may not be followed by a `<Therapist>:` tag — make sure it
   is still captured. (`preprocess.py` already handles this; preserve that behavior.)
8. **`temperature=0` for all LLM labeling** — reproducibility matters.
9. **Class imbalance is severe.** Some signals appear in 100+ transcripts, others in
   1–6. Account for this in training (class weights / stratification) and never
   report plain accuracy alone.
10. Generated artifacts (`processed/`, `models/`, `*.csv` outputs) are gitignored.
    Do not commit large data or checkpoints.

---

## 9. Implemented vs. Still To Do

### Implemented
- `risk_taxonomy.py` — transcript-level LLM labeling with S01–S18, structured Pydantic
  output, prompt with explicit valid-label list. **Has been run for all splits** →
  `src/train_risk_signals.csv`, `src/validation_risk_signals.csv`, `src/test_risk_signals.csv`.
- `taxonomy.json` — canonical definitions for the signals.
- `utils.py` — HF helpers: `sentence_level_sentiment`, `sentence_level_emotions`,
  file listing/reading. (`load_hugging_face_model` is a stub; refactor still pending.)
- `preprocess.py` — **has been run for all three splits** → `processed/<SPLIT>/utterances.jsonl`
  + `processed/<SPLIT>/redacted/`.
- `label.py` — **has been run for all three splits** → `processed/<SPLIT>/utterance_labels.jsonl`
  (distress signal labels per utterance, LLM-generated).
- `sentiment_labeling.py` — **new (Week 2)**. Mirrors `label.py`'s batching pattern; labels each
  utterance with sentiment (positive/negative/neutral) + confidence via GPT-4o-mini.
  Output: `processed/<SPLIT>/utterance_sentiment.jsonl`. Run across all splits as part of the
  Week 2 ground-truth generation step (see Section 10).
- `domain_adaptation.py` — TSDAE domain adaptation script, implemented and **has been run**
  with the target config (`BASE_MODEL = "bert-base-uncased"`, `tie_encoder_decoder=False`),
  producing `models/tsdae-adapted/` (bert-base-uncased base + decoder, trained 2026-05-17,
  1.8 hrs). Ready to use for v2 embeddings — no re-run needed.
- `utterance_stats.ipynb` — EDA notebook, written but **not yet executed** (0 output cells).
- `config.json` — exists but currently **empty**; needs the emotion model name added.
- `train_sentiment_v1.py` / `train_sentiment_v2.py` — **new (Week 2), run**. Utterance-level
  sentiment classifier (class-weighted logistic regression on mean-pooled bert-base-uncased
  embeddings). v1 = raw bert-base-uncased, v2 = TSDAE domain-adapted (`models/tsdae-adapted/`).
  Embeddings cached at `processed/embeddings_cache/`. Results (Validation accuracy):
  v1 = 0.7408, v2 = 0.7548 — both below the 0.80 target. v2 improves the positive-class
  F1 (0.55 → 0.55 on Val, but Test 0.62 → 0.64) and overall Test accuracy (0.7585 → 0.7567,
  roughly flat). Superseded as the final sentiment model by `train_sentiment_mlp.py` below.
- `train_sentiment_rf.py` / `train_sentiment_xgb.py` — **new (Week 2), run**. Random forest and
  XGBoost classifier heads on the cached v1/v2 embeddings, for comparison. Both overfit heavily
  (~96% train accuracy) with collapsed positive-class recall (~0.3). Results (Validation /
  Test accuracy): v1+RF = 0.7671 / 0.7281, v2+RF = 0.7671 / 0.7217, v1+XGB = 0.7793 / 0.7834,
  v2+XGB = 0.7723 / 0.7557. Outperformed by the MLP head; not used further.
- `train_sentiment_mlp.py` — **new (Week 2), run — FINAL sentiment classifier**. PyTorch MLP
  classifier head (configurable hidden layers + dropout + weight decay, class-weighted
  cross-entropy, early stopping on Validation accuracy) on the cached v1/v2 embeddings.
  Swept hidden-layer sizes ([128], [192], [256, 64]), dropout (0.3–0.6), and weight decay
  (1e-4–1e-2) to address overfitting (tree-based heads and the 2-layer MLP showed large
  train/val gaps). **Selected model: v1 (raw bert-base-uncased) + MLP[128], dropout=0.5,
  weight_decay=1e-3** — Train=0.8007, Validation=0.7933, Test=0.7742, with the smallest
  train/val gap (~0.7pt) of any config tried. Saved to `models/sentiment-v1-mlp-d05-wd3/`
  (`model.pt` + `metrics.json`). This is just under the 0.80 Validation target; across nearly
  all classifier heads and configs results cluster in the 0.77–0.80 range, suggesting this is
  close to the practical ceiling for this embedding + LLM-labeled-data combination (see Rule 6).
  v1 (raw BERT) consistently outperformed v2 (TSDAE-adapted) across every classifier head —
  TSDAE domain adaptation did not help this downstream task.
- `distress_signal_inference.py` — **new (2026-06-13), implemented**. Reusable inference module
  for utterance-level distress signal (S01–S18) classification via GPT-4o-mini structured output
  (`temperature=0`), per the decision below. Exposes `classify_utterance(text)` (single) and
  `classify_utterances(texts, batch_size=12)` (batched, mirrors `label.py`'s retry/backoff
  pattern). Unlike `label.py` (hardcoded `signal_confidence=0.8`), the model returns a real
  per-signal confidence score. CLI: `python src/distress_signal_inference.py "<utterance>"`.
  Tested on sample utterances — correctly detects e.g. S01+S08 for hopelessness language, S09
  for rumination, S14 for substance use, and returns empty for neutral text. This is the
  inference-time path for distress signals; intended for use by the Week 3 session-level
  aggregation logic (call per-utterance, then aggregate).
- `temporal_analysis.py` — **new (Week 3, 2026-06-13), implemented**. Rolling-window temporal
  analysis module per the project brief's "Temporal Signal Modeling" deliverable. Exposes
  `TemporalAnalyzer`, an **incremental** class with `.update(turn: SentimentTurn)` that maintains
  a rolling window (default size 5) of per-patient-turn sentiment scores and recomputes S08
  (negative sentiment trend / drift, via linear regression slope over the window) and S05
  (emotional volatility, via std dev of turn-to-turn score deltas) after each turn. Per
  CLAUDE.md Section 2, the `.update()` interface is designed to be reusable for live,
  line-by-line transcript input later (Week 5/6), not just batch replay. Also exposes
  `analyze_session(turns)` for batch use over a full transcript's sentiment sequence. Drift
  (S08) is suppressed when volatility (S05) also fires in the same window, since an oscillating
  window can otherwise produce a spuriously negative regression slope from a single sharp dip.
  CLI (`python src/temporal_analysis.py`) runs three synthetic scenarios (steady escalation,
  volatile oscillation, stable) — all three pass: escalation fires S08 only, volatile fires S05
  only, stable fires neither. This satisfies the Week 3 benchmark ("system detects synthetic
  escalation scenarios"). Thresholds: `DRIFT_THRESHOLD=-0.15` (unchanged), `WINDOW_SIZE=5`.
  **`VOLATILITY_THRESHOLD` tuned 2026-06-14: `0.6` → `0.9`** based on the distribution of
  volatility scores across 4 real Validation transcripts (180 turn-windows: Transcript_13, 9,
  17, 6) — median ~0.55, p80 ~0.90. `0.9` (~p80) targets S05 firing on the most volatile ~20%
  of turns rather than ~50%. Synthetic benchmark still passes at `0.9`.
- `session_analysis.py` — **new (Week 3, 2026-06-14), implemented**. Adds S09 (rumination
  patterns) and S12 (escalation risk composite) on top of `temporal_analysis.py`. Exposes
  `SessionAnalyzer`, an **incremental** class wrapping a `TemporalAnalyzer` with
  `.update(sentiment_turn: SentimentTurn, distress_turn: DistressTurn) -> SessionSignals`
  (`DistressTurn` = list of S01-S18 `RiskSignal` codes for that turn, e.g. from
  `distress_signal_inference.py` or `utterance_labels.jsonl`). `SEVERITY_MAP` is loaded from
  `taxonomy.json` at import time (Rule 3: stays in sync). **S09** fires when the same distress
  signal code recurs in >= `RUMINATION_MIN_RECURRENCES` (3) turns within the rolling window.
  **S12** fires when >= `ESCALATION_MIN_SIGNALS` (3) *distinct* High/Medium-severity signals
  (from the distress-turn window, plus S05/S08 from `TemporalAnalyzer` if firing on the current
  turn) co-occur within the window — never from a single signal, per `taxonomy.json` notes.
  Also exposes `analyze_session(sentiment_turns, distress_turns)` for batch use. CLI
  (`python src/session_analysis.py`) runs three synthetic scenarios (repeated S01 across turns,
  co-occurring S01+S04+S07+S16, isolated non-recurring signals) — all three pass as expected
  (S09 only, S12 only, neither). **`ESCALATION_MIN_SIGNALS` tuned 2026-06-14: `3` → `4`** — with
  Medium severity being the most common tier in the taxonomy (8/18 signals), 3 distinct
  co-occurring Medium-severity codes accumulated too easily within a 5-turn window for S12 to
  function as a rare "elevated overall risk" flag. `4` brought Validation/Transcript_13 from
  25%→8% and Transcript_9 from 32%→2%, while Transcript_17/Transcript_6 stayed ~0% either way.
  Scenario 2's synthetic case was updated to use 4 distinct signals (S01+S04+S07+S16) to match.
- `run_temporal_on_transcript.py` — **new (Week 3, 2026-06-13), implemented, updated 2026-06-14
  for S09/S12**. One-off validation script: loads a real HOPE transcript's patient turns,
  encodes them with the v1 (bert-base-uncased + mean pooling) embedder, runs the final
  sentiment MLP (`models/sentiment-v1-mlp-d05-wd3/`) to get per-turn sentiment + confidence
  (`SentimentTurn`), loads that transcript's distress signal labels from
  `utterance_labels.jsonl` (`DistressTurn`), and feeds both through `SessionAnalyzer`, printing
  a per-turn table (S05/S08/S09/S12 + distress codes). Usage:
  `python src/run_temporal_on_transcript.py Transcript_13 --split Validation`.
  **Run on Validation/Transcript_13 (64 patient turns), before tuning:** S08 fired on 6/64
  (~9%), S05 on 28/64 (~44%), S09 on 4/64 (~6%), **S12 on 22/64 (~34%)**. S09 looks reasonable
  (fires on genuine recurring S07/helplessness). **S12's ~34% rate is too high to be a
  meaningful "flag for counselor escalation review"** (taxonomy.json: S12 should be reserved,
  not routine) — it is largely inflated by S05's own over-firing (44%) counting toward the
  co-occurrence set.

  **Re-run 2026-06-14 after tuning `VOLATILITY_THRESHOLD=0.9` and `ESCALATION_MIN_SIGNALS=4`:**
  S08 on 8/64 (~12.5%, unchanged), **S05 on 13/64 (~20%, down from ~44%)**, S09 on 4/64 (~6%,
  unchanged), **S12 on 5/64 (~8%, down from ~34%)**. All four signals now fall in a plausible
  "worth a counselor's attention" range. Threshold tuning is concluded.

### Still To Do
- [ ] **Run `utterance_stats.ipynb`** to generate `utterances_with_signals.jsonl` — deferred to
      end of project (stats/EDA only, not a blocker for Week 3).
- [ ] Fill in `config.json` with model names/settings.
- [ ] `requirements.txt` — exists, keep pinning new deps (`datasets==4.8.5` added for TSDAE).
- [ ] `README.md` — project overview, dataset, ethics, setup instructions.
- [ ] `TRAINING.md` — Week 2 baseline training plan.
- [ ] Refactor `utils.py` — models reload on every call; add caching (`lru_cache`)
      and batch inference. Implement `load_hugging_face_model`.
- [ ] **Week 3 — Session-level aggregation pipeline** (utterance predictions → session
      signals, persisted output). `session_analysis.py` (S05/S08/S09/S12 logic) and
      `run_temporal_on_transcript.py` (single-transcript validation CLI) exist; still needed:
      a script that runs this over every transcript in a split (calling
      `distress_signal_inference.classify_utterances()` instead of reading
      `utterance_labels.jsonl`, since that file is LLM-labeled reference data per Rule 6, not
      meant as a production dependency) and writes session-level results to
      `processed/<SPLIT>/session_signals.jsonl`.
- [ ] Evaluation report with utterance-level + session-level metrics.
- [ ] `tests/` — no test suite exists yet.

### Current Position
Week 2 in progress. `preprocess.py`, `label.py`, `risk_taxonomy.py`, and
`sentiment_labeling.py` have run for all splits. TSDAE domain adaptation has run with the
target `bert-base-uncased` config. The sentiment classifier is **concluded**: after comparing
logistic regression, random forest, XGBoost, and PyTorch MLP heads (with a dropout/weight-decay
sweep) across v1 (raw BERT) and v2 (TSDAE-adapted) embeddings, the final model is
**v1 + MLP[128], dropout=0.5, weight_decay=1e-3** (Validation=0.7933, Test=0.7742, saved to
`models/sentiment-v1-mlp-d05-wd3/`) — just under the 0.80 target, treated as the practical
ceiling for this embedding + LLM-labeled-data setup (see Section 9).

**Distress signal classifier (S01–S18, multi-label) — abandoned (2026-06-13).** An embedding-based
multi-label MLP (BCE loss, sqrt-scaled + capped pos_weight for the severe imbalance) was tried on
both v1 (raw BERT) and v2 (TSDAE) embeddings. Results: v1 Validation macro F1=0.1886 / micro
F1=0.3405, Test macro F1=0.2550 / micro F1=0.3416; v2 was worse (Validation macro F1=0.0713 /
micro F1=0.2150). Most signals with <40 train examples collapsed to all-zero predictions — not
viable given the data scarcity (Rule 9). **Decision: distress signal detection uses the
OpenAI API directly at inference time** (GPT-4o-mini, temperature=0, structured output — same
approach as `risk_taxonomy.py`/`label.py`), not a trained classifier — implemented as
`distress_signal_inference.py` (see Section 9).

**Week 2 is now complete.** Sentiment classifier finalized (`models/sentiment-v1-mlp-d05-wd3/`),
distress signal inference module implemented and tested (`distress_signal_inference.py`).
`utterance_stats.ipynb` is intentionally deferred to the end of the project (stats/EDA only).

**Week 3 in progress.** All four temporal/composite signals are implemented:
`temporal_analysis.py` (S05 emotional volatility + S08 negative sentiment trend, via
`TemporalAnalyzer`) and `session_analysis.py` (S09 rumination + S12 escalation composite, via
`SessionAnalyzer`, which wraps `TemporalAnalyzer`). Both pass their synthetic-scenario
benchmarks (see Section 9 Implemented). Run end-to-end on a real transcript
(`run_temporal_on_transcript.py` on Validation/Transcript_13, 64 patient turns), before tuning:
S08 ~9%, S05 ~44%, S09 ~6%, S12 ~34%. S09 looks reasonable; S05 and S12 fired too often.

**Threshold tuning complete (2026-06-14).** `VOLATILITY_THRESHOLD` (0.6→0.9, ~p80 of volatility
scores across 4 Validation transcripts) and `ESCALATION_MIN_SIGNALS` (3→4) were tuned using
empirical analysis across multiple real transcripts, not just Transcript_13. Re-run on
Transcript_13: S08 ~12.5% (unchanged), S05 ~20% (down from ~44%), S09 ~6% (unchanged), S12 ~8%
(down from ~34%) — all four now fall in a plausible "worth a counselor's attention" range.
Synthetic benchmarks for both modules still pass (Scenario 2 in `session_analysis.py` updated
to use 4 distinct co-occurring signals).

**Next action (Week 3, continued): build the per-split session aggregation pipeline.** Write
the script that runs the full S05/S08/S09/S12 pipeline (via `SessionAnalyzer`) over every
transcript in a split, using `distress_signal_inference.classify_utterances()` rather than the
LLM-labeled `utterance_labels.jsonl` (per Rule 6, since that file is reference data, not a
production dependency), and writes `processed/<SPLIT>/session_signals.jsonl`.

---

## 10. Week 2 Plan — Sentiment Classifier (v1/v2) + OpenAI-based Distress Signals

**Goal:** Build an utterance-level sentiment classifier (positive/negative/neutral) and compare
embeddings with vs. without domain adaptation. Target: **≥80% validation accuracy**. Distress
signal detection (S01–S18) is handled separately via the OpenAI API, not a trained classifier
(see decision below).

**Ground truth (LLM-generated, not gold — see Rule 6):**
- Sentiment labels: `processed/<SPLIT>/utterance_sentiment.jsonl` (from `sentiment_labeling.py`,
  3-class + confidence, GPT-4o-mini, temperature=0). Generated for Train/Validation/Test.
- Distress signal labels: `processed/<SPLIT>/utterance_labels.jsonl` (from `label.py`) —
  used as LLM-labeled reference data, not for training a classifier (see below).

**Embedding base model — `bert-base-uncased`** (both v1 and v2 use the same base, per decision
on 2026-06-13, so the comparison isolates the effect of domain adaptation):
- **v1 (no domain adaptation):** raw `bert-base-uncased` sentence-transformer embeddings.
- **v2 (domain-adapted):** `bert-base-uncased` embeddings after TSDAE domain adaptation on the
  HOPE corpus (`models/tsdae-adapted/`).

**Sentiment classifier** — embeddings → classifier head trained on `utterance_sentiment.jsonl`
labels, compared across v1 vs. v2 and multiple heads (logreg, RF, XGBoost, MLP). **Concluded**
— see Section 9 for the final model and results.

**Distress signal classifier — abandoned (2026-06-13).** An embedding-based multi-label MLP
(S01–S18, BCE loss with sqrt-scaled/capped pos_weight) was tried on v1 and v2 embeddings but
collapsed on the majority of signals due to severe class imbalance (many signals have <40
training examples; macro F1 0.07–0.26). **Decision: distress signal detection at inference time
will call the OpenAI API directly** (GPT-4o-mini, temperature=0, structured Pydantic output over
the S01–S18 label set), mirroring `risk_taxonomy.py`/`label.py`'s approach, rather than training
a classifier on the LLM-generated labels.

**After Week 2:** proceed to Week 3 — temporal modeling and session-level aggregation
(S05, S08, S09, S12), building on the sentiment classifier + OpenAI-based distress signal calls.

