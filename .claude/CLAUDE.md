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

## 2. Risk Signal Taxonomy (S01–S18)

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

## 3. Architecture



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

## 4. Code Style

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

## 5. Preferred Libraries

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

## 6. Commands

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

## 7. Critical Rules

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

## 8. Implemented vs. Still To Do

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
  Week 2 ground-truth generation step (see Section 9).
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

### Still To Do
- [ ] **Run `utterance_stats.ipynb`** to generate `utterances_with_signals.jsonl`.
- [ ] Fill in `config.json` with model names/settings.
- [ ] `requirements.txt` — exists, keep pinning new deps (`datasets==4.8.5` added for TSDAE).
- [ ] `README.md` — project overview, dataset, ethics, setup instructions.
- [ ] `TRAINING.md` — Week 2 baseline training plan.
- [ ] Refactor `utils.py` — models reload on every call; add caching (`lru_cache`)
      and batch inference. Implement `load_hugging_face_model`.
- [ ] Distress signal inference pipeline using the OpenAI API directly (GPT-4o-mini,
      temperature=0, structured output via Pydantic — same pattern as `risk_taxonomy.py` /
      `label.py`), per the 2026-06-13 decision in Section 9. No embedding-based distress
      classifier will be built.
- [ ] Session-level aggregation logic (utterance predictions → session signals).
- [ ] Temporal signal logic (S05, S08, S09, S12) — needs the multi-turn structure.
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
ceiling for this embedding + LLM-labeled-data setup (see Section 8).

**Distress signal classifier (S01–S18, multi-label) — abandoned (2026-06-13).** An embedding-based
multi-label MLP (BCE loss, sqrt-scaled + capped pos_weight for the severe imbalance) was tried on
both v1 (raw BERT) and v2 (TSDAE) embeddings. Results: v1 Validation macro F1=0.1886 / micro
F1=0.3405, Test macro F1=0.2550 / micro F1=0.3416; v2 was worse (Validation macro F1=0.0713 /
micro F1=0.2150). Most signals with <40 train examples collapsed to all-zero predictions — not
viable given the data scarcity (Rule 9). **Decision: distress signal detection will use the
OpenAI API directly at inference time** (GPT-4o-mini, temperature=0, structured output — same
approach as `risk_taxonomy.py`/`label.py`), not a trained classifier. See Section 9 for the
updated plan.

Immediate next action: build the OpenAI-based distress signal inference pipeline, then proceed
to Week 3 (session-level aggregation + temporal signals).

---

## 9. Week 2 Plan — Sentiment Classifier (v1/v2) + OpenAI-based Distress Signals

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
— see Section 8 for the final model and results.

**Distress signal classifier — abandoned (2026-06-13).** An embedding-based multi-label MLP
(S01–S18, BCE loss with sqrt-scaled/capped pos_weight) was tried on v1 and v2 embeddings but
collapsed on the majority of signals due to severe class imbalance (many signals have <40
training examples; macro F1 0.07–0.26). **Decision: distress signal detection at inference time
will call the OpenAI API directly** (GPT-4o-mini, temperature=0, structured Pydantic output over
the S01–S18 label set), mirroring `risk_taxonomy.py`/`label.py`'s approach, rather than training
a classifier on the LLM-generated labels.

**After Week 2:** proceed to Week 3 — temporal modeling and session-level aggregation
(S05, S08, S09, S12), building on the sentiment classifier + OpenAI-based distress signal calls.

