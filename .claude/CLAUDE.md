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
  output, prompt with explicit valid-label list. **Has been run** →
  `src/train_risk_signals.csv` exists with labels for the Train split.
- `taxonomy.json` — canonical definitions for the signals.
- `utils.py` — HF helpers: `sentence_level_sentiment`, `sentence_level_emotions`,
  file listing/reading. (`load_hugging_face_model` is a stub.)
- `preprocess.py` — written but **NOT yet run**. Splits transcripts into patient
  utterances, redacts PII, writes `processed/utterances.jsonl` + redacted copies.
- `label.py` — written, depends on `preprocess.py` output.
- `utterance_stats.ipynb` — EDA notebook, written but not yet run.
- `config.json` — minimal, holds emotion model name.

### Still To Do
- [ ] **Run `preprocess.py`** to generate `processed/utterances.jsonl`.
- [ ] **Run `utterance_stats.ipynb`** to generate `utterances_with_signals.jsonl`.
- [ ] `domain_adaptation.py` — currently empty. Implement TSDAE domain adaptation of a
      base sentence-transformer on the HOPE corpus.
- [ ] `requirements.txt` — does not exist; create and pin dependencies.
- [ ] `README.md` — project overview, dataset, ethics, setup instructions.
- [ ] `TRAINING.md` — Week 2 baseline training plan.
- [ ] Refactor `utils.py` — models reload on every call; add caching (`lru_cache`)
      and batch inference. Implement `load_hugging_face_model`.
- [ ] **Week 2: baseline sentiment + emotion model** — target ≥80% on validation.
- [ ] Utterance-level distress classifier trained on `label.py` output.
- [ ] Session-level aggregation logic (utterance predictions → session signals).
- [ ] Temporal signal logic (S05, S08, S09, S12) — needs the multi-turn structure.
- [ ] Evaluation report with utterance-level + session-level metrics.
- [ ] `tests/` — no test suite exists yet.
- [ ] Optionally run the labeling pipeline on Validation and Test splits
      (`risk_taxonomy.py` is currently hardcoded to Train).

### Current Position
End of Week 1 / start of Week 2. Immediate next action: run `preprocess.py` and
`utterance_stats.ipynb`, then implement `domain_adaptation.py`.
