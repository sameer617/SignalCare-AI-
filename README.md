# SignalCare AI

**SignalCare AI** is a real-time NLP system that detects emotional distress markers in
therapy-session transcripts to support — not replace — human counselors. It is being built
for **CHIEAC**, a nonprofit, as an 8-week portfolio-grade project.

## What this is

Counselors cannot perfectly track every linguistic distress cue across a long session.
SignalCare AI surfaces patterns — hopelessness escalation, catastrophizing, rumination,
emotional volatility, and more — so a counselor can review them alongside the session.

## What this is NOT

- It is **not** a diagnostic tool. It never outputs a clinical diagnosis.
- It is **not** a crisis-intervention system. It is decision **support**.
- It does **not** replace clinical judgment.

All signals are surfaced as flags + confidence scores for a human to review.

---

## Dataset

[`HOPE_WSDM_2022`](https://github.com/LCS2-IIITD/HOPE_WSDM_2022) — 317 real therapy
transcripts, split into Train / Validation / Test. Each transcript is a `.txt` file with
alternating `<Patient>:` / `<Therapist>:` turns. HOPE was originally built for dialogue-act
classification and ships without distress labels — SignalCare AI generates those labels with
an LLM (GPT-4o-mini, `temperature=0`).

> **Note:** Because labels are LLM-generated rather than clinician-annotated ground truth, any
> model trained on them inherits GPT-4o-mini's labeling accuracy as a ceiling. This is stated
> explicitly as a limitation and these labels are never presented as gold-standard.

The raw dataset under `HOPE_WSDM_2022/` is read-only and is not modified by any pipeline stage.

---

## Risk Signal Taxonomy (S01–S18)

The 18 distress signals are the core of the system. Canonical definitions live in
[`taxonomy.json`](taxonomy.json); the corresponding `RiskSignal` enum lives in
`src/risk_taxonomy.py`. The two are kept in sync.

| Code | Signal                          | Severity |
|------|----------------------------------|----------|
| S01  | Hopelessness escalation          | High     |
| S02  | Catastrophizing                  | Medium   |
| S03  | All-or-nothing thinking           | Low      |
| S04  | Self-blame amplification          | Medium   |
| S05  | Emotional volatility               | Medium   |
| S06  | Social withdrawal language        | Low      |
| S07  | Helplessness / loss of agency     | Medium   |
| S08  | Negative sentiment trend           | Medium   |
| S09  | Rumination patterns                | Low      |
| S10  | Emotional numbing                  | Low      |
| S11  | Cognitive distortion density       | Medium   |
| S12  | Escalation risk composite          | High     |
| S13  | Suicidal ideation                  | High     |
| S14  | Substance abuse                    | Low      |
| S15  | Self-harm                           | High     |
| S16  | Trauma or abuse                     | Medium   |
| S17  | Manic / hypomanic episodes          | High     |
| S18  | Psychotic symptoms                  | High     |

S05, S08, S09, and S12 are **temporal/session-level** signals — they require multi-turn
context and are computed by a rolling-window analyzer, not per-utterance. S12 is a **composite
meta-signal**: it never fires from a single signal alone, only when multiple
High/Medium-severity signals co-occur within a session window.

---

## Architecture / Pipeline

```
HOPE_WSDM_2022/*.txt (read-only)
   │
   ├──> src/risk_taxonomy.py ──────────> train/validation/test_risk_signals.csv
   │                                       (transcript-level LLM labels)
   │
   └──> src/preprocess.py ─────────────> processed/<SPLIT>/utterances.jsonl
              │                            (PII-redacted utterances)
              │
              ├──> src/label.py ────────> processed/<SPLIT>/utterance_labels.jsonl
              │                            (utterance-level distress signal labels)
              │
              └──> src/sentiment_labeling.py ─> processed/<SPLIT>/utterance_sentiment.jsonl
                                                  (utterance-level sentiment labels)

processed/<SPLIT>/utterances.jsonl + utterance_sentiment.jsonl
   │
   ├──> src/domain_adaptation.py ──> models/tsdae-adapted/ (TSDAE domain-adapted BERT)
   │
   └──> src/train_sentiment_*.py ──> models/sentiment-v1-mlp-d05-wd3/  (final sentiment model)

processed/<SPLIT>/utterances.jsonl
   │
   ├──> src/run_temporal_on_transcript.py  (single-transcript debug/inspection script)
   │       - embeds patient turns (v1 BERT) + sentiment MLP -> SentimentTurn
   │       - loads DistressTurn from utterance_labels.jsonl
   │       - feeds both through SessionAnalyzer -> S05/S08/S09/S12
   │
   └──> src/run_session_pipeline.py  (production per-split pipeline)
           - SentimentTurn via v1 embedder + sentiment MLP
           - DistressTurn via src/distress_signal_inference.py (OpenAI, S01-S18)
           - SessionAnalyzer -> S05/S08/S09/S12
           -> processed/<SPLIT>/session_signals.jsonl

app/  (FastAPI dashboard — v1 prototype)
   - Upload a video file OR a YouTube link
   - app/pipeline/run_video_pipeline.py orchestrates, per video segment:
       transcribe (Whisper) -> redact_pii -> sentiment MLP -> distress signals
       (OpenAI) -> SessionAnalyzer -> media/<user_id>/<session_id>/analysis.json
   - Dashboard replays the video with synced transcript + signal badges
```

### Where new files belong

- All Python source → `src/` (or `app/` for the dashboard application layer)
- All generated data → `processed/` (gitignored, never committed)
- Model checkpoints → `models/` (gitignored, never committed)
- Training scripts → `src/train_*.py`

---

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/sameer617/SignalCare-AI-.git
cd SignalCare-AI-

# Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `.env`

Create a `.env` file at the repo root (gitignored — never commit this) with:

```
OPENAI_API_KEY=sk-...
SECRET_KEY=<random string for session cookie signing>
```

### `ffmpeg`

The dashboard's video pipeline (audio extraction for Whisper transcription) requires
`ffmpeg`/`ffprobe` on `PATH`. On Windows:

```bash
winget install --id Gyan.FFmpeg -e
```

> If you start a server from a shell that was open before `ffmpeg` was added to `PATH`, the
> video pipeline will fail with a file-not-found error — open a fresh shell.

---

## Commands

Run everything from the repo root with the virtual environment activated.

```bash
# Stage 1: transcript-level signal labeling -> src/<split>_risk_signals.csv
python src/risk_taxonomy.py

# Stage 2: preprocessing -> processed/<SPLIT>/utterances.jsonl + redacted/
python src/preprocess.py

# Stage 3: utterance-level distress signal labeling -> utterance_labels.jsonl
python src/label.py

# Stage 4: utterance-level sentiment labeling -> utterance_sentiment.jsonl
python src/sentiment_labeling.py

# Domain adaptation (TSDAE) -> models/tsdae-adapted/
python src/domain_adaptation.py

# Sentiment classifier training (final: train_sentiment_mlp.py)
python src/train_sentiment_mlp.py

# Inspect temporal/session signals on a single real transcript
python src/run_temporal_on_transcript.py Transcript_13 --split Validation

# Run the full session aggregation pipeline for a split
python src/run_session_pipeline.py --split Validation [--limit N]

# Run the dashboard app
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000` and register an account.

---

## Dashboard (v1 prototype)

A FastAPI + SQLite + Jinja2 dashboard that lets a counselor:

1. **Register / log in** (cookie-based session auth).
2. **Upload a session recording** — either a video file, **or a YouTube link**
   (downloaded server-side via `yt-dlp`). Exactly one of the two is required.
3. **Start analysis** — runs in the background:
   - Extracts audio (`ffmpeg`) and transcribes with timestamps (OpenAI Whisper).
   - Redacts PII from each segment (Presidio).
   - Scores sentiment per segment (the v1 sentiment MLP).
   - Classifies distress signals S01–S18 per segment (GPT-4o-mini, structured output).
   - Feeds both through a rolling-window `SessionAnalyzer` for S05/S08/S09/S12.
   - Writes one record per segment to `analysis.json`.
4. **Replay** — the dashboard plays the video and progressively reveals the transcript,
   sentiment, and S05/S08/S09/S12 badges (with confidence + taxonomy definitions as
   tooltips) in sync with playback.

**Known limitation:** Whisper does not perform speaker diarization, so every transcript
segment is currently treated as a patient turn. True patient/therapist separation is a future
improvement.

`media/` (uploaded videos + analysis output) and `signalcare.db` are local, gitignored
artifacts.

---

## Modeling notes

- **Sentiment classifier (final):** `bert-base-uncased` embeddings (v1, no domain adaptation)
  + a PyTorch MLP head (`[128]`, dropout=0.5, weight_decay=1e-3), class-weighted
  cross-entropy with early stopping. Validation accuracy 0.7933, Test accuracy 0.7742.
  TSDAE domain adaptation (v2 embeddings) was tried but did not outperform v1 for this task.
- **Distress signal detection (S01–S18):** an embedding-based multi-label classifier was
  tried and abandoned due to severe class imbalance (many signals have <40 training
  examples). Distress signals are instead classified **at inference time via the OpenAI API**
  (GPT-4o-mini, `temperature=0`, structured Pydantic output) — see
  `src/distress_signal_inference.py`.
- **Temporal/session signals (S05, S08, S09, S12):** computed by a rolling-window
  `SessionAnalyzer` (`src/session_analysis.py`, wrapping `src/temporal_analysis.py`), with
  thresholds tuned against real Validation transcripts so each signal fires on roughly its
  intended "worth a counselor's attention" rate (S05 ~14%, S08 ~14%, S09 ~1%, S12 ~7% on
  Validation).

---

## Project status

- ✅ Data preprocessing, PII redaction, transcript-level + utterance-level LLM labeling
  (signals + sentiment) — run for all splits.
- ✅ Sentiment classifier finalized.
- ✅ Distress signal inference module (OpenAI-based).
- ✅ Temporal + session-level signal modules (S05/S08/S09/S12), tuned and validated.
- ✅ Per-split session aggregation pipeline — run on Validation; Train/Test pending.
- ✅ FastAPI dashboard v1 — auth, upload (file or YouTube link), background analysis
  pipeline, synced video/transcript/signal replay. Verified end-to-end.
- ⏳ Run session aggregation pipeline on Train/Test splits.
- ⏳ `utterance_stats.ipynb` EDA pass.
- ⏳ Evaluation report (utterance-level + session-level metrics).
- ⏳ Test suite.

---

## Ethics & limitations

- Labels used for training are LLM-generated (GPT-4o-mini), not clinician ground truth.
- Outputs are signals + confidence scores for a counselor to review — never a diagnosis,
  risk score, or crisis verdict.
- Patient text is PII-redacted (Presidio) before being persisted or sent to any model.
- Class imbalance is severe for several signals (some appear in only 1–6 transcripts);
  results should always be read alongside per-class metrics, not plain accuracy.
