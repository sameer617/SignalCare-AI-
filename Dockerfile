# SignalCare AI — dashboard app image
#
# Build (from repo root):
#   docker build -t signalcare-ai .
#
# Run locally:
#   docker run --rm -p 8000:8000 --env-file .env -v ./media:/app/media -v ./signalcare.db:/app/signalcare.db signalcare-ai

FROM python:3.12-slim

# ffmpeg: required by app/pipeline/transcribe.py (audio extraction) and
# merging separate YouTube video/audio streams (youtube_download.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install the CPU-only torch build first (pinned to the same versions as
# requirements.txt) so pip doesn't pull torch's default CUDA wheels — those
# add several GB of unused NVIDIA packages on a CPU-only EC2 instance.
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_lg

# App code + the two models actually used by the live pipeline
# (run_temporal_on_transcript.build_embedding_model/build_tsdae_embedding_model/load_sentiment_mlp):
#   - models/tsdae-adapted/        (v2 TSDAE embedder)
#   - models/sentiment-v4-ensemble/ (final sentiment MLP head)
# bert-base-uncased (v1 embedder) is downloaded from the HF Hub on first run
# and cached in the image layer below.
COPY src/ ./src/
COPY app/ ./app/
COPY taxonomy.json .
COPY models/tsdae-adapted/ ./models/tsdae-adapted/
COPY models/sentiment-v4-ensemble/ ./models/sentiment-v4-ensemble/

# Pre-download bert-base-uncased so it's baked into the image, not fetched
# on first request.
RUN python -c "from sentence_transformers.sentence_transformer.modules import Transformer; Transformer('bert-base-uncased')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
