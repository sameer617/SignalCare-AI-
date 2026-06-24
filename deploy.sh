#!/bin/bash
set -euo pipefail

REGION="us-east-1"
SECRET_ID="signalcare-ai/app-secrets"
IMAGE="648445704296.dkr.ecr.us-east-1.amazonaws.com/signalcare-ai:latest"
APP_DIR="/home/ubuntu/signalcare-ai"

mkdir -p "$APP_DIR/media"
touch "$APP_DIR/signalcare.db"

SECRET_JSON=$(aws secretsmanager get-secret-value --secret-id "$SECRET_ID" --region "$REGION" --query SecretString --output text)
OPENAI_API_KEY=$(echo "$SECRET_JSON" | jq -r '.OPENAI_API_KEY')
SECRET_KEY=$(echo "$SECRET_JSON" | jq -r '.SECRET_KEY')

echo "OPENAI_API_KEY=$OPENAI_API_KEY" > "$APP_DIR/.env"
echo "SECRET_KEY=$SECRET_KEY" >> "$APP_DIR/.env"
chmod 600 "$APP_DIR/.env"

docker pull "$IMAGE"
docker rm -f signalcare-ai 2>/dev/null || true

COOKIES_MOUNT=""
if [ -f "/home/ubuntu/cookies.txt" ]; then
  COOKIES_MOUNT="-v /home/ubuntu/cookies.txt:/app/cookies.txt -e YT_DLP_COOKIES_FILE=/app/cookies.txt"
fi

docker run -d --name signalcare-ai --restart unless-stopped -p 80:8000 --env-file "$APP_DIR/.env" -v "$APP_DIR/media:/app/media" -v "$APP_DIR/signalcare.db:/app/signalcare.db" $COOKIES_MOUNT "$IMAGE"

echo "Deployed. Container status:"
docker ps --filter "name=signalcare-ai"
