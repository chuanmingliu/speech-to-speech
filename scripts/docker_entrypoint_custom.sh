#!/usr/bin/env bash
# Daemon entrypoint for the Tencent/DeepSeek/MiniMax realtime profile.
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" && -n "${DEEPSEEK_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${DEEPSEEK_API_KEY}"
fi

missing=()
for key in \
  DEEPSEEK_API_KEY \
  TENCENT_ASR_APP_ID \
  TENCENT_ASR_SECRET_ID \
  TENCENT_ASR_SECRET_KEY \
  MINIMAX_TTS_API_KEY \
  MINIMAX_TTS_VOICE_ID
do
  if [[ -z "${!key:-}" ]]; then
    missing+=("$key")
  fi
done

if ((${#missing[@]})); then
  echo "Missing required env vars: ${missing[*]}" >&2
  echo "Pass --env-file .env.local (or set them in docker-compose.custom.yml)." >&2
  exit 1
fi

exec "$@"
