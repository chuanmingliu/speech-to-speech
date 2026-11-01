#!/usr/bin/env bash
# One-command local lab: realtime backend on :8765 + demo UI on :7860, then
# opens http://localhost:7860/lab. Ctrl+C stops both.
#
#   ./demo/start_lab.sh                      # default Mac-local models
#   ./demo/start_lab.sh --tts kokoro         # extra args go to speech-to-speech
#
# If something already listens on :8765 it is reused instead of started.
set -euo pipefail

cd "$(dirname "$0")/.."

BACKEND_PORT="${BACKEND_PORT:-8765}"
DEMO_PORT="${DEMO_PORT:-7860}"
LAB_URL="http://localhost:${DEMO_PORT}/lab"

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "==> Installing dependencies"
uv sync
uv pip install -q -r demo/requirements.txt

if port_open "$BACKEND_PORT"; then
  echo "==> Reusing backend already listening on :${BACKEND_PORT}"
else
  if [ "$#" -eq 0 ]; then
    set -- \
      --device mps \
      --stt parakeet-tdt \
      --llm_backend mlx-lm \
      --model_name mlx-community/Qwen3-4B-Instruct-2507-bf16 \
      --tts qwen3
  fi
  echo "==> Starting speech-to-speech backend on :${BACKEND_PORT} (first run downloads models)"
  uv run speech-to-speech --mode realtime --ws_port "$BACKEND_PORT" --enable_live_transcription "$@" &
  pids+=("$!")

  until port_open "$BACKEND_PORT"; do
    if ! kill -0 "${pids[0]}" 2>/dev/null; then
      echo "Backend exited before listening on :${BACKEND_PORT}" >&2
      exit 1
    fi
    sleep 2
  done
fi

echo "==> Starting demo UI on :${DEMO_PORT}"
SPEECH_TO_SPEECH_URL="${SPEECH_TO_SPEECH_URL:-ws://127.0.0.1:${BACKEND_PORT}/v1/realtime}" \
  uv run uvicorn --app-dir demo server:app --host 127.0.0.1 --port "$DEMO_PORT" &
pids+=("$!")

until port_open "$DEMO_PORT"; do sleep 1; done

echo
echo "    Lab ready: ${LAB_URL}"
echo "    Ctrl+C to stop."
echo
if command -v open >/dev/null 2>&1; then
  open "$LAB_URL"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$LAB_URL" >/dev/null 2>&1 || true
fi

wait
