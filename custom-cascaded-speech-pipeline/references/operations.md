# Tencent, DeepSeek, and MiniMax operations

## Files

- `.env.custom.example`: safe environment-variable template
- `.env.local`: ignored local credentials
- `configs/tencent-deepseek-minimax.json`: realtime provider selection
- `scripts/smoke_custom_services.py`: independent live provider checks
- `scripts/run_custom_services_test_app.py`: backend and browser app launcher
- `tests/test_custom_service_handlers.py`: adapter unit tests
- `tests/test_custom_service_launcher.py`: environment and port handling tests

## Required environment

Require these names without displaying their values:

```text
DEEPSEEK_API_KEY
TENCENT_ASR_SECRET_ID
TENCENT_ASR_SECRET_KEY
MINIMAX_TTS_API_KEY
MINIMAX_TTS_VOICE_ID
```

Useful optional settings:

```text
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
TENCENT_ASR_ENGINE=16k_zh
TENCENT_ASR_LANGUAGE=zh
TENCENT_ASR_APP_ID=
TENCENT_ASR_OPEN_TIMEOUT_S=1.5
MINIMAX_TTS_MODEL=speech-2.8-turbo
MINIMAX_TTS_LANGUAGE_BOOST=auto
MINIMAX_TTS_WARMUP=true
MINIMAX_TTS_MODEL_WARMUP=true
MINIMAX_TTS_MODEL_WARMUP_TEXT=Hi.
MINIMAX_TTS_WEBSOCKET_OPEN_TIMEOUT_S=5
MINIMAX_TTS_WEBSOCKET_RECEIVE_TIMEOUT_S=30
MINIMAX_TTS_WEBSOCKET_MAX_IDLE_S=90
MINIMAX_TTS_CONNECTION_KEEPALIVE_S=300
MINIMAX_TTS_CACHE_MAX_MB=32
PROVIDER_CONNECTION_MAINTENANCE_S=30
```

`TENCENT_ASR_APP_ID` is required for streaming Tencent ASR (realtime
WebSocket). Without it the handler falls back to one-shot
SentenceRecognition and live captions will not appear.
Realtime WebSocket setup fails fast after 1.5 seconds by default. If setup
fails for a turn, subsequent progressive snapshots prefetch
SentenceRecognition and finalization does not retry the same WebSocket.

MiniMax credentials are platform-specific. Use:

```text
MINIMAX_TTS_WEBSOCKET_ENDPOINT=wss://api.minimaxi.com/ws/v1/t2a_v2
MINIMAX_TTS_ENDPOINT=https://api.minimaxi.com/v1/t2a_v2
```

for China-platform keys, or:

```text
MINIMAX_TTS_WEBSOCKET_ENDPOINT=wss://api.minimax.io/ws/v1/t2a_v2
MINIMAX_TTS_ENDPOINT=https://api.minimax.io/v1/t2a_v2
```

for global-platform keys. Streaming uses the WebSocket endpoint by default;
the HTTP endpoint is used only when `MINIMAX_TTS_STREAM=false` (and for the
one-shot connection probe). If the WebSocket setting is omitted, it is derived
from `MINIMAX_TTS_ENDPOINT`. An application-level nonzero
`base_resp.status_code` is a failure even if the transport itself succeeded.

The low-latency profile disables DeepSeek thinking, keeps hosted HTTP
connections reusable for five minutes, and retains only eight conversational
turns without background compaction. Its one-token startup request uses the
same assembled voice-system prompt as a real turn plus an empty user message,
so provider prefix/KV caching can retain the expensive static prompt prefix.
Each Realtime pool lane installs that same `init_chat_prompt` as its session
default; a later client `session.update` can still override it.
MiniMax establishes one persistent
WebSocket `task_start` per pipeline lane, reuses it for sequential
`task_continue` sentences, and runs a hidden text synthesis to warm its model
(this may be billed). The hidden PCM is discarded from playback but retained
in the exact-text cache. Set `MINIMAX_TTS_MODEL_WARMUP_TEXT` to a common
agent-first greeting to make that exact greeting immediately reusable; arbitrary
later text only benefits from the warmed model, not from the exact-text cache.
A barge-in closes the active MiniMax task so unread audio cannot leak into the
next response. Idle LLM and TTS connections/model state are refreshed in
background threads on session connection or speech start, before final ASR
normally reaches the LLM. Unclaimed telephony pool lanes are also maintained
every 30 seconds: DeepSeek's HTTP transport is probed when stale, and each
MiniMax task is recycled after 90 idle seconds, ahead of MiniMax's documented
120-second application-idle disconnect. This keeps one isolated provider lane
ready per configured `num_pipelines`; Tencent ASR remains per utterance and is
never shared across callers. Set `PROVIDER_CONNECTION_MAINTENANCE_S=0` to
disable idle-pool maintenance. Set `MINIMAX_TTS_MODEL_WARMUP=false` to keep only the
non-billable WebSocket connection/task handshake,
`MINIMAX_TTS_WARMUP=false` to disable that preconnection too, or
`MINIMAX_TTS_CACHE_MAX_MB=0` to disable the cache. Redis is unnecessary for one
pipeline process; use a shared cache only when several processes need to reuse
the same synthesized sentences.

## Setup

Create the local file and fill it with fresh credentials:

```bash
cp .env.custom.example .env.local
```

Install the Tencent optional dependency:

```bash
uv sync --extra tencent-asr
```

The launcher safely loads simple `KEY=VALUE` entries from `.env.local`. The
chat-completions adapter reads `DEEPSEEK_API_KEY` and `responses_api_base_url`
directly, so `speech-to-speech configs/tencent-deepseek-minimax.json` does not
need `OPENAI_API_KEY`.

## Verification ladder

Run the independent live checks:

```bash
set -a
source .env.local
set +a
uv run --extra tencent-asr python scripts/smoke_custom_services.py
```

Expected outcome:

- Tencent returns one nonempty transcript from 16 kHz mono input.
- DeepSeek returns nonempty response text.
- MiniMax returns decodable 16 kHz mono PCM16 WAV audio.

Launch the complete browser app:

```bash
uv run --extra tencent-asr python scripts/run_custom_services_test_app.py
```

Open `http://127.0.0.1:7860`, click the center orb, grant microphone access, and
speak. Stop both child processes with `Ctrl-C`.

The launcher prefers backend port `8765`. If another process owns it, the
launcher creates a temporary profile with a free port and injects the matching
realtime URL into the demo. Do not treat a successful TCP connection to the
preferred port as proof that the intended backend started; also watch the child
process and its startup logs.

For a deterministic spoken-turn check on macOS, run one client and one turn with
`scripts/synthetic_conversation_realtime_client.py` against the realtime URL
printed by the launcher. Confirm:

- one accepted WebSocket session;
- one final ASR transcript;
- one LLM response;
- one successful persistent MiniMax WebSocket T2A task;
- one `response.done`;
- a nonempty 16 kHz mono WAV.

## Final checks

```bash
uv run ruff check src scripts tests
uv run --extra tencent-asr pytest -q
git diff --check
git status --short
```

Scan tracked and untracked source while excluding `.env.local`, `.venv`, `.git`,
and generated audio. Confirm `.env.local` is ignored with:

```bash
git check-ignore -v .env.local
```

Headless browsers often cannot provide a microphone or camera. Treat that as an
environment limitation only after the page, assets, `/api/config`, console, and
synthetic WebSocket turn have been checked.
