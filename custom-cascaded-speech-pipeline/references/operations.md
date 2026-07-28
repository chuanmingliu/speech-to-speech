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
DEEPSEEK_MODEL=deepseek-chat
TENCENT_ASR_ENGINE=16k_zh
TENCENT_ASR_LANGUAGE=zh
MINIMAX_TTS_MODEL=speech-2.8-turbo
MINIMAX_TTS_LANGUAGE_BOOST=auto
```

MiniMax credentials are platform-specific. Use:

```text
MINIMAX_TTS_ENDPOINT=https://api.minimaxi.com/v1/t2a_v2
```

for China-platform keys, or:

```text
MINIMAX_TTS_ENDPOINT=https://api.minimax.io/v1/t2a_v2
```

for global-platform keys. An application-level nonzero `base_resp.status_code`
is a failure even if HTTP itself succeeded.

## Setup

Create the local file and fill it with fresh credentials:

```bash
cp .env.custom.example .env.local
```

Install the Tencent optional dependency:

```bash
uv sync --extra tencent-asr
```

The launcher safely loads simple `KEY=VALUE` entries from `.env.local` and maps
`DEEPSEEK_API_KEY` to the OpenAI-compatible key expected by the LLM adapter.

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
- one successful MiniMax HTTP response;
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
