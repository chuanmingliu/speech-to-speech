---
name: custom-cascaded-speech-pipeline
description: Configure, implement, test, and troubleshoot custom ASR, LLM, and TTS providers in this speech-to-speech repository. Use for hosted or self-hosted provider integration, realtime cascade profiles, credential templates, provider smoke tests, browser test-app launches, end-to-end spoken-turn validation, cancellation behavior, audio-format compatibility, or replacement of Tencent ASR, DeepSeek, and MiniMax with other services.
---

# Custom Cascaded Speech Pipeline

Build provider adapters around the repository's existing realtime contracts, then
prove each service independently before testing the complete spoken turn.

## Start safely

1. Work from the repository root containing `pyproject.toml` and
   `src/speech_to_speech`.
2. Inspect `git status --short` and preserve unrelated user changes.
3. Keep credentials in `.env.local`; confirm it is ignored before adding keys.
4. Never print, copy into source, commit, or echo secret values. Report only
   whether each required variable is set.
5. Tell the user to rotate any credential pasted into chat, logs, or committed
   history.

## Choose the task path

- For the checked-in Tencent + DeepSeek + MiniMax stack, read
  [references/operations.md](references/operations.md) and follow its exact
  setup and verification ladder.
- For a new provider, read
  [references/architecture-contracts.md](references/architecture-contracts.md)
  before editing handlers, registration, or profiles.
- For latency, interruption, retries, fallbacks, or future streaming work, read
  the design and improvement sections in
  [references/architecture-contracts.md](references/architecture-contracts.md).

## Implement a provider

1. Locate the closest existing handler and reuse its queue/message contract.
2. Keep network-specific request and response conversion inside the adapter.
3. Register the provider in
   `src/speech_to_speech/arguments_classes/module_arguments.py` and
   `src/speech_to_speech/s2s_pipeline.py`.
4. Add an optional dependency only when the provider needs a non-core SDK.
5. Put non-secret defaults in a checked-in JSON profile and credential names in
   a safe example environment file.
6. Preserve `turn_id` and `turn_revision` across stage boundaries.
7. Wire TTS to `CancelScope` and `SpeculativeTurnTracker`; discard stale output
   after interruption or a newer turn revision.
8. Add unit tests with injected fake clients. Do not make live provider calls
   from the ordinary test suite.

## Verify in increasing scope

Run checks in this order so a failure identifies its layer:

1. Environment presence checks without values.
2. Handler unit tests using fake clients.
3. Independent live ASR, LLM, and TTS smoke tests.
4. Realtime backend plus browser test app.
5. One synthetic WebSocket spoken turn through ASR → LLM → TTS.
6. Browser asset, configuration, and console inspection.
7. Ruff, the complete pytest suite, `git diff --check`, and a secret-pattern
   scan that excludes ignored environment and virtual-environment files.

Do not claim end-to-end success from three independent provider calls. Require a
single turn that produces both a transcript and playable response audio.

## Diagnose by boundary

- No transcript: validate final VAD audio, sample rate, duration, encoding, ASR
  engine, and ASR region/credentials.
- No LLM text: validate API base, compatibility mode, model name, credential
  alias, and streaming response parsing.
- No speech: validate MiniMax platform endpoint, voice ID ownership, response
  status in the JSON body, WAV encoding, sample rate, channels, and sample width.
- UI loads but cannot talk: inspect `/api/config`, the realtime URL, occupied
  ports, WebSocket acceptance, and browser microphone permissions.
- Old speech continues after interruption: inspect cancel generation and
  speculative turn/revision checks before yielding each audio chunk.

Lead the handoff with the exact launch command, URL, verified stages, test
results, known limitations, and clickable links to changed files.
