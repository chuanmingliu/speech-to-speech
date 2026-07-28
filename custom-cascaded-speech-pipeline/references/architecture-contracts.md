# Architecture and provider contracts

## Cascade

```text
Browser microphone
  → realtime WebSocket input
  → 16 kHz mono PCM frames
  → VAD
  → finalized utterance
  → ASR adapter
  → Transcription(turn_id, turn_revision)
  → OpenAI-compatible LLM adapter
  → sentence-sized TTSInput messages
  → TTS adapter
  → PCM16 audio chunks
  → realtime WebSocket output
  → browser playback
```

Control state travels alongside media:

- `turn_id` identifies a conversational turn.
- `turn_revision` identifies newer speculative/final versions of that turn.
- `CancelScope.generation` invalidates in-flight output after interruption.
- `SpeculativeTurnTracker` prevents stale revisions from reaching playback.
- `EndOfResponse` becomes `AUDIO_RESPONSE_DONE`.

## ASR contract

Accept VAD audio as normalized floating-point samples and emit a
`Transcription`. Convert to the provider's required encoding at the adapter
boundary.

For Tencent `SentenceRecognition`:

- consume only `mode="final"` messages;
- ignore progressive snapshots to avoid duplicate billable requests;
- convert finite float samples to clipped little-endian PCM16;
- use 16 kHz mono PCM and base64-encode it;
- reject utterances over the provider's 60-second limit;
- preserve turn ID, revision, and speech stop timestamp.

If replacing Tencent with a streaming ASR, progressive audio may be useful, but
deduplicate partials and still emit an authoritative final transcript.

## LLM contract

Prefer the existing OpenAI-compatible `chat-completions` adapter when the
provider implements `/v1/chat/completions`. Keep provider selection in profile
data:

```json
{
  "llm_backend": "chat-completions",
  "model_name": "deepseek-chat",
  "responses_api_base_url": "https://api.deepseek.com",
  "responses_api_stream": true
}
```

Map the provider key to the environment name expected by the compatibility
client at launch time. Do not duplicate the LLM handler solely to rename an
environment variable.

Stream text when supported so the output processor can send sentence-sized
segments to TTS before the whole answer is complete. Keep tool calls and usage
events separate from spoken text.

## TTS contract

Accept `TTSInput` text or `EndOfResponse`. Emit fixed-size NumPy `int16` PCM
chunks and finally the audio-done sentinel.

For MiniMax synchronous T2A:

- send Bearer authorization;
- request hex-encoded WAV, 16 kHz, one channel;
- check both HTTP status and `base_resp.status_code`;
- validate WAV channel count, PCM16 width, and sample rate;
- chunk decoded samples and pad the final block;
- check cancellation before yielding every chunk;
- drop stale turn revisions before making the request;
- close owned HTTP clients during cleanup.

The current synchronous request minimizes adapter complexity but adds
time-to-first-audio. A future streaming MiniMax adapter should incrementally
decode provider frames while retaining the same PCM chunk and cancellation
contract.

## Configuration boundary

Keep these separate:

- Source code: message conversion, validation, cancellation, errors.
- JSON profile: provider choices and safe runtime defaults.
- Environment: credentials, account-specific endpoints, voice IDs.
- Launcher: environment loading, credential aliases, dynamic local ports.
- Smoke script: real provider reachability and response-format validation.
- Unit tests: deterministic behavior with fake provider clients.

## Failure and retry policy

Do not retry invalid credentials, unsupported voices, malformed payloads, or
application-level provider errors. Consider bounded retries with jitter only for
timeouts, connection resets, 429 responses, and transient 5xx responses.

Do not replay ASR or TTS blindly after the user has interrupted or a newer turn
revision exists. Any retry loop must recheck cancellation and stale-turn state.

Surface errors with provider, stage, HTTP status, and safe provider error code.
Never include authorization headers, full request bodies containing sensitive
text, or credential values.

## Observability

Measure timestamps at:

- speech start and VAD finalization;
- ASR request start and final transcript;
- LLM request start, first token, and completion;
- TTS request start, first audio, and completion;
- first browser playback.

Report stage latency and end-to-end latency separately. A healthy independent
provider smoke test does not reveal queueing, sentence segmentation, port
miswiring, cancellation, or playback failures.

## Improvements worth considering

1. Add a provider registry so new adapters register without expanding central
   conditionals.
2. Add true streaming ASR and TTS adapters for lower time-to-first-audio.
3. Add per-stage timeout, bounded transient retry, and circuit-breaker policy.
4. Add optional fallback providers with explicit format-normalization tests.
5. Add structured safe telemetry keyed by session and turn IDs.
6. Add a provider-neutral conformance suite for audio formats, message metadata,
   cancellation, cleanup, and error mapping.
7. Let the demo receive optional provider labels so a custom stack does not
   retain unrelated upstream branding.
