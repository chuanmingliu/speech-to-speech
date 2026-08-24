# Architecture and provider contracts

## Cascade

```text
Browser microphone
  → realtime WebSocket input
  → 16 kHz mono PCM frames
  → VAD
  → progressive + finalized utterance
  → streaming ASR adapter
  → PartialTranscription then Transcription(turn_id, turn_revision)
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

For Tencent ASR:

- convert finite float samples to clipped little-endian PCM16;
- preserve turn ID, revision, and speech stop timestamp;
- the Tencent profile enables live transcription so VAD emits progressive
  snapshots about every 0.2 s while the user is speaking;
- when `TENCENT_ASR_APP_ID` is set, stream those snapshots over the
  realtime WebSocket API (`needvad=0`, `voice_format=1`) and emit
  `PartialTranscription` updates, then one final `Transcription`;
- otherwise use `SentenceRecognition` on finalized audio, with one
  background prefetch on trailing silence that is reused when the extra
  tail is short;
- reject SentenceRecognition utterances over the provider's 60-second limit.

## LLM contract

Prefer the existing OpenAI-compatible `chat-completions` adapter when the
provider implements `/v1/chat/completions`. Keep provider selection in profile
data:

```json
{
  "llm_backend": "chat-completions",
  "model_name": "deepseek-v4-flash",
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

For MiniMax T2A:

- send Bearer authorization;
- stream by default (`stream=true`) and request hex-encoded 16 kHz mono PCM;
- parse SSE `data:` frames as they arrive and yield PCM16 chunks immediately;
- skip aggregated `status=2` audio when incremental `status=1` frames already played;
- fall back to a single hex WAV response when `MINIMAX_TTS_STREAM=false`;
- check both HTTP status and `base_resp.status_code`;
- check cancellation before yielding every chunk and while reading the stream;
- drop stale turn revisions before making the request;
- close owned HTTP clients during cleanup.

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
2. MiniMax TTS streams hex PCM. Tencent realtime WebSocket ASR is used when
   an AppId is configured; otherwise SentenceRecognition prefetches on trailing
   silence.
3. Add per-stage timeout, bounded transient retry, and circuit-breaker policy.
4. Add optional fallback providers with explicit format-normalization tests.
5. Add structured safe telemetry keyed by session and turn IDs.
6. Add a provider-neutral conformance suite for audio formats, message metadata,
   cancellation, cleanup, and error mapping.
7. Let the demo receive optional provider labels so a custom stack does not
   retain unrelated upstream branding.
