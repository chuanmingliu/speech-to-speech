# Provider-Native Streaming Speech Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Tencent ASR, DeepSeek inference, and MiniMax TTS stream provider output end-to-end so microphone speech produces partial text and audible assistant audio with minimal batch delay.

**Architecture:** Tencent gets a per-turn signed WebSocket session with a paced writer and concurrent result reader; DeepSeek keeps SSE streaming but flushes one complete sentence at a time; MiniMax keeps one WebSocket T2A task per assistant response and incrementally decodes ordered MP3 fragments into the existing 16 kHz mono PCM16 output contract. Existing turn/revision and cancel-generation fences own all provider teardown and stale-output suppression.

**Tech Stack:** Python 3.10+, `websockets.sync`, Tencent realtime ASR WebSocket v2, DeepSeek OpenAI-compatible SSE, MiniMax WebSocket T2A v2, PyAV incremental MP3 decoding/resampling, NumPy, pytest, Ruff.

## Global Constraints

- Public `/v1/realtime` events and the browser's PCM contract do not change.
- Tencent, DeepSeek, and MiniMax credentials remain server-side environment values and never enter URLs, logs, metrics, transcripts, or committed configuration.
- Tencent audio is 16 kHz mono PCM16 and is sent suffix-only in frames no larger than 200 ms / 6,400 bytes.
- Only final Tencent transcription invokes the LLM; partial transcription is presentation-only.
- DeepSeek remains `deepseek-v4-flash`, `stream=true`, and `thinking.type=disabled`.
- MiniMax uses `wss://api.minimax.io/ws/v1/t2a_v2`, ordered `task_start` / `task_continue` / `task_finish`, and MP3 streaming output.
- Downstream audio remains 16 kHz mono signed PCM16 in fixed 512-sample blocks; only the final block may be zero-padded.
- Every provider connection/read/write/terminal wait has a deadline and bounded message size.
- Barge-in or stale turn/revision closes provider sockets and no late partial, final, token, or audio crosses the active generation fence.
- No batch fallback is allowed for Tencent or MiniMax; unavailable streaming support fails closed.
- Live credentialed acceptance is reported separately and is never inferred from fake-provider tests.

---

## File Structure

- `src/speech_to_speech/STT/tencent_realtime_client.py`: signing, connection, paced suffix-only audio writer, concurrent response parsing, and bounded session lifecycle.
- `src/speech_to_speech/STT/tencent_asr_handler.py`: adapts progressive/final `VADAudio` messages to one Tencent realtime session and emits partial/final transcription messages.
- `src/speech_to_speech/TTS/incremental_mp3_decoder.py`: bounded PyAV MP3 parser/decoder/resampler producing PCM16 blocks.
- `src/speech_to_speech/TTS/minimax_tts_handler.py`: MiniMax WebSocket protocol state machine, response-scoped connection reuse, cancellation, and streamed audio output.
- `src/speech_to_speech/LLM/base_openai_compatible_language_model.py`: existing sentence-boundary streaming remains unchanged except content-free first-delta timing.
- `configs/tencent-deepseek-minimax.json`: enables progressive transcription and one-sentence speech batching.
- `tests/test_tencent_realtime_asr.py`: Tencent signing, suffix framing, results, deadlines, and cancellation.
- `tests/test_minimax_streaming_tts.py`: MiniMax protocol order, incremental decoding, ordering, bounds, finalization, and cancellation.
- `tests/test_provider_streaming_pipeline.py`: profile and cross-stage first-output-before-terminal/cancellation regression.
- `tests/test_custom_service_handlers.py`: remove batch-provider assertions and retain provider selection/error contracts.
- `pyproject.toml`, `uv.lock`: make PyAV an explicit runtime dependency for streaming MiniMax decode.
- `README.md`: identify all three provider stages as streaming and document required AppID and latency verification.

### Task 1: Tencent Realtime ASR WebSocket

**Files:**
- Create: `src/speech_to_speech/STT/tencent_realtime_client.py`
- Modify: `src/speech_to_speech/STT/tencent_asr_handler.py`
- Modify: `configs/tencent-deepseek-minimax.json`
- Create: `tests/test_tencent_realtime_asr.py`
- Modify: `tests/test_custom_service_handlers.py`

**Interfaces:**
- Produces: `build_tencent_realtime_url(config: TencentRealtimeConfig, *, voice_id: str, now_s: int, nonce: int) -> str`.
- Produces: `TencentRealtimeSession(config, *, connect_fn=websockets.sync.client.connect, clock=monotonic, sleep=time.sleep)` with `push_snapshot(audio: np.ndarray)`, `finish(audio: np.ndarray)`, `drain_results() -> list[TencentRecognitionResult]`, and `close()`.
- Produces: `TencentRecognitionResult(text: str, *, final: bool, stable: bool)`; handler maps unstable text to `PartialTranscription` and final text to `Transcription`.
- Consumes: cumulative progressive/final `VADAudio.audio`; session owns `samples_sent` and sends only `audio[samples_sent:]`.

- [ ] **Step 1: Write signing and privacy tests**

Add tests with fixed AppID, SecretID, SecretKey, timestamp, nonce, and voice ID. Independently calculate the expected HMAC-SHA1 over the lexically sorted query without `signature`, assert the returned URL contains the encoded signature and required `voice_format=1`, and assert `repr(config)`, raised errors, and captured logs never contain SecretKey or signature.

```python
def test_tencent_url_signs_the_exact_canonical_query_without_exposing_secret(caplog):
    cfg = TencentRealtimeConfig(app_id="1250000000", secret_id="sid", secret_key="sentinel-secret")
    url = build_tencent_realtime_url(cfg, voice_id="voice-1", now_s=1000, nonce=7)
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "wss"
    assert query["voice_format"] == ["1"]
    assert query["signature"] == [expected_signature]
    assert "sentinel-secret" not in url + repr(cfg) + caplog.text
```

- [ ] **Step 2: Run the signing test and verify RED**

Run: `.venv/bin/pytest -q tests/test_tencent_realtime_asr.py -k signing`

Expected: FAIL because `tencent_realtime_client` and `TencentRealtimeConfig` do not exist.

- [ ] **Step 3: Implement canonical signing and bounded connection configuration**

Create frozen configuration/result dataclasses. Validate AppID, SecretID, SecretKey, engine, endpoint, connect/read/write/final timeouts, 6,400-byte frame maximum, and 1 MiB JSON maximum. Construct the canonical query with `timestamp`, `expired=timestamp+3600`, `nonce`, `engine_model_type`, `voice_id`, `voice_format=1`, `needvad=1`, and `filter_empty_result=1`; HMAC-SHA1 the authority/path/query string and append the base64 signature only to the dial URL. Use `repr=False` for all secret fields.

- [ ] **Step 4: Write suffix-only streaming and concurrent-result tests**

Use a fake synchronous WebSocket and deterministic clock. Feed cumulative snapshots of 3,200, 6,400, and 8,000 samples. Assert the provider receives each PCM sample exactly once, every binary message is at most 6,400 bytes, the writer paces consecutive frames, and `finish()` sends `{"type":"end"}` after the final unseen suffix. Queue provider events with `slice_type` 1 and 2 followed by `final:1`; assert the partial result is observable before `finish()` and only the stable/final transcript is terminal.

```python
assert b"".join(fake.binary_messages) == expected_pcm_for_8000_samples
assert all(len(frame) <= 6400 for frame in fake.binary_messages)
assert fake.text_messages[-1] == '{"type":"end"}'
assert session.drain_results()[0] == TencentRecognitionResult("你好", final=False, stable=False)
```

- [ ] **Step 5: Run the session tests and verify RED**

Run: `.venv/bin/pytest -q tests/test_tencent_realtime_asr.py -k 'suffix or partial or final or pace'`

Expected: FAIL because the realtime session lifecycle is missing.

- [ ] **Step 6: Implement the realtime session**

Create one writer thread and one reader thread per session. The writer consumes a bounded queue of immutable PCM frames, paces by audio duration from a monotonic origin, and sends the terminal JSON only after all queued audio. The reader validates text-only JSON, maximum size, `code == 0`, matching voice ID, monotonically accepted provider messages, `slice_type`, and terminal `final == 1`; it stores only the latest bounded partial per sentence index plus stable sentence text. `finish()` waits for the terminal event using `final_timeout_s`; `close()` sets cancellation, closes the socket, joins threads within the close deadline, clears buffers, and is idempotent.

- [ ] **Step 7: Write handler turn/revision, partial/final, and closure tests**

Inject a fake session factory into `TencentASRHandler`. Assert a progressive message opens one session and yields `PartialTranscription`; another cumulative progressive message reuses it; final reuses it, calls `finish`, and yields one `Transcription`; a new turn/revision closes the prior session; provider failure and `on_session_end()` close it. Assert required `TENCENT_ASR_APP_ID` is read without logging its value.

- [ ] **Step 8: Run handler tests and verify RED**

Run: `.venv/bin/pytest -q tests/test_tencent_realtime_asr.py -k handler`

Expected: FAIL because `TencentASRHandler` still drops progressive messages and calls `SentenceRecognition`.

- [ ] **Step 9: Replace the batch handler and enable progressive snapshots**

Remove Tencent Cloud SentenceRecognition request construction and SDK use. Configure `TencentRealtimeConfig` from `TENCENT_ASR_APP_ID`, `TENCENT_ASR_SECRET_ID`, `TENCENT_ASR_SECRET_KEY`, engine, and language. Track the active `(turn_id, turn_revision)` session; map session results into existing typed messages; preserve `speech_stopped_at_s` from the final VAD message. Set `enable_live_transcription: true` and `live_transcription_update_interval: 0.2` in the checked-in provider profile.

- [ ] **Step 10: Run Task 1 tests and lint**

Run: `.venv/bin/pytest -q tests/test_tencent_realtime_asr.py tests/test_custom_service_handlers.py tests/test_stt_stale_filter.py`

Run: `.venv/bin/ruff check src/speech_to_speech/STT/tencent_realtime_client.py src/speech_to_speech/STT/tencent_asr_handler.py tests/test_tencent_realtime_asr.py`

Expected: PASS.

- [ ] **Step 11: Commit Task 1**

```bash
git add src/speech_to_speech/STT/tencent_realtime_client.py src/speech_to_speech/STT/tencent_asr_handler.py configs/tencent-deepseek-minimax.json tests/test_tencent_realtime_asr.py tests/test_custom_service_handlers.py
git commit -m "feat: stream Tencent realtime ASR"
```

### Task 2: MiniMax WebSocket TTS and Incremental MP3 Decode

**Files:**
- Create: `src/speech_to_speech/TTS/incremental_mp3_decoder.py`
- Rewrite: `src/speech_to_speech/TTS/minimax_tts_handler.py`
- Create: `tests/test_minimax_streaming_tts.py`
- Modify: `tests/test_custom_service_handlers.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `IncrementalMP3Decoder(sample_rate=16000, channels=1, block_samples=512)` with `feed(encoded: bytes) -> list[np.ndarray]`, `finish() -> list[np.ndarray]`, and `close()`.
- Produces: `MiniMaxStreamingClient(config, *, connect_fn=websockets.sync.client.connect, decoder_factory=IncrementalMP3Decoder)` with `start()`, `synthesize(text, *, cancelled: Callable[[], bool]) -> Iterator[np.ndarray]`, `finish()`, and `close()`.
- `MiniMaxTTSHandler` owns one streaming client for all `TTSInput` messages in a response and closes it on `EndOfResponse`, cancellation, stale turn, provider error, session end, or cleanup.

- [ ] **Step 1: Add PyAV runtime dependency and write decoder tests**

Make `av>=14.0.0` a direct runtime dependency. In the test, encode a deterministic 440 Hz mono signal into multiple MP3 packets with PyAV, split the bytes at non-frame boundaries, feed them incrementally, and assert PCM arrives before `finish()`, ordering is preserved, output is 16 kHz mono `int16`, blocks contain 512 samples, and only the final block is padded.

```python
first = decoder.feed(encoded[:split])
second = decoder.feed(encoded[split:])
tail = decoder.finish()
assert first or second
assert all(x.dtype == np.int16 and x.shape == (512,) for x in first + second + tail)
```

- [ ] **Step 2: Run decoder test and verify RED**

Run: `.venv/bin/pytest -q tests/test_minimax_streaming_tts.py -k decoder`

Expected: FAIL because `IncrementalMP3Decoder` does not exist.

- [ ] **Step 3: Implement bounded incremental decode**

Use `av.CodecContext.create("mp3", "r")` to parse arbitrary byte fragments and decode complete frames. Resample via `av.AudioResampler(format="s16", layout="mono", rate=16000)`. Copy decoded samples into a private NumPy reservoir capped at two output blocks plus decoder-owned state; emit every complete 512-sample block immediately. `finish()` parses/decodes the terminal buffered bytes, flushes codec/resampler, emits remaining complete blocks, pads at most one final block, and releases codec/frame references. Reject input fragments over 1 MiB and decoded growth beyond the declared cap.

- [ ] **Step 4: Write MiniMax handshake/order/streaming tests**

Inject a fake WebSocket and decoder. Assert the client authenticates with an `Authorization` header, receives `connected_success`, sends `task_start` with model/voice and MP3 mono settings, waits for `task_started`, sends `task_continue` text, yields PCM from the first audio event before the fake emits `is_final`, handles multiple audio events in exact order, then sends `task_finish` and requires `task_finished`.

```python
assert sent_events == ["task_start", "task_continue"]
assert next(audio_iter).tolist() == first_decoded_block.tolist()
assert provider_terminal_has_not_been_released.is_set() is False
```

- [ ] **Step 5: Run protocol tests and verify RED**

Run: `.venv/bin/pytest -q tests/test_minimax_streaming_tts.py -k 'handshake or order or before_terminal'`

Expected: FAIL because the handler still sends batch HTTP with `stream=false`.

- [ ] **Step 6: Implement the WebSocket protocol client**

Validate an exact `wss://api.minimax.io/ws/v1/t2a_v2` endpoint by default, connect with `proxy=None`, TLS verification enabled, compression disabled, `max_size=1 MiB`, bounded queue, and explicit open/close timeouts. Validate every JSON event and `base_resp.status_code`. Decode hex audio fragments once, feed the decoder, and yield each PCM block immediately. Poll reads with a short bounded timeout so `cancelled()` can close the socket promptly. Never log text, audio, auth headers, trace IDs, or raw provider messages.

- [ ] **Step 7: Write cancellation, bounds, and handler-reuse tests**

Assert a cancellation between two provider fragments closes the socket and drops the second fragment; malformed hex, oversized event/audio, wrong event order, `task_failed`, read timeout, and missing terminal fail closed. Feed two `TTSInput` sentences plus `EndOfResponse`; assert exactly one task start, two ordered task continues, one finish, and one `AUDIO_RESPONSE_DONE`. A new response after completion creates a new task.

- [ ] **Step 8: Run error/reuse tests and verify RED**

Run: `.venv/bin/pytest -q tests/test_minimax_streaming_tts.py -k 'cancel or oversized or failed or reuse'`

Expected: FAIL until the handler owns response-scoped streaming state.

- [ ] **Step 9: Rewrite `MiniMaxTTSHandler`**

Remove HTTP/WAV imports and batch code. Configure the streaming client from existing MiniMax environment values. On first current `TTSInput`, create/start a client; on subsequent current chunks, reuse it and yield audio from `synthesize`; on stale/cancelled input, close and clear it; on `EndOfResponse`, finish/close before yielding `AUDIO_RESPONSE_DONE`. `on_session_end()` and `cleanup()` close active clients idempotently.

- [ ] **Step 10: Lock dependencies and run Task 2 tests**

Run: `uv lock`

Run: `.venv/bin/pytest -q tests/test_minimax_streaming_tts.py tests/test_custom_service_handlers.py tests/test_lm_output_processor.py`

Run: `.venv/bin/ruff check src/speech_to_speech/TTS/incremental_mp3_decoder.py src/speech_to_speech/TTS/minimax_tts_handler.py tests/test_minimax_streaming_tts.py`

Expected: PASS.

- [ ] **Step 11: Commit Task 2**

```bash
git add pyproject.toml uv.lock src/speech_to_speech/TTS/incremental_mp3_decoder.py src/speech_to_speech/TTS/minimax_tts_handler.py tests/test_minimax_streaming_tts.py tests/test_custom_service_handlers.py
git commit -m "feat: stream MiniMax speech synthesis"
```

### Task 3: DeepSeek First-Sentence Flush, Cancellation, and Latency Metrics

**Files:**
- Modify: `configs/tencent-deepseek-minimax.json`
- Modify: `src/speech_to_speech/LLM/base_openai_compatible_language_model.py`
- Modify: `src/speech_to_speech/LLM/chat_completions_language_model.py`
- Modify: `src/speech_to_speech/pipeline/messages.py`
- Modify: `src/speech_to_speech/TTS/minimax_tts_handler.py`
- Modify: `src/speech_to_speech/STT/tencent_asr_handler.py`
- Create: `tests/test_provider_streaming_pipeline.py`
- Modify: `tests/test_chat_completions_backend.py`

**Interfaces:**
- Adds optional monotonic timestamp fields to internal pipeline messages only: `first_partial_at_s`, `final_at_s`, `first_delta_at_s`, and `first_audio_at_s`; these fields never enter persisted conversation content.
- Produces content-free structured logger events with durations and turn/revision only; no provider payload or text.

- [ ] **Step 1: Write first-sentence-before-provider-terminal test**

Use a controlled DeepSeek stream yielding `"First sentence. Second"`, pause before the remaining text/terminal, and drive the real base handler. Assert an `LLMResponseChunk(text="First sentence.")` reaches `LMOutputProcessor` and produces `TTSInput` before terminal release. Assert profile parsing returns `stream_batch_sentences == 1`, streaming true, and thinking disabled.

- [ ] **Step 2: Run the phrase test and verify RED**

Run: `.venv/bin/pytest -q tests/test_provider_streaming_pipeline.py -k first_sentence`

Expected: FAIL because the profile still inherits the three-sentence default.

- [ ] **Step 3: Configure one-sentence streaming and provider-close cancellation**

Set `stream_batch_sentences: 1` in `configs/tencent-deepseek-minimax.json`. Retain punctuation-aware sentence splitting. In `_generate`, when cancellation/staleness is detected, close the OpenAI stream immediately before leaving the iterator; keep the existing `finally` close as idempotent fallback. Add a regression whose fake stream blocks further chunks until `close()` and assert barge-in releases it without emitting stale text.

- [ ] **Step 4: Write content-free latency tests**

Inject a monotonic clock at handler setup boundaries. Assert the first Tencent partial/final, DeepSeek delta, MiniMax PCM block, and barge-in closure record only numeric duration, turn ID, and revision. Capture logs and assert sentinel transcript, API key, signature, audio hex, and raw provider JSON are absent.

- [ ] **Step 5: Run latency/privacy tests and verify RED**

Run: `.venv/bin/pytest -q tests/test_provider_streaming_pipeline.py -k 'latency or privacy or cancel'`

Expected: FAIL because stage timings and immediate stream close are missing.

- [ ] **Step 6: Implement monotonic timing propagation**

Record timestamp once at first useful output in each provider handler. Propagate internal timestamps only as needed to calculate stage durations. Log named numeric values at INFO once per turn and duplicates at DEBUG. Do not attach timestamps or provider IDs to public transcript/audio events. Preserve default construction for all existing internal message callers.

- [ ] **Step 7: Run Task 3 tests and lint**

Run: `.venv/bin/pytest -q tests/test_provider_streaming_pipeline.py tests/test_chat_completions_backend.py tests/test_responses_api_language_model.py tests/test_lm_output_processor.py`

Run: `.venv/bin/ruff check src/speech_to_speech/LLM src/speech_to_speech/pipeline/messages.py tests/test_provider_streaming_pipeline.py`

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add configs/tencent-deepseek-minimax.json src/speech_to_speech/LLM/base_openai_compatible_language_model.py src/speech_to_speech/LLM/chat_completions_language_model.py src/speech_to_speech/pipeline/messages.py src/speech_to_speech/TTS/minimax_tts_handler.py src/speech_to_speech/STT/tencent_asr_handler.py tests/test_provider_streaming_pipeline.py tests/test_chat_completions_backend.py
git commit -m "perf: minimize provider streaming latency"
```

### Task 4: End-to-End Streaming Contract and Operator Documentation

**Files:**
- Modify: `tests/test_provider_streaming_pipeline.py`
- Modify: `README.md`
- Modify: `configs/tencent-deepseek-minimax.json`
- Modify: `.env.example` if present; otherwise create `.env.example` containing names and non-secret descriptions only.

**Interfaces:**
- Consumes all Task 1–3 provider interfaces.
- Produces a fake-provider end-to-end acceptance test proving output timing and stale-output fencing without live credentials.

- [ ] **Step 1: Write the end-to-end fake-provider timing test**

Connect cumulative progressive/final audio to the real Tencent handler, pass final transcription through the real notifier and controlled DeepSeek stream, then pass the first sentence to the real MiniMax handler with a fake WebSocket and real incremental decoder. Gate each fake provider's terminal event and assert partial transcript, LLM sentence, and first PCM block all appear before their respective terminal gates. Trigger barge-in and assert late Tencent/DeepSeek/MiniMax events produce no output.

- [ ] **Step 2: Run the end-to-end test and verify RED if any cross-stage contract is incomplete**

Run: `.venv/bin/pytest -q tests/test_provider_streaming_pipeline.py -k end_to_end`

Expected: FAIL on the first missing timing, lifecycle, or cancellation edge; fix only that demonstrated edge and rerun until PASS.

- [ ] **Step 3: Document exact setup and streaming verification**

Update README with required environment names (`TENCENT_ASR_APP_ID`, SecretID, SecretKey, engine/language; DeepSeek key; MiniMax key/model/voice), the checked-in profile command, the three provider protocols, and a troubleshooting table. Explicitly distinguish partial transcription from final LLM input and streamed provider audio from browser chunking. Include content-free signs of success and a manual barge-in test; never include example secret values.

- [ ] **Step 4: Run focused and full verification**

Run: `.venv/bin/pytest -q tests/test_tencent_realtime_asr.py tests/test_minimax_streaming_tts.py tests/test_provider_streaming_pipeline.py tests/test_custom_service_handlers.py tests/test_chat_completions_backend.py tests/test_lm_output_processor.py`

Run: `.venv/bin/pytest -q`

Run: `.venv/bin/ruff check .`

Run: `.venv/bin/python -m compileall -q src tests`

Run: `git diff --check`

Expected: all commands exit 0. Any live provider test is separately labeled and skipped unless the operator explicitly supplies credentials and authorizes provider calls.

- [ ] **Step 5: Perform privacy and scope review**

Run searches for credential values in tracked files, provider query/signature logging, audio/transcript logging introduced by the diff, `stream: false`, `SentenceRecognition`, and unbounded queues/buffers. Review every match; tests may retain forbidden literals only as negative assertions. Confirm unrelated dirty files remain untouched.

- [ ] **Step 6: Commit Task 4**

```bash
git add README.md .env.example configs/tencent-deepseek-minimax.json tests/test_provider_streaming_pipeline.py
git commit -m "docs: operate provider streaming pipeline"
```

## Final Acceptance

- Tencent partial text is observable while speech is still in progress.
- Tencent final text alone enters DeepSeek.
- DeepSeek first completed sentence enters MiniMax before the LLM stream ends.
- MiniMax first decoded PCM block is emitted before synthesis completes.
- Audio fragments remain ordered with no missing or duplicated PCM across boundaries.
- Barge-in closes all current provider streams and no stale output reaches the client.
- Focused tests, full pytest, Ruff, compileall, diff check, and privacy review pass.
- Live provider latency remains unclaimed until an operator-authorized credentialed run records the content-free metrics.
