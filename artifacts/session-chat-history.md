# Session chat history

Merged user/assistant text from the mechanism-canvas session. Tool calls omitted.

## User

visualize how the framework is built. /mechanism-canvas

## Assistant

I'll turn this repo into a clickable mechanism canvas. First I'll read the canvas skill, SDK, and architecture docs so the blocks match how the framework is actually built.

Next I’ll read the canvas SDK, the construction-path reference, and the repo’s architecture docs before drawing any blocks.

This repo is the Hugging Face cascade, not the qwen-audio-agent stack. I’ll map its real construction path from the architecture docs and source before drawing anything.

I have the product boundary. Next I’ll walk the real `new` order: CLI entry, handler factory, realtime server, and per-connection voice graph.

I have the construction path and protocol surface. Next I’ll write the four-view canvas: system map, request path, protocol machine, and boot order.

The canvas type-check failed on a `React` namespace I shouldn’t import. I’ll type that children prop without it.

The framework is a **cascade that pretends to be one voice**: OpenAI Realtime on the wire, six handler threads per connection inside.

You can open the interactive canvas beside chat: [speech-to-speech mechanism](/Users/humeng/.cursor/projects/Users-humeng-speech-to-speech/canvases/speech-to-speech-mechanism.canvas.tsx)

Four views, each block tied to who constructs it:

- **System map** — click a block for role, mechanism, constructor, wiring, and the invariant a change must not break. Every cited file has an `openFile` button.
- **Request path** — step a spoken turn, tool round-trip, barge-in, or injected text. Live blocks highlight on each beat.
- **Protocol machine** — OpenAI Realtime states (`session.created`, `speech_started`, `function_call_arguments.done`, …) plus the unit/CancelScope machine (`idle` → `draining` → `stuck`).
- **Boot wiring** — the actual `new` order: CLI → parse → per-unit `CancelScope` / `RealtimeService` / handler chain → `PipelineUnit` → `RealtimeServer` → `ThreadManager`.

What the user thinks they are talking to is one assistant at `/v1/realtime`. What actually runs is a pool of isolated `PipelineUnit`s. `conversation.item.create` does not generate; `response.create` does. Interruption cancels the current spoken generation (`CancelScope.generation++`) and keeps the unit, the listen path, and user-side transcripts.

## User

give me more details on how the streaming asr and streaming tts interacts with the current speech-to-speech service. the more details the better.do not change the cavas, directly explain to me

## Assistant

I’ll trace the live ASR and TTS paths in the current service — VAD partials, provider streaming, and how those chunks reach the Realtime send loop — and explain that interaction in detail without changing the canvas.

Streaming ASR and streaming TTS never talk to each other. They share a turn identity, a cancellation counter, and one send loop that always reads transcripts before audio. That is the whole coupling.

## What “streaming” means here

There are two different streaming ASR modes, and they do different jobs.

**Live / progressive ASR** (`--enable_live_transcription`, default on) is a display path. While you are still talking, VAD periodically dumps the growing utterance to STT. STT emits `PartialTranscription`. The notifier copies that to `text_output_queue`. The Realtime send loop turns it into `conversation.item.input_audio_transcription.delta`. The LLM is not started. TTS is not started.

**Soft-final / speculative ASR** is always on in realtime mode, because every `PipelineUnit` owns a `SpeculativeTurnTracker`. When VAD thinks the utterance ended (silence), it emits `VADAudio(mode="final")` with a `turn_id` / `turn_revision`. That final `Transcription` is what starts the LLM, and therefore TTS. The turn can still be reopened if you keep talking. A new revision then races the old one; the old TTS is dropped.

Streaming TTS is also not “the model talks to ASR.” It is: LLM flushes spoken sentences onto `lm_processed_queue` → TTS synthesizes PCM in 512-sample blocks → the send loop batches those into `response.output_audio.delta`.

```text
mic PCM
  → VAD (progressive snapshots + soft-final)
  → STT
      PartialTranscription  → notifier → client deltas only
      Transcription         → service  → Chat + GenerateResponseRequest
  → LLM (stream tokens, flush 1 then N sentences)
  → LMOutputProcessor (text → TTS, tools → client)
  → TTS (stream PCM)
  → send loop (text events first, then audio)
```

## How live ASR is fed

VAD is the clock. The STT thread does not pull the microphone.

On every 512-sample / 16 kHz chunk, `VADHandler.process` runs Silero. Once active speech crosses `min_speech_ms` (default 384 ms), it emits `SpeechStartedEvent` on `text_output_queue` and allocates `turn_id` / `turn_revision`.

If live transcription is on, VAD also enters `_process_realtime` and, while Silero is still triggered, yields a **growing** `VADAudio(mode="progressive")` every `live_transcription_update_interval` (default 0.5 s; Tencent realtime forces 0.2 s). The audio is the whole current buffer, not a delta, concatenated with any `_speculative_audio_prefix` from a reopened turn.

Before enqueueing, VAD drops superseded items already sitting on `spoken_prompt_queue`: any older progressive snapshot for the same turn, and any audio whose revision is no longer latest. STT therefore sees at most one in-flight progressive buffer per turn, plus the later final.

When Silero ends the segment, VAD emits `SpeechStoppedEvent`, stores the audio as `_speculative_audio_prefix`, starts a reopen-grace window (`speculative_reopen_ms`, default 1000 ms; unanswered turns can reopen up to `unanswered_reopen_ms`, 7000 ms), and yields `VADAudio(mode="final")`. In realtime it does **not** clear `should_listen`. Listening stays up so a continuation can reopen the same `turn_id` with `revision+1`.

There is a third VAD emit: **silence prefetch**. The first trailing-silence frame (Silero `temp_end`) yields one progressive snapshot so SentenceRecognition-style ASR can start during `min_silence_ms` instead of waiting for finalize. Tencent’s non-realtime path uses that to overlap the HTTP call with the remaining silence.

`SpeechStartedEvent` from a confirmed barge-in carries `interrupt_response` (default true). A late `speech_started` synthesized only to close a pair at soft-end sets `interrupt_response=False`, so a soft-end does not cancel a reply that has not started yet.

## What each streaming ASR backend does with those snapshots

**Parakeet TDT** (default) is local re-decode, not a provider stream.

- Progressive: `SmartProgressiveStreamingHandler.transcribe_incremental` on the growing buffer. It needs ≥500 ms of new audio. Window grows to 15 s; beyond that, completed sentences become “fixed” and only the last ~2 s plus the tail are re-decoded. Output is `fixed_text + active_text` as one `PartialTranscription`.
- Compute lock: progressive tries the lock with a **10 ms** timeout and skips if the final pass or another progressive is running. Final waits up to **5 s**.
- On Apple Silicon, that lock is the global MLX lock. With `--num_pipelines > 1`, live transcription is turned off at boot so progressive STT does not flood warnings.
- Final: if progressive already fixed sentences, MLX only decodes audio after `fixed_end_time` and concatenates. Otherwise it transcribes the whole utterance. Then the streaming handler resets so timing cannot leak into the next turn.
- While `processing_final` is true, late progressive updates are ignored.

**Tencent realtime** (`TENCENT_ASR_APP_ID` set) is a true provider stream.

- First progressive for a `(turn_id, revision)` opens one WebSocket (`needvad=0`, PCM16). Later progressives send only the **new** samples (`_rt_sent` watermark), in 640-byte frames.
- Each send returns the provider’s current hypothesis as `PartialTranscription`.
- Final sends any remaining PCM, calls `finish()`, then closes the session.
- A new revision closes the old socket and opens a new one. Revisions do not share a voice_id.

**Tencent SentenceRecognition** (no AppId) is not streaming in the WebSocket sense. Progressive/silence-prefetch starts one background HTTP recognize; the final reuses that future if the extra tail is ≤350 ms. Otherwise it recognizes again. Partials are not the live-delta path unless you also have the realtime session.

**Paraformer** will transcribe progressive snapshots if VAD sends them, but it re-runs the whole buffer each time. There is no incremental decoder.

Whisper-family handlers ignore `mode` and only produce a final `Transcription`. Live VAD snapshots still hit them if live transcription is on, which is wasted work; the intended live pair is Parakeet or Tencent.

## What the service does with those STT messages

`TranscriptionNotifier` is the split:

- `PartialTranscription` → `PartialTranscriptionEvent` on `text_output_queue` → **return, yield nothing**. LLM queue stays empty.
- `Transcription` → `TranscriptionCompletedEvent` (even if empty, so the client can close a partial item). Empty text re-enables `should_listen` and does **not** start the LLM.
- In realtime, the notifier does not append Chat. `RealtimeService._on_transcription_completed` does.

The service then:

1. Emits `conversation.item.input_audio_transcription.completed` with duration usage.
2. If this is the same speculative `turn_id` as last time, **replaces** the previous user Chat item instead of appending a second one. Empty replacement deletes the item.
3. If the transcript is non-empty, sets `response_pending = True` and puts `GenerateResponseRequest` on `text_prompt_queue`, carrying `turn_id`, `turn_revision`, and `speech_stopped_at_s`.

That is the only ASR → TTS ignition. Partials never reach this function.

`ConversationHandler.on_partial_transcription` maps each partial to `conversation.item.input_audio_transcription.delta` with a monotonically increasing `content_index`. The protocol does not send “replace the whole hypothesis”; it sends another delta. Clients that want a stable live caption usually treat the latest completed event as truth and treat deltas as ephemeral.

Stale-turn gating: `RealtimeService.dispatch_pipeline_event` drops `PartialTranscriptionEvent` / `TranscriptionCompletedEvent` whose revision is not latest. Assistant text and token usage wait for reopen grace (`is_latest_after_reopen_grace`) so a soft-final that is about to be reopened does not commit a reply.

## How that starts streaming TTS

The LLM handler reads `GenerateResponseRequest`, captures `cancel_scope.generation` as `turn.gen`, and streams.

If `--responses_api_stream` (or the local equivalent) is on:

- First spoken flush is **one sentence** (`first_flush_done` is false, batch limit 1). That is deliberate: TTS can start before the rest of the answer exists.
- Later flushes wait for `stream_batch_sentences` (default 3) so TTS is not restarted on every clause.
- `remove_unspeechable` strips TTS-hostile symbols only on the audio path.
- A tool call flushes any pending spoken text first, then emits the tool on the side channel.
- Every flush and every token checks `is_stale(gen)` and `is_latest(turn_id, revision)`.

`LMOutputProcessor` then splits the chunk:

- `AssistantTextEvent` → `text_output_queue` (transcript + tools for the client).
- `TTSInput` → `lm_processed_queue` only if `response_wants_audio` and text is non-empty. Same `turn_id` / `revision` / `cancel_generation` / `speech_stopped_at_s`.

TTS is therefore sentence-sized, not token-sized. That is the contract in `architecture-contracts.md`: “sentence-sized TTSInput messages.”

Qwen3 (default) and MiniMax both stream PCM after that:

- Capture `generation` at the start of the utterance.
- Check `is_stale` on every provider chunk.
- Resample to 16 kHz, pack into `blocksize` (512 samples, 32 ms).
- Qwen3 trims the initial silent ramp, keeps 40 ms preroll, and can **coalesce** already-queued `TTSInput`s of the same turn so three LLM sentences become one synthesis call if they arrived while the previous call was running.
- MiniMax SSE: play `status=1` hex PCM immediately; skip the aggregated `status=2` copy if incrementals already played. The first chunk is yielded even if it is smaller than `blocksize` so TTFA is not blocked on 32 ms.
- Before synthesizing, MiniMax/Qwen3 call `speculative_turns.commit(turn_id, revision)` once audio is about to be produced. A committed revision cannot be reopened by VAD. That is the moment “this reply is the one.”
- `EndOfResponse` becomes `AUDIO_RESPONSE_DONE` (`b"__RESPONSE_DONE__"`) on the audio queue.

The send loop is the only socket writer:

1. Drain `text_output_queue` first. A new `SpeechStartedEvent` can cancel the current reply before the next audio delta goes out.
2. Drop audio whose `cancel_generation` is stale, or that sits in the discard window.
3. Batch PCM up to 6400 bytes (~200 ms) into one `response.output_audio.delta`.
4. On first audio of a response, set `response_playing` and `should_listen` so barge-in is possible **while TTS is still speaking**.
5. On `AUDIO_RESPONSE_DONE`, `finish_response(completed)`, `cancel_scope.response_done(gen)`, `should_listen.set`.

`response.created` is emitted on the first outbound audio chunk (or on explicit `response.create`), not when STT finished.

## Where they actually interact

### 1. Shared turn identity

VAD stamps `turn_id` / `turn_revision` on every `VADAudio`. That id rides STT → notifier → `GenerateResponseRequest` → LLM chunks → `TTSInput` → (implicitly) the audio generation tag.

Streaming ASR partials use the same ids so the service can ignore a late delta from revision 1 after revision 2 exists. Streaming TTS uses them so a sentence from revision 1 cannot play after revision 2 committed.

### 2. Speculative overlap (the important one)

This is the real ASR↔TTS interaction, and it does not require live transcription.

1. You pause. VAD soft-finalizes revision 1. STT produces a `Transcription`. Service starts the LLM. TTS may start speaking from the first sentence.
2. You continue within the reopen window. VAD confirms a reopen candidate, emits a new `speech_started` (possibly with `reopened=True`), and later a new final with revision 2 and **concatenated** audio (`_speculative_audio_prefix + new segment`).
3. STT re-transcribes the longer utterance. Service **replaces** the user Chat item, enqueues another `GenerateResponseRequest` for revision 2.
4. Tracker: revision 2 is latest. LMOutputProcessor and TTS drop revision 1. If TTS already committed revision 1, VAD would have refused the reopen (`is_committed`). So once you hear the assistant, further speech is a **new turn** (barge-in), not a revision of the old one.
5. If revision 1’s TTS had started but not committed, or LLM was still generating, `is_latest` / reopen grace drops it. `CancelScope` is a separate axis: barge-in increments `generation` even when the turn id is new.

Live progressive ASR sits beside this. It makes the client caption update during step 1’s pause and during step 2’s continuation, but it does not decide whether TTS starts. Only `Transcription` does.

### 3. Barge-in: new ASR cancels old TTS

If a reply is `in_response` or `response_pending` and a new `SpeechStartedEvent` arrives with `interrupt_response` and session `turn_detection.interrupt_response` (default true):

- Send loop emits `response.done` `cancelled` / `turn_detected`.
- `cancel_scope.cancel()` → `generation++`, `discarding=True`.
- Audio queue flushed, but `AUDIO_RESPONSE_DONE` and `SESSION_END` are kept.
- Text queue flushed, but `speech_stopped`, partials, completed transcripts, and token usage are kept.
- LLM/TTS see `is_stale(old_gen)` and abort mid-stream.
- Current-generation audio is never discarded. A leftover discard window cannot swallow the next reply; `response.create` calls `new_response()` to clear it.

That is why the send loop processes text before audio. A streaming ASR `speech_started` must beat a streaming TTS delta already sitting in `output_queue`.

### 4. Compute and queue contention

They share hardware, not APIs.

- Parakeet progressive vs final share one compute lock (or the process-wide MLX lock). Progressive is best-effort; final is mandatory. If TTS is also MLX (Qwen3 mlx-audio) on the same machine, they serialize on that lock. That is why multi-pipeline Mac disables live transcription.
- VAD → STT is a queue. A slow final STT does not block VAD; VAD just drops superseded progressives.
- LLM → TTS is a queue. TTS coalescing exists because the LLM can enqueue several sentence `TTSInput`s while Qwen3 is still synthesizing the first.
- TTS → send loop is a queue. The send loop is the single consumer. Handler threads never write the WebSocket.

### 5. What each side is forbidden to do

- Progressive ASR must not put anything on `text_prompt_queue`. That would start TTS from an unstable hypothesis.
- TTS must not read VAD or STT queues. It only sees `TTSInput` / `EndOfResponse`.
- The client must not see `turn_id`, `turn_revision`, `cancel_generation`, or `SESSION_END`. It sees OpenAI events only.
- Interruption cancels the **current spoken generation**, not the pipeline unit, and not in-flight user transcripts.

## End-to-end timing of one spoken turn with both streams on

1. `input_audio_buffer.append` → 16 kHz / 512-sample chunks on `input_queue`.
2. ~384 ms of active speech → `speech_started` to the client. If a reply was playing, TTS stops here.
3. Every 0.5 s (0.2 s for Tencent realtime) → progressive `VADAudio` → Parakeet/Tencent hypothesis → `transcription.delta`. Still no LLM.
4. Silence ≥ `min_silence_ms` → optional silence-prefetch progressive (Tencent HTTP overlap) → then `speech_stopped` + `VADAudio(final)`.
5. Final STT (Parakeet may reuse fixed sentences from step 3) → `transcription.completed` → Chat user item → `GenerateResponseRequest`.
6. LLM first sentence (~one `sent_tokenize` unit) → `TTSInput` → first PCM → `response.created` + `output_audio.delta`. TTFA logs “last speech detected to first speech out” using `speech_stopped_at_s` from the final VAD audio.
7. Later LLM sentences batch (default 3) or get coalesced by Qwen3. More deltas.
8. `EndOfResponse` → `AUDIO_RESPONSE_DONE` → `response.output_audio.done` + `response.done`.
9. If you resume during the reopen window before TTS `commit`, go back to step 3 with `revision+1` and replace the Chat user item. Old deltas and old PCM die on the tracker / CancelScope gates.

The default install (`parakeet-tdt` + `responses-api` stream + `qwen3`) is exactly this: local growing-window ASR for captions, soft-final ASR to start the model, sentence-streamed LLM, chunk-streamed TTS, and a send loop that lets the next `speech_started` preempt the speaker.

## User

more details on tencent streaming asr and minimax streaming tts should be considered as an extra complementary

## Assistant

I'll read the Tencent realtime ASR and MiniMax streaming TTS adapters in full so I can explain how they sit beside the default local cascade.

Tencent streaming ASR and MiniMax streaming TTS are **optional hosted adapters**. They do not change the cascade. They sit in the same STT and TTS slots as Parakeet and Qwen3, convert at the adapter boundary, and reuse VAD progressive snapshots, `turn_id` / `turn_revision`, `CancelScope`, and the Realtime send loop.

The default product is still local: Silero → Parakeet → OpenAI-compatible LLM → Qwen3. The complementary stack is the checked-in profile `configs/tencent-deepseek-minimax.json`: Tencent in, DeepSeek in the LLM slot, MiniMax out. You install Tencent with `speech-to-speech[tencent-asr]`. MiniMax is built-in; it only needs env credentials.

```text
same Realtime client / same PipelineUnit
        │
        ├─ VAD (unchanged)
        │
        ├─ STT slot:  Parakeet (default)  |  Tencent (extra)
        ├─ LLM slot:  responses-api       |  DeepSeek chat-completions
        └─ TTS slot:  Qwen3 (default)     |  MiniMax T2A (extra)
```

Credentials stay in `.env.local`. The JSON profile only picks `--stt tencent` and `--tts minimax`.

---

## Why they exist

The local path already “streams” in a pipeline sense: VAD snapshots, Parakeet re-decodes a growing window, the LLM flushes sentences, Qwen3 yields PCM. That still waits on local GPU/MLX.

Tencent + MiniMax move the heavy recognition and synthesis off-box, and they were wired so those provider RTTs **overlap** speech and playback instead of stacking after silence:

| Stage | Default local | This extra pair |
|---|---|---|
| Live captions | Parakeet re-decodes the whole growing buffer every 0.5 s | Tencent WebSocket gets **new PCM only**, every 0.2 s |
| Final transcript | Local full-utterance decode | WebSocket `{"type":"end"}`, or SentenceRecognition that started during VAD silence |
| First audio | Qwen3 local codec stream | MiniMax SSE hex PCM, first frame even if &lt; 512 samples |

They still obey the same rule: **partials never start the LLM**. Only a final `Transcription` does. MiniMax still only sees sentence-sized `TTSInput`s.

---

## Tencent ASR: two complementary modes in one handler

`TencentASRHandler` is one class with a switch.

`TENCENT_ASR_SECRET_ID` + `TENCENT_ASR_SECRET_KEY` are required. That is enough for **SentenceRecognition** (HTTP, one utterance).

`TENCENT_ASR_APP_ID` turns on **realtime WebSocket**. `enable_tencent_realtime_transcription()` then forces `enable_live_transcription=True` and, if the interval is still the 0.5 s Parakeet default, drops it to **0.2 s**. The profile sets `enable_live_transcription: false` on disk; boot overrides that when the AppId is present. That is how VAD starts emitting `VADAudio(mode="progressive")` for Tencent even though the JSON looks like “finals only.”

`process()` tries realtime first. A failed **progressive** update logs and returns (no SentenceRecognition spam mid-utterance). A failed **final** closes the socket and falls back to SentenceRecognition for that turn.

### Mode A — realtime WebSocket (the true stream)

This is a **provider** stream, not a local re-decode.

1. First progressive for `(turn_id, revision)` builds a signed URL and opens one `TencentRealtimeASRSession` (one `voice_id`).
2. Later progressives send only samples after `_rt_sent`. VAD still gives the **full growing buffer**; the adapter diffs it.
3. PCM is clipped little-endian int16, then sliced into **6400-byte** frames (200 ms at 16 kHz mono 16-bit). That is Tencent’s frame size, not the pipeline’s 512-sample VAD chunk.
4. Each `send_pcm` writes raw bytes (not JSON), then drains inbound events for 50 ms and returns `current_text()`.
5. Non-empty text becomes `PartialTranscription` → notifier → `transcription.delta` on the Realtime socket. DeepSeek is not called.
6. On `mode="final"`, remaining samples are sent, then `{"type":"end"}`. `finish()` waits up to 2 s for `final=1`, but returns earlier once a stable sentence is in hand and the socket has gone quiet. Then the session is closed.
7. That text is the one `Transcription` that starts the LLM.

URL shape (`tencent_realtime.py`):

- `wss://asr.cloud.tencent.com/asr/v2/<app_id>?…`
- HMAC-SHA1 over the **unquoted** `host/path/query`, same layout as Tencent’s speech SDK. Values are URL-encoded only in the request.
- `needvad=0` — Silero already decided speech; Tencent must not VAD again.
- `voice_format=1` — PCM.
- `engine_model_type` from `TENCENT_ASR_ENGINE` (default `16k_zh`).
- Filters off (`filter_dirty/modal/punc=0`), `word_info=0`, `convert_num_mode=0`.

Inbound result handling:

- `code` other than `0`/`None` raises (no retry of bad credentials).
- `result.slice_type == 2` with text: that slice is **stable**. It is appended to `stable_parts` and `partial` is cleared.
- Any other text: unstable hypothesis, stored as `partial`.
- `current_text()` prefers concatenated stable parts; if none, the live partial.

A new `turn_revision` closes the old socket and opens a new `voice_id`. Revisions do not share a stream. `on_session_end` / `cleanup` also close, so a client disconnect cannot leak a billed WebSocket into the next claim.

Handshake: `start()` connects (`open_timeout=5s`) and must receive one JSON event within 5 s or it raises. That is why a dead AppId fails fast on the first progressive, not after the user stops.

### Mode B — SentenceRecognition (the complementary fallback)

No AppId, or realtime final failed. This is **not** a caption stream.

- Progressive / silence-prefetch: if the buffer is ≥ 0.4 s, submit **one** background HTTP `SentenceRecognition` on a single-worker pool. A second progressive for the same turn is ignored while that future is in flight.
- Final: if the extra tail is ≤ **350 ms** (typical `min_silence_ms` plus a couple of frames), reuse that future. Otherwise recognize the full final buffer again.
- Payload: base64 PCM, `SourceType=1`, `VoiceFormat=pcm`, engine `16k_zh` by default.
- Hard limit: **60 s** per utterance. Longer audio is rejected, not truncated.
- HTTP keep-alive when the SDK supports it; timeout `TENCENT_ASR_TIMEOUT_S` (default 10 s).

VAD’s **silence-prefetch** exists mainly for this mode: the first Silero `temp_end` yields one progressive snapshot so the HTTP RTT runs during `min_silence_ms` instead of after finalize. Realtime mode does not need that trick; it has already been streaming PCM for the whole utterance.

The checked-in profile uses `speech_pad_ms: 80` (default VAD pad is 30 ms in code, older cascade used 500 ms). Less pad means a smaller Tencent payload and less tail after you stop.

---

## MiniMax TTS: complementary streaming out

`MiniMaxTTSHandler` replaces Qwen3 in the TTS slot. It is constructed with `cancel_scope` and `speculative_turns` in `setup_kwargs` (Qwen3 gets those via mutated handler kwargs; MiniMax takes them explicitly). Same queues: `lm_processed_queue` in, `send_audio_chunks_queue` out.

It does **not** speak Tencent’s protocol. It only accepts `TTSInput` / `EndOfResponse` and yields 16 kHz mono int16 blocks, then `AUDIO_RESPONSE_DONE`.

### Request

- `POST` `MINIMAX_TTS_ENDPOINT`, default `https://api.minimax.io/v1/t2a_v2` (global). China-platform keys need `https://api.minimaxi.com/v1/t2a_v2`. Mixing key and host fails with an application error, not a helpful HTTP 401.
- Bearer `MINIMAX_TTS_API_KEY`. Voice is `MINIMAX_TTS_VOICE_ID` (must be a voice that key owns).
- Model default `speech-2.8-turbo`, `language_boost` default `auto`.
- Streaming default on (`MINIMAX_TTS_STREAM=true`): `Accept: text/event-stream`, `stream: true`, `output_format: hex`, `audio_setting.format: pcm`, 16 kHz, mono.
- Non-stream: `format: wav`, one JSON body.

Setup does a `HEAD` to the origin so TLS + keep-alive are warm before the first spoken turn. The client is an `httpx.Client` with 30 s request timeout, 5 s connect, 8 keep-alive connections.

### Streaming decode (the latency path)

MiniMax sends SSE `data:` frames. Each JSON has `data.audio` (hex) and `data.status`.

- **`status=1`**: incremental PCM. Decode immediately, play immediately.
- **`status=2`**: aggregated copy of the whole utterance. Saved aside. If any `status=1` already played, it is **skipped** so you do not hear the sentence twice.
- If the stream only ever sent `status=2` (no incrementals), that hex is used as the whole utterance (WAV or raw PCM).
- `base_resp.status_code` is checked on **every** event. HTTP 200 with a nonzero MiniMax code is still a failure.

Hex is reassembled across frames. A leftover nibble or odd last byte is held until the next event (`_decode_hex_pcm`). That is why split frames still produce valid int16.

Playback packing:

- `blocksize` 512 samples (32 ms), same as the rest of the pipeline.
- **First** incremental frame is yielded even if it is shorter than 512. That was the old bottleneck: TTFA waited for a full block.
- Later frames emit only complete 512-sample multiples; remainder stays in `pending`.
- After the stream ends, leftover is **padded** to 512 so the send loop is not left with a ragged tail.

Cancellation is checked:

- before handling each SSE event,
- before each 512-sample emit,
- after the stream if `generation` went stale mid-request.

`speculative_turns.commit(turn_id, revision)` runs **before** the HTTP POST. Once MiniMax is asked to speak a revision, VAD will not reopen that turn. Further speech is barge-in (new turn + `CancelScope`), not a revised transcript of the same turn. That is stricter than “wait until first PCM”: the network call itself commits.

`EndOfResponse` still becomes `AUDIO_RESPONSE_DONE` only if the revision is latest after reopen grace. A stale end-of-response must not close a newer reply.

Non-stream path (`MINIMAX_TTS_STREAM=false`) waits for one hex WAV, validates mono / 16-bit / 16 kHz, then emits padded 512-sample chunks. Useful for debugging; it puts the entire synthesis RTT back in front of first audio.

---

## How they interact with the current service (as extras)

They do not talk to each other. The service in the middle is unchanged.

**VAD → Tencent.** With AppId, VAD yields a growing progressive buffer every 200 ms. Tencent sends the delta. Without AppId, VAD’s silence-prefetch starts one HTTP recognize during trailing silence. Soft-final still produces `VADAudio(mode="final")` with the same `turn_id` / `revision` the rest of the unit uses.

**Tencent → notifier → RealtimeService.** Same split as Parakeet: `PartialTranscription` is captions only; `Transcription` appends/replaces Chat and enqueues `GenerateResponseRequest`. Empty finals still emit `transcription.completed` and do not start DeepSeek.

**LLM → MiniMax.** The profile sets `stream_batch_sentences: 1` and DeepSeek `responses_api_stream: true`. Combined with CJK-aware `split_spoken_units()` (`。！？；…`), the first Chinese sentence reaches MiniMax as soon as it closes. That is the complementary half of MiniMax SSE: streaming TTS is wasted if the LLM still batches three sentences.

**MiniMax → send loop.** Same `_send_loop_for`: text events first (so Tencent-driven `speech_started` can cancel MiniMax), then PCM batched to 6400 bytes, generation-tagged discard, `AUDIO_RESPONSE_DONE` → `response.done`. MiniMax’s short first frame is why `response.created` can fire before a full 32 ms block exists.

**Fallback composition.** If the Tencent WebSocket dies on finalize, SentenceRecognition still produces a `Transcription` with the same metadata. MiniMax never knows which ASR mode ran. If MiniMax streaming fails, that is a TTS error for that `TTSInput`; there is no automatic fallback to Qwen3.

**Session cleanup.** `SESSION_END` hits `TencentASRHandler.on_session_end` → close WebSocket, drop speculative future. MiniMax `cleanup()` closes the owned `httpx` client. Injected test clients are left open.

**What a smoke test does not prove.** `scripts/smoke_custom_services.py` calls Tencent on a **final** buffer and MiniMax on a canned sentence. That checks credentials and formats. It does not exercise progressive WebSocket, silence prefetch, sentence flush, cancellation, or the send loop. The complementary path only shows up in a full realtime turn.

---

## Practical boundary

Treat them as extras that **implement the existing contracts**, not as a second product:

- Encoding stays in the adapter: float32 → clipped PCM16 for Tencent; hex SSE → int16 blocks for MiniMax.
- `turn_id` / `turn_revision` / `speech_stopped_at_s` pass through unchanged.
- Do not retry after interrupt or a newer revision.
- Do not retry bad keys, bad voices, or nonzero `base_resp.status_code`.
- Never log Authorization, hex bodies, or PCM.

Enable the pair with the profile plus env, not by forking the gateway:

```bash
pip install "speech-to-speech[tencent-asr]"
# .env.local: Tencent secrets, optional TENCENT_ASR_APP_ID, MiniMax key/voice/endpoint
speech-to-speech configs/tencent-deepseek-minimax.json
```

Without `TENCENT_ASR_APP_ID` you still get Tencent ASR, but only the silence-prefetch + SentenceRecognition complement. With it, Tencent becomes a live PCM stream and MiniMax remains the matching live PCM speaker — both still hanging off the same six-thread unit as Parakeet and Qwen3.

## User

tencent id is TENCENT_ASR_APP_ID=<redacted>

## Assistant

I'll put that App ID into the ignored local env file so Tencent realtime ASR can turn on, without printing the value.

`TENCENT_ASR_APP_ID` is already set in `.env.local`, and that file is gitignored. No change was needed.

With that App ID present, `speech-to-speech configs/tencent-deepseek-minimax.json` will:

- turn **live transcription on** even though the JSON sets it `false`
- drop the VAD snapshot interval to **0.2 s**
- open Tencent’s realtime WebSocket (`needvad=0`, PCM) instead of waiting for SentenceRecognition after you stop talking

Load the file before launch so the process actually sees it:

```bash
set -a && source .env.local && set +a
speech-to-speech configs/tencent-deepseek-minimax.json
```

You want the log line `Tencent ASR realtime WebSocket enabled`. If the App ID is unset, it stays on the HTTP SentenceRecognition fallback.

The App ID is an account identifier. Keep `TENCENT_ASR_SECRET_ID` / `TENCENT_ASR_SECRET_KEY` only in `.env.local`; do not commit or paste those.

## User

add more details on openai realtime event protocol interactions of streaming asr and streaming tts

## Assistant

I’ll trace the exact OpenAI Realtime events that streaming ASR and TTS emit and consume, including field names and how they interleave on the wire.

Streaming ASR and streaming TTS never appear as their own event types. The client only sees OpenAI Realtime names. Tencent and MiniMax are hidden behind `conversation.item.input_audio_transcription.*` and `response.output_audio.*`.

The send loop is the serializer: **text-side events always go out before the next audio batch**. That is why a live ASR `speech_started` can close a MiniMax reply before the next `output_audio.delta`.

---

## What the client actually sends

Inbound events that matter for the streams:

| Client event | What it does for streaming ASR/TTS |
|---|---|
| `session.update` | Deep-merges instructions, tools, voice, `audio.input.turn_detection` (`server_vad`, `interrupt_response`). Does **not** start ASR. A `transcription` session type is rejected (`invalid_session_type`). |
| `input_audio_buffer.append` | WebSocket-only. Base64 PCM → resample to 16 kHz → 512-sample chunks on VAD. This is the only ASR fuel. WebRTC rejects it (`invalid_event_for_transport`); audio is on the media track. |
| `input_audio_buffer.commit` | Bookkeeping. Empty buffer → `input_audio_buffer_commit_empty`. Does not finalize Tencent. Silero does. |
| `response.create` | Explicit generate. Optional. The VAD→STT path already enqueues `GenerateResponseRequest` on `transcription.completed`. |
| `response.cancel` | Client barge-in. Same cancel path as VAD: `CancelScope.cancel()`, flush queues, `response.done` `cancelled` / `client_cancelled`. |
| `output_audio_buffer.clear` | WebRTC-only. Drops unplayed RTP. Over WebSocket, unplayed audio is client-side. |
| `conversation.item.create` | Injects text or `function_call_output`. **Does not generate** and does not touch ASR/TTS streams. |

Live transcription is **not** a protocol flag. Tencent App ID (or `--enable_live_transcription`) makes VAD emit progressive audio. The client just keeps appending PCM.

---

## Streaming ASR on the wire

Internal path: Tencent `PartialTranscription` / `Transcription` → `TranscriptionNotifier` → `text_output_queue` → `RealtimeService.dispatch_pipeline_event`.

### 1. `input_audio_buffer.speech_started`

Emitted when Silero crosses `min_speech_ms`, not when the first Tencent partial arrives.

```json
{
  "type": "input_audio_buffer.speech_started",
  "event_id": "event_…",
  "audio_start_ms": 1280,
  "item_id": "item_…"
}
```

This **allocates the user item id**. Every later transcription event for this utterance reuses `speculative_input_item_id`.

If a MiniMax reply is `in_response` and `interrupt_response` is on, the same dispatch **first** emits:

- `response.output_audio.done`
- `response.done` with `status: "cancelled"`, `status_details.reason: "turn_detected"`

then `speech_started`. The client sees the old response close before the new speech item starts.

A **reopen** (same `turn_id`, new revision, you continued after a soft-end) reuses that `item_id`. `content_index` resets to 0. The client should treat later deltas as a new hypothesis on the **same** conversation item, not a second user turn.

A brand-new barge-in (assistant already `commit`ted) gets a new `item_id` and increments `response_usage.turns`.

### 2. `conversation.item.input_audio_transcription.delta`

Each Tencent progressive (≈ every 200 ms with App ID) or Parakeet live decode becomes one of these:

```json
{
  "type": "conversation.item.input_audio_transcription.delta",
  "event_id": "event_…",
  "item_id": "item_…",
  "content_index": 0,
  "delta": "你好世界"
}
```

Important mismatch with hosted OpenAI:

- Hosted GA `delta` is usually an **increment** (next tokens).
- This server puts Tencent’s **current full hypothesis** in `delta` (`voice_text_str` / Parakeet `fixed + active`).
- `content_index` increments on **every** partial (`_next_input_content_index`). It is a sequence number, not “which audio content part.”

So a Tencent stream looks like:

```text
delta content_index=0  "你"
delta content_index=1  "你好"
delta content_index=2  "你好世界"
```

not `"你"` + `"好"` + `"世界"`. Clients that concatenate deltas will duplicate text. The safe UI rule: **display the latest `delta` until `completed` arrives; then display `transcript`.**

Stale revisions are dropped in `dispatch_pipeline_event` before this event is built. A late Tencent partial for revision 1 after revision 2 exists never becomes a protocol delta.

On barge-in flush, these events are **kept** (`_keep_user_text_event`). The client can still see you talking after the assistant was cancelled.

### 3. `input_audio_buffer.speech_stopped`

```json
{
  "type": "input_audio_buffer.speech_stopped",
  "event_id": "event_…",
  "audio_end_ms": 3520,
  "item_id": "item_…"
}
```

This is VAD soft-end, not Tencent `final=1`. Duration is stored on `ConnState.input_audio_duration_s` for the later usage object. Tencent may still be draining its WebSocket.

Also preserved across interrupt flushes.

### 4. `conversation.item.input_audio_transcription.completed`

Built in `ConversationHandler.on_transcription_completed` after Tencent `finish()` (or SentenceRecognition):

```json
{
  "type": "conversation.item.input_audio_transcription.completed",
  "event_id": "event_…",
  "item_id": "item_…",
  "content_index": 0,
  "transcript": "你好世界。",
  "usage": { "type": "duration", "seconds": 2.24 }
}
```

`content_index` is **always 0** here, even if deltas went 0,1,2. `item_id` is the same as `speech_started`. `usage.seconds` is the VAD duration, not Tencent billable time.

This event is the **protocol ignition for streaming TTS**:

1. Emit `completed` to the client.
2. Append or **replace** the user Chat item if this is the same speculative `turn_id`.
3. If `transcript` is nonempty, set `response_pending = true` and `text_prompt_queue.put(GenerateResponseRequest)`.

There is no `response.create` on the socket for a normal spoken turn. The client sees `transcription.completed` and then, some time later, `response.created` when MiniMax’s first PCM is encoded.

Empty transcript: `completed` still goes out (so the client can close the partial item), `should_listen` is set, **no** `GenerateResponseRequest`. You will not get a ghost MiniMax reply.

A reopen final for the same turn emits **another** `completed` on the **same** `item_id` with a longer `transcript`. That is how a revised Tencent result shows up in protocol form.

---

## Streaming TTS on the wire

Internal path: LLM sentence → `AssistantTextEvent` + `TTSInput` → MiniMax PCM on `output_queue` → send loop.

Two independent protocol streams share one `response_id` / output `item_id`:

- **Transcript stream** from `LMOutputProcessor` (text queue)
- **Audio stream** from MiniMax (audio queue)

They are not one multiplexed event. They interleave because the send loop prefers the text queue.

### 5. `response.created`

Two ways:

**Explicit.** Client `response.create` → `ResponseHandler.handle_response_create` immediately returns `response.created` with `status: "in_progress"` and allocates `resp_…` / `item_…`. Also calls `cancel_scope.new_response()` so a leftover discard window cannot swallow MiniMax.

**Implicit (normal Tencent turn).** `current_response_id` is still `None` until the first MiniMax PCM. `AudioHandler.begin_audio_response` emits `response.created` **on that first audio chunk**, then the delta. WebRTC sends only this bookkeeping on `oai-events`; PCM goes on RTP.

So with Tencent + MiniMax you typically see:

```text
transcription.completed     ← Tencent final
… silence while DeepSeek + MiniMax TTFA …
response.created            ← first MiniMax frame
response.output_audio.delta
```

not `response.created` at STT time. `response_pending` is true in between; a barge-in in that window still cancels (no audio yet).

### 6. `response.output_audio_transcript.done` (not token deltas)

For audio responses, each LLM sentence/`TTSInput` text becomes:

```json
{
  "type": "response.output_audio_transcript.done",
  "event_id": "event_…",
  "response_id": "resp_…",
  "item_id": "item_…",
  "output_index": 0,
  "content_index": 0,
  "transcript": "你好，我是助手。"
}
```

This is **not** `response.output_audio_transcript.delta`. The server does not stream assistant tokens on the audio path. With `stream_batch_sentences: 1` (Tencent profile) you get **one `.done` per Chinese sentence**, possibly several per `response_id`, each carrying that sentence’s text only — not the cumulative answer.

Text-only responses (`response_wants_audio` false) use `response.output_text.delta` instead, and a single `response.output_text.done` at close.

Tools on the same `AssistantTextEvent` become `response.function_call_arguments.done` after the transcript event, same `response_id`, incrementing `output_index`.

`on_assistant_text` also `commit`s the speculative revision. After the first assistant sentence is protocol-visible, VAD will not reopen that turn. Further Tencent speech is a new `item_id` + interrupt, not another `completed` on the old item.

### 7. `response.output_audio.delta`

WebSocket only:

```json
{
  "type": "response.output_audio.delta",
  "event_id": "event_…",
  "response_id": "resp_…",
  "item_id": "item_…",
  "output_index": 0,
  "content_index": 3,
  "delta": "<base64 PCM16>"
}
```

- `delta` is base64 of **pipeline 16 kHz PCM**, resampled to `session.audio.output.format.rate` if the client asked for another rate.
- Send loop batches MiniMax’s 512-sample (or short first) chunks up to **6400 bytes** (~200 ms) per event. One MiniMax SSE frame can become one delta, or several MiniMax frames can merge.
- `content_index` increments per **encoded batch**, via `_next_content_index`. These are output-side indexes, unrelated to ASR `content_index`.
- First audio of a response sets `response_playing` and `should_listen`, so Tencent/VAD can barge-in **while MiniMax is still streaming**.

WebRTC: **no** `output_audio.delta`. `begin_audio_response` may still send `response.created` on the data channel; PCM is 20 ms RTP (Opus at 48 kHz after resample). `output_audio_buffer.clear` flushes the server jitter buffer.

### 8. Close: `response.output_audio.done` + `response.done`

When MiniMax’s handler yields `AUDIO_RESPONSE_DONE` (`EndOfResponse`):

```text
response.output_audio.done     (any terminal status, if the response wanted audio)
response.done                  status=completed | cancelled | failed
```

Then deferred `conversation.item.create`s (tool outputs that arrived mid-MiniMax) flush as `conversation.item.created`.

Cancel reasons on `response.done`:

| Reason | Trigger |
|---|---|
| *(none / completed)* | MiniMax stream finished, sentinel observed |
| `turn_detected` | New `speech_started` while in/pending response |
| `client_cancelled` | Client `response.cancel` |

`finish_response` always emits `output_audio.done` for audio responses, including cancel. The client must stop playback on that, not wait for more deltas. Over WebSocket, leftover client-buffered PCM is the client’s problem. Over WebRTC the server also `discard_pending_audio()`.

A **stale** MiniMax sentinel (`cancel_generation` old) does **not** emit `response.done`. It only `response_done()`s the CancelScope and re-enables listen. The response was already closed by the barge-in `finish_response`.

---

## One Tencent + MiniMax turn, event order

```text
── you speak ──────────────────────────────────────────────
C  input_audio_buffer.append          (many; or WebRTC RTP)
S  input_audio_buffer.speech_started  item_id=item_u1
S  conversation.item.input_audio_transcription.delta   # "你"
S  conversation.item.input_audio_transcription.delta   # "你好"
S  input_audio_buffer.speech_stopped
S  conversation.item.input_audio_transcription.completed
      transcript="你好。"
      ← GenerateResponseRequest (not a client event)

── DeepSeek streams, first sentence ───────────────────────
S  response.output_audio_transcript.done
      transcript="你好，我在。"     # text queue, may beat first PCM

── MiniMax first SSE frame ────────────────────────────────
S  response.created                   status=in_progress
S  response.output_audio.delta        # maybe < 32 ms
S  response.output_audio.delta        …
S  response.output_audio_transcript.done   # second sentence, if any
S  response.output_audio.delta        …
S  response.output_audio.done
S  response.done                      status=completed
```

`C` = client, `S` = server. `response.created` can sit **after** the first `output_audio_transcript.done` if DeepSeek’s sentence hit the text queue before MiniMax’s first PCM hit the audio queue. Clients must tolerate transcript-before-audio and audio-before-transcript.

---

## How the two streams preempt each other

The send loop’s one rule: drain `text_output_queue` before taking from `output_queue`.

**ASR cancels TTS**

1. MiniMax deltas are in flight (`in_response` or `response_pending`).
2. VAD `SpeechStartedEvent` is dequeued first.
3. Protocol: `output_audio.done` + `response.done(cancelled, turn_detected)` + `speech_started`.
4. `cancel_scope.cancel()`, flush MiniMax PCM (keep `AUDIO_RESPONSE_DONE` / `SESSION_END`), flush assistant text, **keep** `speech_stopped` / transcription deltas / completed.
5. Later MiniMax chunks with the old `cancel_generation` are dropped; no extra deltas after `response.done`.

**Soft-end + reopen (same user item)**

1. `speech_stopped` + `transcription.completed` (revision 1) may already have started DeepSeek (`response_pending`).
2. You continue. New `speech_started` with `reopened=true` **reuses `item_u1`**. If MiniMax has not `commit`ted yet, interrupt may be false on that synthesized start; the tracker drops revision-1 assistant events.
3. New deltas / a second `completed` land on **the same `item_id`**.
4. A new `GenerateResponseRequest` (revision 2) produces a new `response_id`. The client sees two `response.created` if revision 1 already created one; the first should have been cancelled or dropped before audio.

**TTS does not emit ASR events.** MiniMax never writes `transcription.*`. The only coupling back to ASR is `should_listen.set` on first audio and on done, plus CancelScope when you speak over it.

---

## IDs the client should key on

| ID | Lifetime | Used by |
|---|---|---|
| `session_id` | Connection (`session.created`) | Not on media events |
| `item_id` (user) | One spoken utterance; reused on reopen | `speech_*`, `transcription.delta/completed` |
| `content_index` (input) | Increments per **partial** | Only on `transcription.delta` |
| `response_id` | One generate (implicit or `response.create`) | All `response.*` |
| `item_id` (assistant) | That response’s output item | `output_audio.*`, transcript `.done`, tools |
| `content_index` (output) | Increments per audio **batch** | `output_audio.delta` |

Hidden: `turn_id`, `turn_revision`, `cancel_generation`, Tencent `voice_id`, MiniMax SSE `status`. Those never appear on the socket.

---

## WebSocket vs WebRTC (same JSON events, different audio)

| | WebSocket `/v1/realtime` | WebRTC `POST /v1/realtime/calls` |
|---|---|---|
| In ASR | `input_audio_buffer.append` | RTP → `append_pcm` (append event forbidden) |
| Out TTS | `response.output_audio.delta` base64 | RTP frames; `response.created` on `oai-events` only |
| Barge-in leftover | Client must stop playback | Server `output_audio_buffer.clear` + auto flush on cancel |
| `session.created` | On accept | When data channel opens |

Tencent and MiniMax do not know which transport you used. They only see VAD float audio and `TTSInput` text.

---

## What a correct client should do

1. On `transcription.delta`: **replace** the live caption, do not concatenate.
2. On `transcription.completed`: freeze that caption; treat it as the turn’s user text. A second `completed` on the same `item_id` is a speculative revision — replace again.
3. Do not wait for `response.create` after you speak. Wait for `response.created` or the first audio/transcript event.
4. On `output_audio_transcript.done`: append that sentence to the assistant caption (each event is one flush, not the full answer).
5. On `output_audio.delta`: decode base64 as PCM16 at the negotiated rate; start playback immediately (first frame may be short).
6. On `speech_started` while playing: if you also get `response.done` `turn_detected`, stop playback. Over WebRTC, expect the server to have cleared its buffer; you may also send `output_audio_buffer.clear`.
7. `conversation.item.create` after tools does not restart MiniMax. Send `response.create` if you want the result spoken.

That is the whole protocol surface of the two streams: Tencent is `speech_*` + `input_audio_transcription.{delta,completed}`; MiniMax is `response.created` + `output_audio.delta` + `output_audio_transcript.done` + `{output_audio.done, response.done}`. Everything else is the cascade deciding *when* those events are legal, not extra event names.

## User

this is tencent streaming asr doc https://www.tencentcloud.com/document/product/1118/53937  and this is minimax streaming tts doc https://platform.minimax.io/docs/api-reference/speech-t2a-websocket given these context, polish your last answer

## Assistant

I’ll read the official Tencent realtime ASR and MiniMax T2A docs, then rewrite the last protocol answer so event names and field mappings match those specs.

The client still only sees OpenAI Realtime names. Tencent and MiniMax stay behind adapters. The official docs make the **inner** sockets precise: [Tencent realtime ASR (WebSocket)](https://www.tencentcloud.com/document/product/1118/53937) and [MiniMax T2A WebSocket](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket). This repo implements Tencent’s WebSocket ASR as documented. It does **not** implement MiniMax’s `wss://…/ws/v1/t2a_v2`; MiniMax here is HTTP SSE `POST /v1/t2a_v2`. The hex-audio + final-frame idea is the same. The session events are not.

Three layers, never mixed on one socket:

```text
OpenAI Realtime client
    ↕  JSON events on /v1/realtime  (or oai-events + RTP)
speech-to-speech send loop / RealtimeService
    ↕  PartialTranscription / Transcription / TTSInput / PCM
Tencent wss://asr.cloud.tencent.com/asr/v2/<appid>
MiniMax https://…/v1/t2a_v2   (SSE; not the T2A WebSocket)
```

---

## What the Realtime client sends

Inbound events that feed the two streams:

| Client event | Role |
|---|---|
| `session.update` | Instructions, tools, voice, `audio.input.turn_detection`. Does not start Tencent. A `transcription` session type is rejected. |
| `input_audio_buffer.append` | WebSocket-only. Base64 PCM → 16 kHz / 512-sample VAD chunks. WebRTC forbids this (`invalid_event_for_transport`); audio is RTP. |
| `input_audio_buffer.commit` | Empty → `input_audio_buffer_commit_empty`. Does **not** send Tencent `{"type":"end"}`. Silero does. |
| `response.create` | Optional generate. A normal spoken turn already enqueues `GenerateResponseRequest` on `transcription.completed`. |
| `response.cancel` | Same cancel path as barge-in: `CancelScope`, flush, `response.done` `client_cancelled`. |
| `output_audio_buffer.clear` | WebRTC-only. Drops unplayed RTP. |
| `conversation.item.create` | Context / tool output. Does not generate and does not touch either provider stream. |

Live ASR is not a Realtime flag. `TENCENT_ASR_APP_ID` makes VAD emit progressive audio. The client just keeps appending PCM.

---

## Streaming ASR: Tencent WebSocket → OpenAI Realtime

Official call: handshake, then recognition. Both stages return JSON. After handshake `code=0`, the client uploads **binary** audio and later a text `{"type":"end"}`. That is exactly what `TencentRealtimeASRSession` does ([Tencent Real-Time Speech Recognition (WebSocket)](https://www.tencentcloud.com/document/product/1118/53937)).

### Handshake (never visible to the Realtime client)

URL: `wss://asr.cloud.tencent.com/asr/v2/<appid>?…`

This repo signs it the official way: HMAC-SHA1 over the **unquoted** `host/path/query`, then URL-encodes `signature`. Query fields we set:

| Official param | This repo | Why |
|---|---|---|
| `engine_model_type` | `TENCENT_ASR_ENGINE` default `16k_zh` | 16 kHz Mandarin engine |
| `voice_format` | `1` (pcm) | Official: `1` = pcm |
| `needvad` | `0` | Silero already segmented; Tencent must not VAD again |
| `voice_id` | new UUID per `(turn_id, revision)` | Official: client-generated unique stream id |
| `filter_*` / `word_info` / `convert_num_mode` | `0` | No extra rewriting; we want raw `voice_text_str` |
| `secretid`, `timestamp`, `expired`, `nonce`, `signature` | signed in `build_realtime_url` | Handshake ACK is `code=0` |

A new Realtime user turn (or a new speculative revision) is a **new Tencent `voice_id`**. Revisions do not share a stream.

Official constraints that matter on our VAD clock:

- Audio: 16 kHz, 16-bit, **mono**. We convert float32 → clipped LE PCM16 at the adapter.
- Recommended send: **40 ms** packets, 1:1 realtime → **1280 bytes** PCM at 16 kHz. Gap **> 6 s** → error `4008` and disconnect.
- We slice deltas into **6400-byte** frames (200 ms). That is 5 official packets per write. Because we only send **new** samples (`_rt_sent`), a live speaker at 0.2 s VAD snapshots is still ~1:1. A first progressive that includes a long prefix can **burst** many frames; that is the case most likely to trip Tencent’s “faster than realtime” rule.
- Default concurrency is 20 connections per Tencent account.

### Recognition results (official fields → pipeline → Realtime)

Official envelope:

```json
{
  "code": 0,
  "message": "success",
  "voice_id": "…",
  "message_id": "…",
  "result": {
    "slice_type": 1,
    "index": 0,
    "start_time": 0,
    "end_time": 1240,
    "voice_text_str": "你好",
    "word_size": 0,
    "word_list": []
  },
  "final": 0
}
```

Official `slice_type` ([same doc](https://www.tencentcloud.com/document/product/1118/53937)):

| `slice_type` | Meaning | This adapter |
|---|---|---|
| `0` | Sentence start | Treated as unstable if text is present → `partial` |
| `1` | In progress; `voice_text_str` **may still change** | `self.partial = text` |
| `2` | Sentence end; `voice_text_str` is **steady** | Append to `stable_parts`, clear `partial` |

`current_text()` = joined `stable_parts`, else `partial`. That string becomes `PartialTranscription.text`.

`final: 1` means the **stream** is done (after `{"type":"end"}`). That is not VAD `speech_stopped`. `finish()` waits for `final==1` or a quiet socket once stable text exists.

`code != 0` raises and closes. Progressive failure does not fall back mid-utterance; a failed **final** falls back to SentenceRecognition.

### Realtime events the client sees

**`input_audio_buffer.speech_started`** — Silero, not Tencent handshake.

```json
{
  "type": "input_audio_buffer.speech_started",
  "event_id": "event_…",
  "audio_start_ms": 1280,
  "item_id": "item_…"
}
```

Allocates `speculative_input_item_id`. All later ASR events for this utterance reuse it.

If MiniMax is `in_response` and `interrupt_response` is on, the same dispatch **first** emits `response.output_audio.done` + `response.done` (`cancelled`, `turn_detected`), then `speech_started`.

A **reopen** (same `turn_id`, new revision) reuses that `item_id` and resets input `content_index`. A barge-in after MiniMax `commit` gets a new `item_id`.

**`conversation.item.input_audio_transcription.delta`** — one per Tencent result that has text (≈ every 200 ms while speaking):

```json
{
  "type": "conversation.item.input_audio_transcription.delta",
  "item_id": "item_…",
  "content_index": 2,
  "delta": "你好世界"
}
```

`delta` is **not** an official increment and **not** hosted-OpenAI token delta. It is the current `voice_text_str` (or concatenated stable sentences). Official `slice_type=1` hypotheses are replaced by later `1`s and then locked by `2`. Concatenating Realtime `delta`s duplicates Chinese. **Replace the live caption with the latest `delta`.**

`content_index` increments per partial. Official `result.index` (sentence number in the Tencent stream) is **dropped**. So is `start_time` / `end_time` / `word_list`. The Realtime client cannot recover Tencent word timestamps (`word_info` is `0` on our URL).

**`input_audio_buffer.speech_stopped`** — VAD soft-end. Duration is stored for usage. Tencent may still be receiving PCM and has not seen `{"type":"end"}` yet.

**`conversation.item.input_audio_transcription.completed`** — after `{"type":"end"}` and `finish()`:

```json
{
  "type": "conversation.item.input_audio_transcription.completed",
  "item_id": "item_…",
  "content_index": 0,
  "transcript": "你好世界。",
  "usage": { "type": "duration", "seconds": 2.24 }
}
```

`content_index` is always `0` here. `usage.seconds` is VAD duration, not Tencent billable time. `voice_id` / `message_id` never appear.

This is the **ignition** for TTS: emit `completed`, append or replace the user Chat item, and if `transcript` is nonempty put `GenerateResponseRequest` on `text_prompt_queue`. No client `response.create` on a normal spoken turn.

Empty `transcript`: `completed` still goes out; no generate.

A reopen final emits **another** `completed` on the **same** `item_id`.

Tencent `4008` (upload timeout) / `4006` (concurrency) become adapter exceptions, not Realtime events, unless the whole turn fails later as `error` / `response.done` `failed`.

---

## Streaming TTS: MiniMax audio → OpenAI Realtime

### What the official WebSocket API is

[MiniMax T2A WebSocket](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket) is `wss://api.minimax.io/ws/v1/t2a_v2` (China host analogue exists). Session:

```text
connect
  ← event=connected_success   base_resp.status_code=0
  → event=task_start          model, voice_setting, audio_setting
  ← event=task_started
  → event=task_continue       text  (repeatable; 10k chars; idle 120s closes)
  ← event=task_continued      data.audio = hex, is_final?
  → event=task_finish
  ← event=task_finished
```

`task_continued.data.audio` is hex in the requested format (mp3 / pcm / flac). `is_final: true` ends that continue. `task_failed` + nonzero `base_resp` means close the socket. Multiple `task_continue` can ride **one** WebSocket (LLM sentences as continues).

### What this repo actually calls

`MiniMaxTTSHandler` uses **HTTP SSE**, not that WebSocket:

- `POST` `MINIMAX_TTS_ENDPOINT` (`/v1/t2a_v2`)
- `Accept: text/event-stream`, `stream: true`
- `output_format: hex`, `audio_setting.format: pcm`, **16 kHz**, mono
- One HTTP stream per `TTSInput` (one LLM sentence with this profile)

SSE body fields we honor:

| HTTP SSE (this repo) | T2A WebSocket analogue | What we do |
|---|---|---|
| `data.audio` hex | `data.audio` hex | Decode to int16 PCM |
| `data.status == 1` | incremental `task_continued` (`is_final=false`) | Play immediately |
| `data.status == 2` | final aggregated audio / last continue | Skip if incrementals already played |
| `base_resp.status_code` | same | Fail if not `0` |
| (no `event`) | `task_continued` / `is_final` | Stream end = HTTP body end |
| new POST per sentence | `task_continue` on one socket | No persistent MiniMax session |

`MINIMAX_TTS_STREAM=false` is the old one-shot hex WAV. The WebSocket `task_*` machine is unused.

So: official WS docs explain **payload shape** (hex audio, `base_resp`, session vs trace ids). They do **not** describe our connection lifecycle. `session_id` / `trace_id` from MiniMax never become OpenAI `session_id` / `response_id`.

### Realtime events the client sees

Two queues, one `response_id`:

**`response.created`**

- Explicit `response.create` → immediate `in_progress`.
- Spoken Tencent turn: **deferred** until the first MiniMax PCM. `begin_audio_response` emits it on that chunk. WebRTC sends only this bookkeeping on `oai-events`; PCM is RTP.

Between `transcription.completed` and this event, `response_pending` is true. Barge-in still cancels.

**`response.output_audio_transcript.done`**

From DeepSeek’s sentence flush (`stream_batch_sentences: 1`), **not** from MiniMax. MiniMax has no transcript in `data.audio`.

```json
{
  "type": "response.output_audio_transcript.done",
  "response_id": "resp_…",
  "item_id": "item_…",
  "output_index": 0,
  "content_index": 0,
  "transcript": "你好，我在。"
}
```

There is no `response.output_audio_transcript.delta`. Each `.done` is one LLM sentence, not the cumulative answer. Official MiniMax `subtitle_enable` / `word_streaming` are not turned on, so there are no MiniMax word timestamps on the Realtime socket either.

The first of these also `commit`s the speculative revision. After that, Tencent reopen is refused; further speech is a new `item_id` + interrupt.

**`response.output_audio.delta`** (WebSocket transport only)

```json
{
  "type": "response.output_audio.delta",
  "response_id": "resp_…",
  "item_id": "item_…",
  "output_index": 0,
  "content_index": 3,
  "delta": "<base64 PCM16>"
}
```

Path: MiniMax hex → int16 16 kHz → 512-sample blocks (first frame may be shorter) → send loop batches up to 6400 bytes → base64. Resampled if `session.audio.output.format.rate` ≠ 16 kHz.

`content_index` here is output-batch index. Unrelated to Tencent / ASR `content_index`.

WebRTC: no audio deltas. Same `response.created`, PCM on RTP. `output_audio_buffer.clear` maps to MiniMax leftover that never left the server jitter buffer.

**`response.output_audio.done` + `response.done`**

When the handler yields `AUDIO_RESPONSE_DONE` (LLM `EndOfResponse`, not MiniMax `is_final` — we already drained the HTTP stream). Cancel reasons: `turn_detected`, `client_cancelled`. A stale MiniMax sentinel after barge-in does **not** emit a second `response.done`.

MiniMax `base_resp` 1002 / 1039 / 1004 become a failed generate (`error` + `response.done` `failed`) if they abort the HTTP stream. They are not forwarded as MiniMax event names.

---

## One spoken turn (official inner events vs Realtime)

```text
Realtime client                    Tencent wss                    MiniMax HTTP SSE
─────────────────                  ──────────                    ────────────────
append (PCM)…
                                   handshake ACK code=0
speech_started  item_u1
                                   binary PCM 200ms frames
transcription.delta  "你"          slice_type=1 voice_text_str
transcription.delta  "你好"        slice_type=1
speech_stopped                     (VAD only; socket still open)
                                   {"type":"end"}
transcription.completed            final=1, stable slice_type=2
  → GenerateResponseRequest
                                   close voice_id

                                   DeepSeek tokens…

output_audio_transcript.done       —                             —
                                   —                             POST /t2a_v2 stream
response.created                                                 status=1 hex pcm
output_audio.delta                                               status=1 …
output_audio_transcript.done       —                             (next sentence = new POST)
output_audio.delta                                               …
output_audio.done                                                body ends
response.done completed
```

`response.created` can follow the first `output_audio_transcript.done` if DeepSeek hits the text queue before MiniMax’s first SSE frame. Clients must accept either order.

---

## How the two official streams preempt each other (still via Realtime)

The send loop drains `text_output_queue` before `output_queue`. That is the only interleaving rule.

**Tencent speech cancels MiniMax**

1. `speech_started` is dequeued first.
2. Realtime: `output_audio.done` + `response.done(cancelled, turn_detected)` + `speech_started`.
3. `CancelScope.generation++`. MiniMax’s next SSE read / 512-sample emit sees `is_stale` and **returns** — the HTTP stream is abandoned (no MiniMax `task_finish`).
4. Queued PCM flushed; `AUDIO_RESPONSE_DONE` and `SESSION_END` kept. ASR `speech_stopped` / deltas / `completed` kept.

**Soft-end reopen (same Realtime `item_id`, new Tencent `voice_id`)**

Official Tencent: new handshake, new `voice_id`. We close the old socket and open another. Realtime: same user `item_id`, new deltas + a second `completed`. MiniMax for revision 1 is dropped if it has not `commit`ted; if it has, this is barge-in instead.

**MiniMax does not write `transcription.*`.** Official WS `session_id` / `trace_id` never leak onto `/v1/realtime`.

---

## IDs (do not conflate)

| ID | Whose | On OpenAI Realtime? |
|---|---|---|
| Tencent `voice_id` | One ASR WebSocket | No |
| Tencent `message_id` | One ASR JSON | No |
| MiniMax `session_id` / `trace_id` | T2A WS (unused here) / HTTP extra_info | No |
| Realtime `item_id` (user) | One spoken utterance; reused on reopen | Yes — `speech_*`, `transcription.*` |
| Realtime `response_id` | One generate | Yes — all `response.*` |
| `content_index` (input) | Per ASR **partial** | Yes — `transcription.delta` only |
| `content_index` (output) | Per audio **batch** | Yes — `output_audio.delta` |
| Pipeline `turn_id` / `turn_revision` / `cancel_generation` | Internal | No |

---

## What a correct Realtime client should do

1. On `transcription.delta`: **replace** the caption. Official Tencent `slice_type=1` is unstable; we already collapsed that into a full-string `delta`.
2. On `transcription.completed`: freeze it. A second `completed` on the same `item_id` is a speculative revision.
3. After you stop speaking, do not wait for `response.create`. Wait for `response.created` or the first audio/transcript event.
4. On `output_audio_transcript.done`: **append** that sentence (one DeepSeek flush, not MiniMax subtitles).
5. On `output_audio.delta`: decode as PCM16 at the negotiated rate; play immediately (first frame may be short). This is **not** MiniMax hex and **not** mp3 (even though the [T2A WebSocket examples](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket) often use `format: mp3`).
6. On `speech_started` plus `response.done` `turn_detected`: stop playback. WebRTC: server also clears RTP; you may send `output_audio_buffer.clear`.
7. Tool `conversation.item.create` does not restart MiniMax. Send `response.create` to speak the result.

The Realtime surface is unchanged: Tencent is `speech_*` + `input_audio_transcription.{delta,completed}`; MiniMax is `response.created` + `output_audio.delta` + `output_audio_transcript.done` + `{output_audio.done, response.done}`. The official docs only specify the **hidden** sockets — Tencent’s `slice_type` / `final` / `{"type":"end"}`, and MiniMax’s hex frames — which this service translates and must not leak.

## User

save the current canvas and artifacts to a local file
