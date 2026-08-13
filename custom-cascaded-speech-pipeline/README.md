# Custom cascade latency work

Local report for the Tencent ASR + DeepSeek + MiniMax speech-to-speech path
in this worktree. Written when the latency task was functionally complete.

- Date: 2026-08-13
- Worktree: `feat-0813`
- Base commit: `e0ab857` — Add custom cascaded speech providers and test app
- Status: implemented, reviewed, demo launched; **not committed**

## Goal

Cut end-to-end spoken-turn latency on:

```text
Browser microphone
  → VAD
  → Tencent ASR
  → DeepSeek (chat-completions)
  → MiniMax TTS
  → browser playback
```

## Starting point (what was slow)

The checked-in cascade waited serially at every stage:

| Stage | Before | Why it hurt |
|---|---|---|
| VAD | `speech_pad_ms = 500` | Extra audio on every ASR payload |
| ASR | One-shot `SentenceRecognition` **after** VAD finalized | Full HTTP RTT after the user stopped (often 200–500 ms+) |
| LLM | `nltk.sent_tokenize` + batch of **3** sentences | Chinese `。！？` never split; `remove_unspeechable` **deleted** those marks, so TTS waited for the whole reply |
| TTS | MiniMax `stream: false`, full hex **WAV** | Time-to-first-audio = entire sentence synthesis |
| Playback | Wait for 512 PCM samples | First SSE frame smaller than 512 sat in a buffer |

Architecture docs already called MiniMax “sync T2A, streaming later” and Tencent “finals only.”

## What shipped

Work landed in three passes, then a review pass.

### 1. First-audio path (MiniMax + Chinese TTS flush)

**MiniMax streaming TTS** (`src/speech_to_speech/TTS/minimax_tts_handler.py`)

- Default is HTTP SSE (`stream=true`), hex **PCM** at 16 kHz, not a full WAV.
- Incremental `status=1` frames play immediately; aggregated `status=2` is skipped if incrementals already played.
- `MINIMAX_TTS_STREAM=false` keeps the old one-shot WAV path.
- Cancellation is checked while reading the stream, not only after the HTTP body.
- TLS warmup on setup (`HEAD` to the MiniMax origin).
- First frame is yielded even if it is shorter than 512 samples.

**LLM sentence flush** (`src/speech_to_speech/LLM/utils.py` and both LLM handlers)

- Keep CJK punctuation in `remove_unspeechable`.
- `split_spoken_units()` splits on `。！？；…` and still uses NLTK for Latin.
- First complete sentence is sent to TTS immediately; later sentences can still batch.

**Profile** (`configs/tencent-deepseek-minimax.json`)

- `stream_batch_sentences: 1`
- `speech_pad_ms: 80` (was 500)

### 2. Hide Tencent’s HTTP wait behind VAD silence

**VAD** (`src/speech_to_speech/VAD/vad_handler.py`)

- One **silence-prefetch** progressive snapshot when trailing silence starts (`temp_end`), so ASR can start during `min_silence_ms`.

**SentenceRecognition fallback** (used if App ID is missing or WebSocket fails)

- At most one background request per turn.
- Reused on final if the extra tail is ≤ 350 ms.
- HTTP keep-alive + 10 s timeout when the SDK supports it.

### 3. True streaming ASR (after `TENCENT_ASR_APP_ID` was provided)

New module: `src/speech_to_speech/STT/tencent_realtime.py`

- Signed `wss://asr.cloud.tencent.com/asr/v2/<appid>` (HMAC-SHA1, same layout as Tencent’s speech SDK).
- PCM frames (`voice_format=1`), **pipeline** VAD (`needvad=0`).
- Progressive VAD audio is sent incrementally; `{"type":"end"}` on VAD final.
- Partial transcripts go to the UI; one final `Transcription` goes to DeepSeek.

**Auto-enable** (`src/speech_to_speech/s2s_pipeline.py`)

- If `TENCENT_ASR_APP_ID` is set: live transcription on, snapshot interval **0.2 s**.

The App ID is stored only in ignored `.env.local`, not in source.

### 4. Review fixes

- A failed progressive WebSocket update no longer leaves a dead session for the next chunk.
- `finish()` no longer waits the full timeout once a stable sentence is already in hand.

## Current spoken-turn path

```text
Browser mic (16 kHz PCM)
  → VAD (80 ms pad; progressive every 200 ms when App ID is set)
  → Tencent realtime WebSocket (overlaps speech)
      fallback: SentenceRecognition + silence prefetch
  → DeepSeek chat-completions (stream=true)
      first sentence flushed on 。 / . / ！ / ?
  → MiniMax T2A SSE (hex PCM)
      first audio frame plays immediately
  → Demo playback worklet
```

Useful log lines:

- `Tencent ASR realtime WebSocket enabled`
- `Tencent realtime ASR session started`
- `Tencent realtime ASR finalized in …`
- `LLM first token in …`
- `MiniMax TTS first audio in …`
- `Last speech detected to first speech out: …`

## Expected latency change

Not a live stopwatch report (no timed spoken-turn benchmark was recorded). Qualitatively:

| Wait | Before | After |
|---|---|---|
| After user stops → ASR done | Full SentenceRecognition RTT | Mostly already done (streamed while speaking) |
| ASR done → first TTS request | Whole Chinese reply, or 3 English sentences | First sentence |
| TTS request → first audio | Full WAV | First PCM SSE frame |
| Extra audio on every ASR call | 500 ms pad | 80 ms pad |

Remaining first-audio wait is mostly **DeepSeek time-to-first-token**, plus MiniMax’s first frame.

## Tests and review

| Check | Result |
|---|---|
| Full pytest (after streaming ASR) | **636 passed** |
| After review fixes (targeted) | **63 passed** (custom handlers / launcher / LLM utils / responses) |
| Ruff | Clean on touched files |
| Code review | 2 bugs found and fixed (dead WS session; `finish()` hang) |
| Live smoke of one full spoken turn | **Not** formally scored; demo + providers did start |

Checked and left as-is:

- Partial transcripts send the full current hypothesis. The demo treats that as a replacement, not an append.
- CJK `。！？` are kept and flush TTS on the first sentence.
- MiniMax plays the first PCM frame without waiting for 512 samples.
- SentenceRecognition remains the fallback if the WebSocket path fails.

## How to run the demo

Credentials stay in ignored `.env.local`. Required names (do not commit values):

```text
DEEPSEEK_API_KEY
TENCENT_ASR_SECRET_ID
TENCENT_ASR_SECRET_KEY
TENCENT_ASR_APP_ID
MINIMAX_TTS_API_KEY
MINIMAX_TTS_VOICE_ID
```

Launch:

```bash
uv run --extra tencent-asr python scripts/run_custom_services_test_app.py
```

Then open:

- Demo: <http://127.0.0.1:7860>
- Realtime backend (default): `ws://127.0.0.1:8765/v1/realtime`

If port `8765` is busy, the launcher picks a free port and injects it into the demo. On 2026-08-13 the live session used `ws://127.0.0.1:51871/v1/realtime` because `8765` was already taken.

Click the center orb, allow the microphone, and speak. Press `Ctrl-C` in the launcher terminal to stop both processes.

## Files changed (uncommitted at report time)

About 1,300 lines across 16 tracked files plus one new file:

- **New:** `src/speech_to_speech/STT/tencent_realtime.py`
- **Core:** Tencent handler, MiniMax handler, VAD, both LLM handlers, `LLM/utils.py`, `s2s_pipeline.py`
- **Config/docs:** profile JSON, root `README.md`, architecture/operations, `.env.custom.example`
- **Tests:** custom handlers, LLM utils, responses API

`.env.local` is gitignored. This report file is local documentation for the worktree.

## What is not done

1. **No commit / PR / branch** yet.
2. **No timed end-to-end spoken-turn benchmark** (seconds from speech-stop to first audio).
3. **DeepSeek first-token** is not optimized further (already streaming `deepseek-v4-flash`).
4. **MiniMax send-faster-than-realtime** is only a risk if a long utterance is dumped in one shot; live 200 ms snapshots keep the rate near 1:1.
5. **Headless browser microphone QA** was not run (no mic in the agent environment).
6. If an App ID or key was pasted into chat, rotate the Tencent API key if the thread is shared.

## Bottom line

The latency task for this cascade is **functionally complete**: streaming ASR, streaming TTS, first-sentence flush, and the serial waits unique to this stack are removed. Remaining work is optional: commit/PR, a timed spoken-turn measurement, and any DeepSeek-side first-token tuning.
