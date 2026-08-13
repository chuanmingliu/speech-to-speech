# Task 3 Report: DeepSeek First-Sentence Flush, Cancellation, and Latency Metrics

## Status

DONE

## Changes

- Set the Tencent/DeepSeek/MiniMax profile to flush one complete sentence at a time while retaining streaming and disabled thinking.
- Added a real base-handler to `LMOutputProcessor` regression proving the first complete sentence reaches `TTSInput` before the provider terminal gate opens.
- Added prompt cancellation of blocked OpenAI-compatible streams through their supported `close()` boundary, with the existing `finally` close retained as an idempotent fallback.
- Added optional, backward-compatible monotonic timestamps to internal partial/final transcription, LLM chunk, and audio queue messages.
- Added injected monotonic clocks and content-free first-partial, final, first-delta, phrase-dispatch, first-audio, speech-end-to-audio, and barge-in-close latency logs.
- Added privacy assertions covering sentinel transcript, API key, signature, audio hex, and raw provider JSON values. All provider interactions are fakes; no credentials or live calls are used.

## TDD Evidence

- First sentence RED: profile parsed `stream_batch_sentences == 3`; GREEN after setting the profile value to `1`.
- Cancellation RED: a provider stream blocked past 200 ms after barge-in; GREEN after adding the close watcher, with no stale text emitted.
- Latency/privacy RED: Tencent setup rejected the injected clock and no stage timing fields/logs existed; GREEN after the minimal timing implementation.
- Barge-in timing RED: close occurred without a content-free duration event; GREEN after adding the numeric turn/revision-scoped close event.

## Verification

Because the shared `.venv` editable install points at the main checkout, worktree verification used `PYTHONPATH=src`.

```text
PYTHONPATH=src .venv/bin/pytest -q \
  tests/test_provider_streaming_pipeline.py \
  tests/test_chat_completions_backend.py \
  tests/test_responses_api_language_model.py \
  tests/test_lm_output_processor.py

60 passed

PYTHONPATH=src .venv/bin/ruff check --no-cache \
  src/speech_to_speech/LLM \
  src/speech_to_speech/pipeline/messages.py \
  src/speech_to_speech/STT/tencent_asr_handler.py \
  src/speech_to_speech/TTS/minimax_tts_handler.py \
  tests/test_provider_streaming_pipeline.py

All checks passed!

git diff --check

passed
```

## Concern

The broader pre-Task-3 Tencent suite has one whole-model equality assertion that expects `first_partial_at_s=None`; the handler now intentionally populates that new optional field. The Task 3 brief does not authorize changing `tests/test_tencent_realtime_asr.py`, so it was left untouched for the owning follow-up task.
