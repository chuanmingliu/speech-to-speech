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

## Review Fixes

Independent review requested two P1 and three P2 fixes. All were addressed in a cohesive follow-up:

- Serialized provider stream closure behind one idempotent owner. A delayed-close fake proves cancellation and final cleanup never call `close()` concurrently or more than once.
- Treats iteration errors caused by stale/cancelled stream closure as normal cancellation, producing `EndOfResponse(error=None)` and no stale text.
- Removed transcript/generated-content logging from the transcription notifier, shared OpenAI-compatible handler, and LM output processor. Non-cancellation provider failures now log only the exception type.
- Strengthened privacy coverage to DEBUG through the real notifier and LM output processor, while actually exercising transcript, API key, signature, raw provider JSON, and decoded audio sentinels.
- Renamed shared LLM metrics to provider-neutral event names.
- Added `speakable_phrase_at_s` propagation so phrase dispatch is measured from the completed speakable phrase, not MiniMax client construction.
- Records the cancellation instant in `CancelScope`; provider closure metrics now measure from that barge-in instant.
- Added MiniMax provider-close timing and a signed last-accepted-audio offset from barge-in.
- Updated the existing Tencent regression to assert semantic fields plus finite, monotonic partial/final timestamps.

### Review Verification

```text
PYTHONPATH=src .venv/bin/pytest -q -p no:cacheprovider \
  tests/test_provider_streaming_pipeline.py \
  tests/test_chat_completions_backend.py \
  tests/test_responses_api_language_model.py \
  tests/test_lm_output_processor.py \
  tests/test_transcription_notifier.py \
  tests/test_tencent_realtime_asr.py \
  tests/test_minimax_streaming_tts.py \
  tests/openai_realtime/test_websocket_router.py

160 passed, 1 deprecation warning

PYTHONPATH=src .venv/bin/ruff check --no-cache \
  src/speech_to_speech/LLM \
  src/speech_to_speech/pipeline/messages.py \
  src/speech_to_speech/pipeline/cancel_scope.py \
  src/speech_to_speech/STT/tencent_asr_handler.py \
  src/speech_to_speech/STT/transcription_notifier.py \
  src/speech_to_speech/TTS/minimax_tts_handler.py \
  tests/test_provider_streaming_pipeline.py \
  tests/test_transcription_notifier.py \
  tests/test_tencent_realtime_asr.py

All checks passed!

git diff --check

passed
```

A repository-wide `pytest` collection attempt aborted in the native ML dependency stack while importing `src/speech_to_speech/LLM/language_model.py`; no tests ran in that attempt. The provider, realtime-router, Tencent, MiniMax, notifier, and shared LLM matrix above completed normally.
