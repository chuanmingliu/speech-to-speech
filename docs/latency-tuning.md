# Reducing end-to-end latency

The number a caller feels is **speech stop → first audio**. On the
`configs/tencent-deepseek-minimax.json` profile the 100×10 synthetic benchmark
(`artifacts/latency-benchmark-100x10-partial/`) breaks that down as:

| stage | p50 | p95 | p99 |
| --- | --- | --- | --- |
| `speech_stop_to_asr_final_ms` | 119 ms | 1726 ms | 2650 ms |
| `asr_final_to_first_assistant_text_ms` | **878 ms** | **4663 ms** | **11762 ms** |
| `first_assistant_text_to_first_audio_ms` | 205 ms | 345 ms | 522 ms |
| `speech_stop_to_first_audio_ms` | 1262 ms | 4874 ms | 13819 ms |

The LLM leg dominates both the median and the tail, so that is where the two
knobs below act. Both live on the OpenAI-compatible LLM handler and are set per
profile.

## `stream_first_chunk_lookahead_chars` (default 8)

The handler used to hold the reply until a sentence terminator (`.`/`。`) before
handing anything to TTS. For a first sentence like *"Sure, I can help with that,
let me check the weather for you now."* that means the listener hears nothing
while the model writes all seventeen tokens, even though the engine could have
started speaking at the first comma.

The first chunk now flushes at the opening **clause** boundary (`,` `;` `:` and
their CJK forms). The split only fires once this many characters are buffered
*past* the break, which is what makes it safe: the follow-up text is already in
hand, so the early flush cannot open a gap in playback. Only the first chunk
works this way — later chunks stay sentence-aligned, so prosody is unaffected
after the opening clause. Punctuation inside numbers (`1,000`, `3.14`, `12:30`)
and versioned names (`gpt-5.4-mini`) is not treated as a pause.

Measured at a 40 tok/s output rate with a 350 ms provider TTFT:

| reply | sentence-only | clause-early | saved |
| --- | --- | --- | --- |
| English, 17 tokens | 792 ms | 427 ms | **365 ms** |
| Chinese, 16 tokens | 730 ms | 502 ms | **227 ms** |

Set to `0` to restore the sentence-only behaviour.

## `request_hedge_after_ms` (default 0, off; 1200 in the Tencent profile)

`asr_final_to_first_assistant_text_ms` has a p99 of nearly twelve seconds — a
cold route, a slow node or an SDK retry occasionally stalls a completion for far
longer than the median. Waiting it out puts that whole stall in front of the
reply.

When set, a second identical completion is issued if the first has produced no
token within the window, and whichever answers first is streamed; the loser is
closed in the background. A failed first attempt is retried immediately rather
than waiting out the timer. Completions have no server-side side effects, so
the duplicate is safe.

This trades tokens for tail latency: the hedge only fires on turns already
slower than the window, so at 1200 ms it costs a duplicate request on the slow
minority of turns and leaves the median untouched. Raise the value to spend
less, lower it to cut more of the tail. Set to `0` to disable.

## Re-measuring

Both knobs are visible in the same benchmark that produced the table above:

```bash
uv run python scripts/synthetic_latency_benchmark.py run --limit-cases 20
```

Watch `speech_stop_to_first_audio_ms` p50 for the first-chunk change and p95/p99
for hedging.
