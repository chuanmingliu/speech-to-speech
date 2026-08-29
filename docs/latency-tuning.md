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

`scripts/latency_ab_benchmark.py stream` replays a 14-reply corpus (Chinese and
English, with and without clause punctuation) through the real handler. Against
an actual `feat_0824` checkout the baseline column reproduces exactly, so the
A/B is faithful. At 40 tok/s with a 350 ms TTFT:

- first audio is earlier in **8 of 14** cases, **median 63 ms**, mean 102 ms,
  best 325 ms;
- the six unchanged cases have no clause punctuation before the terminator,
  which is the honest ceiling on this technique;
- it costs one extra TTS request per early flush.

The saving scales inversely with token rate — a slower model spends longer
writing the first sentence, so skipping that wait is worth more:

| tok/s | mean saving | cases with a new playback gap |
| --- | --- | --- |
| 20 | 204 ms | 2 |
| 25 | 163 ms | 1 |
| 30 | 136 ms | 1 |
| 40 | 102 ms | 1 (27 ms — inaudible) |
| 60 | 68 ms | 0 |

**The gap column is the real caveat.** A short English opening clause ("Sure,",
~300 ms of audio) can finish speaking before the rest of the sentence has been
generated and synthesised. At the profile's 40 tok/s the worst case is a 27 ms
gap, which no one can hear. Below ~30 tok/s it grows: at 20 tok/s that case buys
650 ms of earlier onset at the cost of a 352 ms stutter. If you move to a slower
model, re-run the benchmark before trusting the default.

Set to `0` to restore the sentence-only behaviour.

## `request_hedge_after_ms` (default 0, off; 2000 in the Tencent profile)

`asr_final_to_first_assistant_text_ms` has a p99 of nearly twelve seconds — a
cold route, a slow node or an SDK retry occasionally stalls a completion for far
longer than the median. Waiting it out puts that whole stall in front of the
reply.

When set, a second identical completion is issued if the first has produced no
token within the window, and whichever answers first is streamed; the loser is
closed in the background. A failed first attempt is retried immediately rather
than waiting out the timer. Completions have no server-side side effects, so
the duplicate is safe.

`scripts/latency_ab_benchmark.py hedge` fits a lognormal to the measured
percentiles and simulates the windows (200k samples each):

| hedge after | p50 | p95 | p99 | extra requests |
| --- | --- | --- | --- | --- |
| off | 884 ms | 4668 ms | 9281 ms | — |
| 800 ms | 872 ms | 2322 ms | 3684 ms | 54% |
| 1200 ms | 880 ms | 2531 ms | 3859 ms | 38% |
| **2000 ms** | 874 ms | 2996 ms | 4271 ms | **21%** |
| 3000 ms | 879 ms | 3642 ms | 4888 ms | 11% |

The median is untouched at every setting — hedging only ever acts on turns that
are already slow. The profile ships **2000 ms** because 1200 ms nearly doubles
the request count for about 400 ms more p99; drop to 1200 ms if tail latency
matters more than tokens, and `0` to disable.

Two caveats. The simulation assumes the two attempts fail independently; if a
slow turn is caused by something shared — provider-wide overload, a saturated
link — the retry is slow too and the real gain is smaller. And the fitted
lognormal reaches p99 9.4 s against a measured 11.8 s, so it slightly
understates the extreme tail.

## Re-measuring

Offline, with no API keys and no provider traffic. Two harnesses at different
depths — the first is a model, the second runs the real thing:

```bash
# in-process: drives _consume_streaming directly. Milliseconds to run.
python scripts/latency_ab_benchmark.py stream          # first-chunk A/B + gap check
python scripts/latency_ab_benchmark.py hedge           # tail vs duplicate-request cost

# real pipeline: LLM handler -> LMOutputProcessor -> TTS, in real threads over
# real HTTP/SSE from a local fake provider, with the speculative gate live.
python scripts/latency_pipeline_benchmark.py --repeat 2
```

The two agree closely, which is the point of having both: over the corpus the
real pipeline measured a **median saving of 62 ms and mean of 102 ms** against
the model's 63 ms and 102 ms, and reproduced the `en-weather` gap at 28 ms
against the model's 27 ms. Thread and queue hops plus the speculative gate cost
about 4–5 ms in total, and — the thing worth checking — the gate does **not**
block the earlier first chunk, so the saving survives the real chain.

To A/B against another branch, point either harness at its checkout via
`S2S_SRC` — the old handler absorbs the unknown kwarg, so both of its columns
come out identical, which is how the baseline is validated:

```bash
git archive feat_0824 src | tar -x -C /tmp/base
S2S_SRC=/tmp/base/src python scripts/latency_ab_benchmark.py stream
S2S_SRC=/tmp/base/src python scripts/latency_pipeline_benchmark.py --repeat 2
```

Both report a flat 0 ms on `feat_0824`, and its baseline column matches
`feat_0829`'s, so the A/B is measuring the change and not a strawman.

Against live providers, the full corpus still applies:

```bash
uv run python scripts/synthetic_latency_benchmark.py run --limit-cases 20
```

Watch `speech_stop_to_first_audio_ms` p50 for the first-chunk change and p95/p99
for hedging.

## Unrelated bug this surfaced

The corpus exposed a pre-existing sentence-splitting fault, present on
`feat_0824` too: `nltk.sent_tokenize` breaks inside decimals and versioned
names, so "The total is 1,299.50 dollars" is sent to TTS as "The total is
1,299." followed by "50 dollars", and "gpt-5.4-mini" as "gpt-5." then
"4-mini". The number guard added here protects the *clause* split but not
nltk's sentence split. Worth fixing separately.
