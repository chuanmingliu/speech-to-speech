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

### Tuning the lookahead once openers are primed

The lookahead is pure delay on the first chunk: it waits for N characters past
the clause boundary before flushing. Priming changes the trade, because a cached
opener is audible immediately and its playback then covers more of the wait for
the next chunk. On the Chinese corpus at 40 tok/s with the profile's primed
openers:

| lookahead | median first audio | new playback gaps |
| --- | --- | --- |
| 8 (code default) | 527 ms | none in Chinese |
| **3 (profile)** | **414 ms** | none in Chinese; `en-weather` 77 ms |
| 2 | 414 ms | as above |

The profile therefore ships `3`, worth **113 ms**. The code default stays at `8`
because this trade depends on two things being true, and neither is true
everywhere:

- **Output rate at or above ~40 tok/s.** At 30 tok/s the English gap grows to
  202 ms; at 20 tok/s lookahead 3 opens gaps in three Chinese cases too, the
  worst 220 ms. If you change LLM, re-run the benchmark before keeping `3`.
- **A primed opener.** English openers are not in the profile's prime list,
  which is why `en-weather` is the one case that degrades.

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

## `minimax_tts_prime_texts` (empty by default)

The clause-early flush sends a reply's opening clause to TTS as its own request,
so on a telephony call the first thing the caller hears is almost always one of a
handful of acknowledgements. Pre-synthesising those at startup puts them in the
exact-text cache, turning that request into a hit and removing MiniMax's
first-byte time (~205 ms) from the front of the turn.

Entries are pipe-separated and must match the flushed chunk exactly, punctuation
included — `好的，`, not `好的`. Each costs one billable synthesis at startup,
once per process, so the list is empty unless configured. A prime failure is
logged and skipped; it can cost the speed-up but never the turn.

Stacked with the clause flush on the Chinese cases (speech stop to first audio):

| case | feat_0824 | + clause flush | + priming |
| --- | --- | --- | --- |
| zh-weather | 905 ms | 730 ms | **527 ms** |
| zh-booking | 1030 ms | 730 ms | **527 ms** |
| zh-terse | 605 ms | 605 ms | **402 ms** |
| median | 867 ms | 730 ms | **527 ms** |

This only pays when the model actually opens with a primed clause. The
`MiniMax TTS cache hit` log line on a turn's first chunk is what tells you the
real hit rate on live traffic; nothing here measures that.

## MiniMax bidirectional protocol (`/ws/v1/t2a_v2_bidi`)

The TTS WebSocket now speaks the bidirectional protocol. A plain `https` endpoint
derives `wss://…/ws/v1/t2a_v2_bidi`; an explicitly pinned `ws(s)` endpoint is
left alone, so `/ws/v1/t2a_v2` still works. It is not a URL swap — three
behaviours differ and two of them bear directly on latency.

**The server assembles sentences itself.** Text sent with `task_continue` enters
a server-side buffer and is only synthesised once it forms a sentence. Text
ending in sentence-final punctuation (`。！？…!?.` or newline) goes immediately;
text ending in *secondary* punctuation (`，、；：,;:`) waits until enough has
accumulated, and short unpunctuated text waits for a backstop window.

That last rule would have quietly broken the clause-early flush: `好的，` is
exactly the short, secondary-punctuated chunk the server holds back, so the one
chunk this pipeline most wants back quickly is the one it would have delayed.
Every synthesis therefore ends with `task_flush`, which forces the buffer out
without closing the session. Synthesis now terminates on `task_flushed` rather
than an `is_final` audio frame, and `sentence_start` / `sentence_end` are
interleaved and ignored.

**Barge-in no longer costs a reconnect.** The old protocol had no interrupt: an
interruption closed the socket, so the next turn paid a fresh TCP/TLS connect
plus the `task_start` handshake. `task_cancel` discards buffered text and returns
the task to `task_started`, so the session stays warm. On a phone call, where
callers interrupt often, this removes a cold TTS connection from the turn that
follows every barge-in. The saving is not measured here — it needs a live
provider — but it is one connect plus one handshake round trip.

`continuous_sound` is already `false` in the task_start payload, which MiniMax
documents as the lower-latency segmentation mode.

### Not done: piping LLM tokens straight into `task_continue`

The endpoint is explicitly designed for it, and it would remove the text upload
from the critical path. It is not implemented because the win here is small: the
server will not synthesise a short unpunctuated fragment, so streaming `好`,
`的` early does not start synthesis any sooner — it starts when the `，` lands
either way. What is saved is one upload round trip, not the ~205 ms of synthesis
first-byte time. That does not justify reworking the handler contract from
discrete chunks to a token stream.

## Considered and not done: LLM prefetch on a stable ASR partial

Starting the completion from the last partial while the caller is still speaking
would overlap provider TTFT with the tail of the utterance. It is not
implemented, because `min_silence_ms` is already 64 ms with speculative reopen:
the last partial lands only ~180 ms before the final transcript, so that is the
entire head start available, and buying it costs a duplicate LLM request on
*every* turn against hedging's ~21% for a far larger tail win. It becomes
attractive if `min_silence_ms` is raised.

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
