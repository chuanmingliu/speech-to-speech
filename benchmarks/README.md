# Synthetic conversation latency benchmark

`synthetic-conversations.json` contains 100 deterministic test cases across 10
scenario families. Every case is one coherent 10-turn conversation, for 1,000
total spoken turns. Keeping all turns in one WebSocket session tests conversation
history as well as ASR, LLM, TTS, and Realtime event latency.

The families cover travel planning, device support, restaurant booking, returns,
tutoring, project planning, smart-home troubleshooting, meeting scheduling,
budget coaching, and language practice.

## Offline-only commands

Regenerate the corpus:

```bash
uv run python scripts/synthetic_latency_benchmark.py generate
```

Validate its shape without contacting any provider:

```bash
uv run python scripts/synthetic_latency_benchmark.py validate
```

Prepare local 16 kHz mono microphone WAVs with macOS `say`. This can take a
while for all 1,000 turns, but creates no Tencent, DeepSeek, or MiniMax traffic:

```bash
uv run python scripts/synthetic_latency_benchmark.py prepare \
  --audio-dir /tmp/speech-to-speech-latency-audio
```

Use `--case-limit` and `--turn-limit` for a smaller preparation pass.

## Live execution

Live execution is deliberately a separate `run` command. Start the backend,
then begin with one case:

```bash
uv run python scripts/synthetic_latency_benchmark.py run \
  --case-limit 1 \
  --turn-limit 10 \
  --results-dir /tmp/s2s-latency-pilot
```

The full paid run is:

```bash
uv run python scripts/synthetic_latency_benchmark.py run \
  --concurrency 1 \
  --results-dir /tmp/s2s-latency-full
```

Set `--concurrency` no higher than the backend's `num_pipelines`; excess
sessions are rejected by the pool. For authenticated deployments, place the
token in `REALTIME_API_KEY`, or select another variable with
`--bearer-token-env`.

The full corpus creates at least 1,000 Tencent ASR, DeepSeek, and MiniMax turns.
It was generated and unit-tested here, but was not sent to hosted providers.

## Result files

The selected results directory receives:

- `turns.jsonl`: nested, lossless per-turn records.
- `turns.csv`: flattened timestamps and latency metrics for analysis.
- `summary.json`: run configuration, success/failure counts, and aggregate
  min/mean/p50/p90/p95/p99/max values.
- `summary.md`: compact human-readable percentile summary.

Each turn records protocol receive times for speech start/stop, first ASR
partial, ASR final, response creation, first assistant text, first audio, audio
done, response done, and errors. Derived metrics include:

- speech-stop to ASR final;
- ASR final to response creation and first assistant text;
- first assistant text to first audio;
- speech-stop to first audio and response completion;
- first audio to audio completion;
- input streaming, VAD detection, connection, and total-turn duration;
- audio chunk/byte counts, input transcript, all assistant sentence
  transcripts, response status, and error code.
