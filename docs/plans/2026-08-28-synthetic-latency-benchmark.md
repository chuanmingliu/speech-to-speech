# Synthetic Latency Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an offline-generated corpus of 100 coherent conversations with 10 turns each and a reusable client that records detailed latency when the corpus is later run against a Realtime backend.

**Architecture:** A single Python module owns deterministic case generation, local microphone-audio synthesis/cache, one-WebSocket-per-case execution, event timestamp collection, and JSONL/CSV/summary output. Corpus generation and validation never contact Tencent, DeepSeek, or MiniMax; live execution is explicit.

**Tech Stack:** Python 3.10+, asyncio, websockets, numpy, soundfile, scipy, macOS `say`, pytest.

---

### Task 1: Define and generate the 100-case corpus

**Files:**
- Create: `scripts/synthetic_latency_benchmark.py`
- Create: `benchmarks/synthetic-conversations.json`
- Test: `tests/test_synthetic_latency_benchmark.py`

1. Define stable case/turn dataclasses and ten scenario-family builders.
2. Generate ten variants per family and exactly ten contextual turns per case.
3. Validate unique case IDs, unique turn IDs, nonempty prompts, 100 cases, and at least 1,000 turns.
4. Add `generate` and `validate` CLI commands.
5. Write deterministic JSON and verify regeneration produces no diff.

### Task 2: Record protocol and stage latency

**Files:**
- Modify: `scripts/synthetic_latency_benchmark.py`
- Test: `tests/test_synthetic_latency_benchmark.py`

1. Add a per-turn recorder for speech start/stop, first ASR partial, ASR final, response creation, first assistant transcript, first audio, audio done, response done, and errors.
2. Derive millisecond metrics without inventing missing timestamps.
3. Append all assistant transcript `.done` events because this server emits one per LLM sentence.
4. Count output audio chunks and bytes without saving sensitive protocol payloads.
5. Test event aliases, missing events, multi-sentence transcripts, and error completion.

### Task 3: Add local audio preparation and live runner

**Files:**
- Modify: `scripts/synthetic_latency_benchmark.py`
- Test: `tests/test_synthetic_latency_benchmark.py`

1. Synthesize each user prompt with macOS `say` outside the measured interval and cache WAVs by prompt hash.
2. Normalize cached audio to 16 kHz mono PCM16.
3. Open one Realtime WebSocket per case so all ten turns share conversation history.
4. Stream 20 ms microphone chunks in real time while a receiver task timestamps events immediately.
5. Support explicit case limits, turn limits, concurrency, timeout, trailing silence, optional bearer token, and output paths.
6. Keep live execution opt-in; `generate`, `validate`, and `prepare` do not call hosted providers.

### Task 4: Produce detailed records and aggregate summaries

**Files:**
- Modify: `scripts/synthetic_latency_benchmark.py`
- Create: `benchmarks/README.md`
- Test: `tests/test_synthetic_latency_benchmark.py`

1. Write one lossless turn record per JSONL line.
2. Write a flat CSV suitable for spreadsheets.
3. Compute count, success rate, min, mean, p50, p90, p95, p99, and max for every latency metric.
4. Write `summary.json` and a concise `summary.md` with run configuration and failure counts.
5. Document commands for corpus-only generation, local audio preparation, pilot execution, and the full 100 × 10 run.

### Task 5: Verify without creating provider traffic

**Files:**
- Test: `tests/test_synthetic_latency_benchmark.py`

1. Run corpus generation and validation locally.
2. Run unit tests with fake protocol events and fake sockets only.
3. Run Ruff and `git diff --check`.
4. Confirm no live benchmark command was executed.
