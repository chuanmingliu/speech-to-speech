#!/usr/bin/env python3
"""End-to-end pipeline latency benchmark against a local fake provider.

``latency_ab_benchmark.py`` drives ``_consume_streaming`` in-process. This one
runs the **real handler chain in real threads**:

    GenerateResponseRequest
      -> ChatCompletionsApiModelHandler   (real openai SDK, real HTTP + SSE)
      -> LMOutputProcessor                (real queue hop, real speculative gate)
      -> a stub TTS handler               (real queue hop, models synthesis time)
      -> first audio chunk, wall-clock timed

The provider is a local HTTP server that speaks the Chat Completions streaming
protocol with a configurable time-to-first-token and token rate, so no API key
and no egress are needed. What this adds over the in-process harness is
everything between the handlers: thread scheduling, queue hops, the openai
SDK's own parsing, and ``SpeculativeTurnTracker`` gating -- which can *block*,
and so could in principle eat the saving the early flush buys.

Timings here are real wall-clock, so they carry a millisecond or two of jitter;
the provider's own latency is simulated, but it is identical across branches,
so the A/B delta is unaffected.

    python scripts/latency_pipeline_benchmark.py --repeat 5
    S2S_SRC=/tmp/base/src python scripts/latency_pipeline_benchmark.py --repeat 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from typing import Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("S2S_SRC") or str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from latency_ab_benchmark import CORPUS, speech_duration_s, tokenize  # noqa: E402
from openai.types.realtime.realtime_session_create_request import (  # noqa: E402
    RealtimeSessionCreateRequest,
)

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig  # noqa: E402
from speech_to_speech.baseHandler import BaseHandler  # noqa: E402
from speech_to_speech.LLM.chat import Chat, make_user_message  # noqa: E402
from speech_to_speech.LLM.chat_completions_language_model import (  # noqa: E402
    ChatCompletionsApiModelHandler,
)
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor  # noqa: E402
from speech_to_speech.pipeline.messages import (  # noqa: E402
    PIPELINE_END,
    EndOfResponse,
    GenerateResponseRequest,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker  # noqa: E402

# ── Fake provider ────────────────────────────────────────────────────────────


class _Provider(BaseHTTPRequestHandler):
    """Minimal Chat Completions server that streams at a controlled rate."""

    deltas: Sequence[str] = ()
    ttft_s: float = 0.35
    tokens_per_s: float = 40.0

    def log_message(self, *args: object) -> None:  # silence the default logger
        pass

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = json.dumps({"object": "list", "data": []}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not payload.get("stream"):
            # The handler's warmup posts max_tokens=1 without streaming.
            body = json.dumps(
                {
                    "id": "warm",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        started = time.perf_counter()
        for index, delta in enumerate(self.deltas):
            due = started + self.ttft_s + index / self.tokens_per_s
            remaining = due - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            chunk = {
                "id": "bench",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def start_provider(ttft_s: float, tokens_per_s: float) -> tuple[ThreadingHTTPServer, int]:
    handler = type("_BoundProvider", (_Provider,), {"ttft_s": ttft_s, "tokens_per_s": tokens_per_s})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


# ── Stub TTS ─────────────────────────────────────────────────────────────────


class StubTTSHandler(BaseHandler):
    """Models synthesis without a provider, so the TTS thread hop is real."""

    def setup(self, ttfb_s: float = 0.205, tts_speed: float = 1.2, sink: Queue | None = None) -> None:
        self.ttfb_s = ttfb_s
        self.tts_speed = tts_speed
        self.sink = sink

    def process(self, tts_input: object) -> Iterator[object]:
        if isinstance(tts_input, EndOfResponse):
            # Sentinel so the driver knows the turn is fully drained rather than
            # guessing from queue emptiness, which truncated the timeline.
            if self.sink is not None:
                self.sink.put(None)
            return
        text = getattr(tts_input, "text", "")
        if not text:
            return
        time.sleep(self.ttfb_s)
        if self.sink is not None:
            self.sink.put((time.perf_counter(), text, speech_duration_s(text, self.tts_speed)))
        yield b""


# ── Pipeline run ─────────────────────────────────────────────────────────────


@dataclass
class Result:
    first_audio_s: float
    chunks: list[str]
    gaps_s: list[float]

    @property
    def total_gap_s(self) -> float:
        return sum(self.gaps_s)


def run_once(reply, lookahead: int, port: int, *, ttfb_s: float, tts_speed: float,
             use_speculative: bool) -> Result:
    _Provider.deltas = tokenize(reply.text)
    for klass in _Provider.__subclasses__():
        klass.deltas = _Provider.deltas

    stop_event = threading.Event()
    q_llm_in: Queue = Queue()
    q_llm_out: Queue = Queue()
    q_tts_in: Queue = Queue()
    q_audio: Queue = Queue()
    sink: Queue = Queue()
    speculative = SpeculativeTurnTracker() if use_speculative else None

    llm = ChatCompletionsApiModelHandler(
        stop_event, q_llm_in, q_llm_out,
        setup_kwargs=dict(
            model_name="bench", base_url=f"http://127.0.0.1:{port}/v1", api_key="offline",
            stream=True, init_chat_prompt="Be brief.", disable_thinking=True,
            compact_history=False, stream_batch_sentences=1,
            stream_first_chunk_lookahead_chars=lookahead,
            speculative_turns=speculative,
        ),
    )
    lm_out = LMOutputProcessor(
        stop_event, q_llm_out, q_tts_in,
        setup_kwargs={"text_output_queue": Queue(), "speculative_turns": speculative},
    )
    tts = StubTTSHandler(
        stop_event, q_tts_in, q_audio,
        setup_kwargs={"ttfb_s": ttfb_s, "tts_speed": tts_speed, "sink": sink},
    )
    threads = [threading.Thread(target=h.run, daemon=True) for h in (llm, lm_out, tts)]
    for thread in threads:
        thread.start()

    chat = Chat(10)
    chat.add_item(make_user_message("benchmark"))
    runtime_config = RuntimeConfig(
        chat=chat,
        session=RealtimeSessionCreateRequest(type="realtime", instructions="Be brief."),
    )
    if speculative is not None:
        speculative.observe("bench", 0)

    started = time.perf_counter()
    q_llm_in.put(
        GenerateResponseRequest(
            runtime_config=runtime_config, response=None, language_code=reply.lang,
            turn_id="bench", turn_revision=0,
        )
    )

    audible: list[tuple[float, str, float]] = []
    deadline = started + 30.0
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"pipeline did not finish the turn for {reply.case_id}")
        try:
            item = sink.get(timeout=remaining)
        except Empty:
            raise TimeoutError(f"pipeline stalled for {reply.case_id}") from None
        if item is None:  # EndOfResponse reached TTS: the turn is complete
            break
        audible.append(item)

    stop_event.set()
    for queue_ in (q_llm_in, q_llm_out, q_tts_in):
        queue_.put(PIPELINE_END)
    for thread in threads:
        thread.join(timeout=2)
    llm.cleanup()

    gaps: list[float] = []
    playhead = 0.0
    first_audio = 0.0
    for index, (ready, _text, duration) in enumerate(audible):
        ready_s = ready - started
        if index == 0:
            first_audio = ready_s
            playhead = ready_s
        else:
            if ready_s > playhead + 1e-9:
                gaps.append(ready_s - playhead)
            playhead = max(playhead, ready_s)
        playhead += duration
    return Result(first_audio, [text for _t, text, _d in audible], gaps)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lookahead", type=int, default=8)
    parser.add_argument("--ttft-ms", type=float, default=350.0)
    parser.add_argument("--tokens-per-s", type=float, default=40.0)
    parser.add_argument("--tts-ttfb-ms", type=float, default=205.0)
    parser.add_argument("--tts-speed", type=float, default=1.2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--no-speculative", action="store_true",
                        help="Disable the SpeculativeTurnTracker gate (it can block).")
    args = parser.parse_args(argv)

    server, port = start_provider(args.ttft_ms / 1000.0, args.tokens_per_s)
    corpus = [r for r in CORPUS if not args.cases or r.case_id in args.cases]
    kwargs = dict(ttfb_s=args.tts_ttfb_ms / 1000.0, tts_speed=args.tts_speed,
                  use_speculative=not args.no_speculative)

    print(
        f"Source under test: {Path(ChatCompletionsApiModelHandler.__module__ and __import__('speech_to_speech').__file__).resolve().parents[1]}\n"
        f"Real handler chain over real threads; fake provider on 127.0.0.1:{port}\n"
        f"TTFT {args.ttft_ms:.0f}ms, {args.tokens_per_s:.0f} tok/s, TTS TTFB {args.tts_ttfb_ms:.0f}ms, "
        f"speculative gate {'off' if args.no_speculative else 'on'}, {args.repeat} runs/case (min taken)\n"
    )
    header = f"{'case':<13}{'baseline':>10}{'optimized':>11}{'saved':>9}{'gaps':>7}{'chunks':>8}  first chunk"
    print(header)
    print("-" * len(header))

    saved: list[float] = []
    gap_cases: list[str] = []
    for reply in corpus:
        base = min((run_once(reply, 0, port, **kwargs) for _ in range(args.repeat)),
                   key=lambda r: r.first_audio_s)
        opt = min((run_once(reply, args.lookahead, port, **kwargs) for _ in range(args.repeat)),
                  key=lambda r: r.first_audio_s)
        delta = base.first_audio_s - opt.first_audio_s
        saved.append(delta)
        gap = "-"
        if opt.total_gap_s > base.total_gap_s + 1e-4:
            gap = f"{(opt.total_gap_s - base.total_gap_s) * 1000:.0f}"
            gap_cases.append(reply.case_id)
        first = opt.chunks[0] if opt.chunks else ""
        if len(first) > 22:
            first = first[:21] + "…"
        print(
            f"{reply.case_id:<13}{base.first_audio_s * 1000:>10.0f}{opt.first_audio_s * 1000:>11.0f}"
            f"{delta * 1000:>9.0f}{gap:>7}{len(opt.chunks):>8}  {first}"
        )
    print("-" * len(header))
    print(f"\nmedian saving {statistics.median(saved) * 1000:.0f}ms, mean {statistics.fmean(saved) * 1000:.0f}ms")
    if gap_cases:
        print(f"New playback gaps: {', '.join(gap_cases)}")
    else:
        print("No new playback gaps.")
    server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
