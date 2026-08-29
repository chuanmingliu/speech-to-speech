#!/usr/bin/env python3
"""Offline A/B benchmark for the LLM-leg latency work on ``feat_0829``.

Two subcommands, neither of which contacts a provider or needs an API key:

``stream``
    Replays realistic assistant replies through the *real*
    ``_consume_streaming`` path with the clause-early first flush on and off,
    and reports when the caller would actually hear audio. It also models
    playback, because the risk of flushing an opening clause early is not
    latency but a *gap*: if the second chunk is not synthesised by the time the
    first finishes speaking, the reply stutters. A win that stutters is not a
    win, so gaps are reported alongside the speed-up.

``hedge``
    Monte-Carlo over a fitted first-token latency distribution, showing what
    ``request_hedge_after_ms`` does to the tail and what it costs in duplicate
    requests.

Timing uses a virtual clock rather than sleeps: the handler is a synchronous
generator, so the arrival time of the delta being processed is exactly the
emission time of any chunk it produces. The run is therefore deterministic and
finishes in milliseconds.

    python scripts/latency_ab_benchmark.py stream
    python scripts/latency_ab_benchmark.py hedge
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import random
import re
import statistics
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
# S2S_SRC points the harness at a different checkout's ``src`` so the same
# corpus can be replayed against another branch -- e.g. exporting feat_0824
# and confirming its numbers match this branch's baseline column:
#     git archive feat_0824 src | tar -x -C /tmp/base
#     S2S_SRC=/tmp/base/src python scripts/latency_ab_benchmark.py stream
# The old handler absorbs the unknown lookahead kwarg via **_kwargs, so both
# columns come out identical there, which is the point.
sys.path.insert(0, os.environ.get("S2S_SRC") or str(REPO_ROOT / "src"))

from openai.types.realtime.realtime_session_create_request import (  # noqa: E402
    RealtimeSessionCreateRequest,
)

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod  # noqa: E402
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig  # noqa: E402
from speech_to_speech.LLM.chat import Chat, make_user_message  # noqa: E402
from speech_to_speech.LLM.chat_completions_language_model import (  # noqa: E402
    ChatCompletionsApiModelHandler,
)
from speech_to_speech.pipeline.messages import (  # noqa: E402
    GenerateResponseRequest,
    LLMResponseChunk,
)

# ── Provider model ───────────────────────────────────────────────────────────
# Defaults describe the checked-in Tencent/DeepSeek/MiniMax profile as measured
# in artifacts/latency-benchmark-100x10-partial/summary.md.

DEFAULT_TTFT_S = 0.35  # provider time-to-first-token
DEFAULT_TOKENS_PER_S = 40.0  # DeepSeek-class output rate
DEFAULT_TTS_TTFB_S = 0.205  # first_assistant_text_to_first_audio_ms p50
DEFAULT_TTS_SPEED = 1.2  # minimax_tts_speed in the profile

# Speaking rates at speed 1.0, in characters per second.
LATIN_CHARS_PER_S = 14.0
CJK_CHARS_PER_S = 4.8

_CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ]")
_PUNCT = re.compile(r"[,.!?;:，。！？；：、…]")


@dataclass(frozen=True)
class Reply:
    """One assistant reply to replay."""

    case_id: str
    lang: str
    text: str
    note: str = ""


CORPUS: tuple[Reply, ...] = (
    # ── Chinese: the profile's primary language ──
    Reply("zh-weather", "zh", "好的，我帮你查一下今天的天气。", "typical opening clause"),
    Reply("zh-booking", "zh", "没问题，我先确认一下时间，然后帮你预订。", "two clauses before the terminator"),
    Reply(
        "zh-long",
        "zh",
        "好的，我明白你的意思了。这个问题需要分两步处理，首先要确认订单状态，然后再联系客服。",
        "multi-sentence reply",
    ),
    Reply("zh-terse", "zh", "好的。", "too short to flush early; must fall back cleanly"),
    Reply("zh-noclause", "zh", "今天北京晴天最高气温二十六度。", "no clause punctuation at all"),
    Reply("zh-number", "zh", "一共是1,299元，含税。", "thousands separator must not split"),
    # ── English ──
    Reply("en-weather", "en", "Sure, I can help with that, let me check the weather for you now.", "typical"),
    Reply("en-noclause", "en", "Let me look that up for you right now.", "no clause punctuation"),
    Reply("en-terse", "en", "Done.", "single short sentence"),
    Reply(
        "en-long",
        "en",
        "Of course, I can walk you through it. First, open the settings panel, then choose the network tab, "
        "and finally restart the service.",
        "multi-sentence reply",
    ),
    Reply("en-decimal", "en", "The total is 1,299.50 dollars, including tax.", "decimal and separator"),
    Reply("en-version", "en", "You are running gpt-5.4-mini, which is the fast model.", "versioned name"),
    Reply("en-colon", "en", "Here is the plan: check the logs, restart, then verify.", "colon as the first break"),
    Reply("en-late", "en", "I checked every record in the archive and found the invoice, finally.", "very late break"),
)


def tokenize(text: str) -> list[str]:
    """Split a reply into provider-like streaming deltas.

    CJK streams roughly one character per token. Latin words carry their leading
    space. Punctuation is emitted as its *own* token, which is the conservative
    choice here: attaching it to the preceding word would make the clause
    boundary arrive one token sooner and flatter the optimisation.
    """
    deltas: list[str] = []
    for piece in re.findall(r"\s*[^\s]+", text):
        leading, body = re.match(r"(\s*)(.*)", piece, re.S).groups()
        pending = leading
        for char in body:
            if _CJK.match(char) or _PUNCT.match(char):
                if pending:
                    deltas.append(pending)
                    pending = ""
                deltas.append(char)
            else:
                pending += char
        if pending:
            deltas.append(pending)
    return [d for d in deltas if d]


def speech_duration_s(text: str, tts_speed: float) -> float:
    """Approximate spoken duration of a TTS chunk."""
    cjk = len(_CJK.findall(text))
    latin = max(0, len(text.strip()) - cjk)
    return (cjk / CJK_CHARS_PER_S + latin / LATIN_CHARS_PER_S) / max(0.1, tts_speed)


# ── Handler harness ──────────────────────────────────────────────────────────


class _VirtualStream:
    """Yields deltas while tracking the virtual time each one arrives."""

    def __init__(self, deltas: Sequence[str], ttft_s: float, tokens_per_s: float) -> None:
        self._deltas = list(deltas)
        self._ttft_s = ttft_s
        self._tokens_per_s = tokens_per_s
        self.now_s = 0.0

    def __iter__(self) -> Iterator[object]:
        for index, delta in enumerate(self._deltas):
            self.now_s = self._ttft_s + index / self._tokens_per_s
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=delta, tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )

    def close(self) -> None:
        pass


class _FakeClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: None))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    def with_options(self, **kwargs: object) -> "_FakeClient":
        return self


def build_handler(lookahead_chars: int) -> ChatCompletionsApiModelHandler:
    original = base_mod.OpenAI
    base_mod.OpenAI = _FakeClient
    try:
        return ChatCompletionsApiModelHandler(
            threading.Event(),
            queue.Queue(),
            queue.Queue(),
            setup_kwargs=dict(
                model_name="bench",
                base_url="http://offline/v1",
                api_key="offline",
                stream=True,
                init_chat_prompt="Be brief.",
                disable_thinking=True,
                compact_history=False,
                stream_batch_sentences=1,
                stream_first_chunk_lookahead_chars=lookahead_chars,
            ),
        )
    finally:
        base_mod.OpenAI = original


@dataclass
class Timeline:
    """What the caller would hear, chunk by chunk."""

    chunks: list[str]
    submitted_s: list[float]
    first_audio_s: float
    gaps_s: list[float]
    finished_s: float

    @property
    def total_gap_s(self) -> float:
        return sum(self.gaps_s)


def replay(reply: Reply, lookahead_chars: int, *, ttft_s: float, tokens_per_s: float,
           tts_ttfb_s: float, tts_speed: float) -> Timeline:
    """Run one reply through the real handler and model its playback."""
    handler = build_handler(lookahead_chars)
    stream = _VirtualStream(tokenize(reply.text), ttft_s, tokens_per_s)
    handler.client.chat.completions.create = lambda **k: stream

    chat = Chat(10)
    chat.add_item(make_user_message("benchmark"))
    runtime_config = RuntimeConfig(
        chat=chat,
        session=RealtimeSessionCreateRequest(type="realtime", instructions="Be brief."),
    )
    request = GenerateResponseRequest(
        runtime_config=runtime_config,
        response=None,
        language_code=reply.lang,
        turn_id="bench",
        turn_revision=0,
    )

    chunks: list[str] = []
    submitted: list[float] = []
    for out in handler.process(request):
        if isinstance(out, LLMResponseChunk) and out.text:
            chunks.append(out.text)
            submitted.append(stream.now_s)

    # Playback: a chunk is audible tts_ttfb_s after it is submitted, but cannot
    # start before the previous chunk has finished speaking.
    gaps: list[float] = []
    playhead = 0.0
    first_audio = 0.0
    for index, (text, submit_s) in enumerate(zip(chunks, submitted)):
        ready_s = submit_s + tts_ttfb_s
        if index == 0:
            first_audio = ready_s
            playhead = ready_s
        else:
            if ready_s > playhead + 1e-9:
                gaps.append(ready_s - playhead)
            playhead = max(playhead, ready_s)
        playhead += speech_duration_s(text, tts_speed)
    return Timeline(chunks, submitted, first_audio, gaps, playhead)


# ── stream command ───────────────────────────────────────────────────────────


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}"


def run_stream(args: argparse.Namespace) -> int:
    kwargs = dict(
        ttft_s=args.ttft_ms / 1000.0,
        tokens_per_s=args.tokens_per_s,
        tts_ttfb_s=args.tts_ttfb_ms / 1000.0,
        tts_speed=args.tts_speed,
    )
    print(
        f"Source under test: {Path(base_mod.__file__).resolve().parents[3]}\n"
        f"Provider model: TTFT {args.ttft_ms:.0f}ms, {args.tokens_per_s:.0f} tok/s | "
        f"TTS TTFB {args.tts_ttfb_ms:.0f}ms, speed {args.tts_speed}\n"
        f"Baseline = sentence-only flush (feat_0824). "
        f"Optimized = clause-early flush, lookahead {args.lookahead} chars.\n"
    )
    header = f"{'case':<13}{'baseline':>10}{'optimized':>11}{'saved':>9}{'gaps':>7}{'chunks':>8}  first chunk spoken"
    print(header)
    print("-" * len(header))

    saved: list[float] = []
    base_first: list[float] = []
    opt_first: list[float] = []
    gap_cases: list[str] = []
    total_extra_requests = 0

    for reply in CORPUS:
        base = replay(reply, 0, **kwargs)
        opt = replay(reply, args.lookahead, **kwargs)
        delta = base.first_audio_s - opt.first_audio_s
        saved.append(delta)
        base_first.append(base.first_audio_s)
        opt_first.append(opt.first_audio_s)
        total_extra_requests += len(opt.chunks) - len(base.chunks)
        gap_note = "-"
        if opt.gaps_s:
            gap_note = _fmt_ms(opt.total_gap_s)
            if not base.gaps_s:
                gap_cases.append(reply.case_id)
        first_chunk = opt.chunks[0] if opt.chunks else ""
        if len(first_chunk) > 26:
            first_chunk = first_chunk[:25] + "…"
        print(
            f"{reply.case_id:<13}{_fmt_ms(base.first_audio_s):>10}{_fmt_ms(opt.first_audio_s):>11}"
            f"{_fmt_ms(delta):>9}{gap_note:>7}{len(opt.chunks):>8}  {first_chunk}"
        )

    print("-" * len(header))
    print(
        f"{'median':<13}{_fmt_ms(statistics.median(base_first)):>10}"
        f"{_fmt_ms(statistics.median(opt_first)):>11}{_fmt_ms(statistics.median(saved)):>9}"
    )
    improved = sum(1 for value in saved if value > 0.001)
    print(
        f"\nFirst audio is earlier in {improved}/{len(CORPUS)} cases; "
        f"mean saving {_fmt_ms(statistics.fmean(saved))}ms, best {_fmt_ms(max(saved))}ms."
    )
    print(f"Extra TTS requests across the corpus: {total_extra_requests} (one per early flush).")
    if gap_cases:
        print(f"PLAYBACK GAPS introduced in: {', '.join(gap_cases)} — lower the token rate or raise the lookahead.")
    else:
        print("No playback gap is introduced in any case: every early flush is covered by the text behind it.")
    print(
        "\nAll timings are modelled, not measured against live providers. They isolate the\n"
        "handler change; real network jitter sits on top of both columns equally."
    )
    return 0


# ── hedge command ────────────────────────────────────────────────────────────


def lognormal_from_percentiles(p50_ms: float, p95_ms: float) -> tuple[float, float]:
    """Fit a lognormal to a measured median and 95th percentile."""
    mu = math.log(p50_ms)
    sigma = math.log(p95_ms / p50_ms) / 1.6448536269514722
    return mu, sigma


def run_hedge(args: argparse.Namespace) -> int:
    mu, sigma = lognormal_from_percentiles(args.p50_ms, args.p95_ms)
    rng = random.Random(args.seed)

    def draw() -> float:
        return math.exp(rng.gauss(mu, sigma))

    fitted = sorted(draw() for _ in range(args.samples))

    def pct(values: Sequence[float], q: float) -> float:
        return values[min(len(values) - 1, int(q * len(values)))]

    print(
        f"Fitted lognormal to the measured asr_final -> first assistant text leg\n"
        f"(p50 {args.p50_ms:.0f}ms, p95 {args.p95_ms:.0f}ms): "
        f"fit p99 {pct(fitted, 0.99):.0f}ms vs measured {args.p99_ms:.0f}ms.\n"
        f"{args.samples} samples per window.\n"
    )
    header = f"{'hedge after':>12}{'p50':>8}{'p90':>8}{'p95':>9}{'p99':>10}{'extra reqs':>12}"
    print(header)
    print("-" * len(header))

    for window_ms in args.windows:
        latencies: list[float] = []
        extra = 0
        for _ in range(args.samples):
            first = draw()
            if window_ms <= 0 or first <= window_ms:
                latencies.append(first)
                continue
            extra += 1
            latencies.append(min(first, window_ms + draw()))
        latencies.sort()
        label = "off" if window_ms <= 0 else f"{window_ms:.0f}ms"
        print(
            f"{label:>12}{pct(latencies, 0.50):>8.0f}{pct(latencies, 0.90):>8.0f}"
            f"{pct(latencies, 0.95):>9.0f}{pct(latencies, 0.99):>10.0f}"
            f"{extra / args.samples:>11.1%}"
        )

    print(
        "\nCaveat: this assumes the two attempts fail independently. If a slow turn is\n"
        "caused by something shared — provider-wide overload, a saturated link — the\n"
        "retry is slow too and the real gain is smaller than the table suggests."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    stream = sub.add_parser("stream", help="A/B the clause-early first flush over the reply corpus.")
    stream.add_argument("--lookahead", type=int, default=8, help="stream_first_chunk_lookahead_chars under test.")
    stream.add_argument("--ttft-ms", type=float, default=DEFAULT_TTFT_S * 1000)
    stream.add_argument("--tokens-per-s", type=float, default=DEFAULT_TOKENS_PER_S)
    stream.add_argument("--tts-ttfb-ms", type=float, default=DEFAULT_TTS_TTFB_S * 1000)
    stream.add_argument("--tts-speed", type=float, default=DEFAULT_TTS_SPEED)
    stream.set_defaults(func=run_stream)

    hedge = sub.add_parser("hedge", help="Monte-Carlo the effect of request hedging on the tail.")
    hedge.add_argument("--p50-ms", type=float, default=877.9)
    hedge.add_argument("--p95-ms", type=float, default=4663.5)
    hedge.add_argument("--p99-ms", type=float, default=11761.7)
    hedge.add_argument("--samples", type=int, default=200_000)
    hedge.add_argument("--seed", type=int, default=20260829)
    hedge.add_argument("--windows", type=float, nargs="+", default=[0, 800, 1200, 2000, 3000])
    hedge.set_defaults(func=run_hedge)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
