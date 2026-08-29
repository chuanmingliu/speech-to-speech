from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from queue import Empty, Queue
from threading import Lock, Thread
from time import perf_counter
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from openai import DefaultHttpxClient, OpenAI
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import (
    Chat,
    ChatItemError,
    SupportedItem,
    build_active_chat,
    make_system_message,
    make_user_message,
)
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn, build_compactor
from speech_to_speech.LLM.text_prompt import build_text_system_prompt
from speech_to_speech.LLM.utils import (
    remove_unspeechable,
    resolve_auto_language,
    split_first_spoken_unit,
    split_spoken_units,
)
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import is_out_of_band, response_wants_audio

logger = logging.getLogger(__name__)

# About 18–24 seconds of default SDK backoff before warmup fails.
WARMUP_MAX_RETRIES = 6
DEFAULT_CONNECTION_KEEPALIVE_S = 300.0
CONNECTION_PROBE_INTERVAL_S = 30.0
DEFAULT_PREWARM_WAIT_S = 0.5
# How much text must be buffered past the opening clause before it is sent
# to TTS on its own. Enough lookahead means the follow-up chunk is already
# being written, so the early flush cannot open a gap in playback.
DEFAULT_FIRST_CHUNK_LOOKAHEAD_CHARS = 8
# Hedging is opt-in: a hedge is a second billable completion, and it only
# pays for itself on backends whose first-token latency has a long tail.
DEFAULT_HEDGE_AFTER_MS = 0.0


# ── Normalised provider events ────────────────────────────────────────────────
# Each backend's stream/response is mapped to this small vocabulary so the shared
# speech-pipeline logic (sentence batching, cancellation, history, token usage)
# lives in one place. Subclasses differ only in how they produce these events.


class TextDelta(BaseModel):
    """Incremental assistant text. Always RAW (unfiltered); the base applies
    ``remove_unspeechable`` for the audio path."""

    text: str


class AssistantMessage(BaseModel):
    """A complete assistant turn to write back to history."""

    content: list[AssistantContent]


class ToolCall(BaseModel):
    """A complete function tool call (``call_id`` / ``id`` already regenerated)."""

    item: ResponseFunctionToolCall


class Usage(BaseModel):
    """Token accounting for the turn."""

    input_tokens: int
    output_tokens: int


ProviderEvent = TextDelta | AssistantMessage | ToolCall | Usage

# (attempt index, api response, event iterator, first event, error)
_HedgeResult = tuple[int, Any, Optional[Iterator[ProviderEvent]], Optional[ProviderEvent], Optional[BaseException]]


def _prepend_event(first: ProviderEvent | None, events: Iterator[ProviderEvent] | None) -> Iterator[ProviderEvent]:
    """Re-attach the event a racing attempt already pulled off the stream."""
    if first is not None:
        yield first
    if events is not None:
        yield from events


class _Turn(BaseModel):
    """Per-request context threaded through generation (immutable for the turn)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    language_code: Optional[str]
    gen: int | None
    runtime_config: Any
    response: Any
    turn_id: str | None
    turn_revision: int | None
    speech_stopped_at_s: float | None
    wants_audio: bool


class _GenState(BaseModel):
    """Mutable accumulators collected while consuming a turn's events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)
    pending: list[SupportedItem] = Field(default_factory=list)
    clean_text: str = ""  # filtered text, kept only for the debug log
    input_tokens: int = 0
    output_tokens: int = 0


class BaseOpenAICompatibleHandler(BaseHandler[LLMIn, LLMOut], ABC):
    """Shared lifecycle for OpenAI-compatible LLM backends (Responses & Chat
    Completions).

    Subclasses implement four hooks — :meth:`warmup`,
    :meth:`_build_compaction_generate_fn`, :meth:`_serialize`, :meth:`_request`,
    :meth:`_iter_events` and :meth:`_build_optional_kwargs` — and inherit the
    request/response orchestration: speculative-turn gating, cancellation,
    sentence batching, text-only vs audio handling, history write-back, token
    usage, out-of-band handling and error termination.
    """

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        model_name: str = "gpt-5.4-mini",
        device: str = "cuda",
        gen_kwargs: dict[str, Any] = {},
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: bool = True,
        user_role: str = "user",
        init_chat_prompt: Optional[str] = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        disable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        supports_images: Optional[bool] = None,
        request_timeout_s: float = 20.0,
        stream_batch_sentences: int = 3,
        stream_first_chunk_lookahead_chars: int = DEFAULT_FIRST_CHUNK_LOOKAHEAD_CHARS,
        request_hedge_after_ms: float = DEFAULT_HEDGE_AFTER_MS,
        enable_lang_prompt: bool = False,
        compact_history: bool = False,
        http_client: httpx.Client | None = None,
        **_kwargs: Any,
    ) -> None:
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.model_name = model_name
        self.stream = stream
        self.stream_batch_sentences = max(1, stream_batch_sentences)
        self.stream_first_chunk_lookahead_chars = max(0, int(stream_first_chunk_lookahead_chars))
        self.request_hedge_after_s = max(0.0, float(request_hedge_after_ms)) / 1000.0
        self.enable_lang_prompt = enable_lang_prompt
        self.gen_kwargs = dict(gen_kwargs)
        self.request_timeout_s = float(request_timeout_s)
        self.request_timeout = httpx.Timeout(
            self.request_timeout_s,
            connect=min(10.0, self.request_timeout_s),
        )
        self.warmup_system_prompt = (
            build_voice_system_prompt(init_chat_prompt)
            if init_chat_prompt
            else ""
        )

        self.user_role = user_role
        api_key, base_url, stream, disable_thinking, reasoning_effort = self._resolve_openai_compatible_connection(
            api_key=api_key,
            base_url=base_url,
            stream=stream,
            disable_thinking=disable_thinking,
            reasoning_effort=reasoning_effort,
            extra=_kwargs,
        )
        self.stream = stream
        self.supports_images = (
            not self._is_deepseek(base_url)
            if supports_images is None
            else bool(supports_images)
        )
        self.connection_keepalive_s = float(
            _kwargs.get("responses_api_connection_keepalive_s", DEFAULT_CONNECTION_KEEPALIVE_S)
        )
        if self.connection_keepalive_s <= 0:
            raise ValueError("responses_api_connection_keepalive_s must be greater than zero.")
        self.prewarm_wait_s = float(_kwargs.get("responses_api_prewarm_wait_s", DEFAULT_PREWARM_WAIT_S))
        if self.prewarm_wait_s < 0:
            raise ValueError("responses_api_prewarm_wait_s must be zero or greater.")
        self._owns_http_client = http_client is None
        self._http_client = http_client or DefaultHttpxClient(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=self.connection_keepalive_s,
            )
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=self._http_client)
        self._extra_body = self._build_extra_body(base_url, disable_thinking, reasoning_effort)
        self.compactor = build_compactor(self._build_compaction_generate_fn()) if compact_history else None
        self._connection_probe_lock = Lock()
        self._last_connection_use_s = 0.0
        self.warmup()
        self._last_connection_use_s = perf_counter()

    @staticmethod
    def _resolve_openai_compatible_connection(
        *,
        api_key: Optional[str],
        base_url: Optional[str],
        stream: bool,
        disable_thinking: bool,
        reasoning_effort: Optional[str],
        extra: dict[str, Any],
    ) -> tuple[str, Optional[str], bool, bool, Optional[str]]:
        """Accept CLI/profile ``responses_api_*`` aliases and hosted-provider env vars.

        ``vars(ResponsesApiLanguageModelHandlerArguments)`` uses prefixed field
        names. Without this mapping, ``speech-to-speech configs/*.json`` drops
        DeepSeek's base URL and requires ``OPENAI_API_KEY``.
        """
        api_key = api_key or extra.get("responses_api_api_key")
        base_url = base_url or extra.get("responses_api_base_url")
        if "responses_api_stream" in extra:
            stream = bool(extra["responses_api_stream"])
        if "responses_api_disable_thinking" in extra:
            disable_thinking = bool(extra["responses_api_disable_thinking"])
        reasoning_effort = reasoning_effort or extra.get("responses_api_reasoning_effort")
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not base_url:
            base_url = os.getenv("DEEPSEEK_API_BASE")
        if not api_key:
            raise ValueError(
                "OpenAI-compatible LLM requires an API key. Set DEEPSEEK_API_KEY "
                "for the Tencent/DeepSeek/MiniMax profile, or responses_api_api_key / OPENAI_API_KEY."
            )
        return api_key, base_url, stream, disable_thinking, reasoning_effort

    @staticmethod
    def _is_official_openai(base_url: Optional[str]) -> bool:
        """Whether ``base_url`` points at the official OpenAI server.

        Normalises a trailing slash so ``https://api.openai.com/v1/`` is also
        recognised; the official server rejects the provider-specific extra_body
        keys we send to vLLM / the HF router.
        """
        if base_url is None:
            return False
        return base_url.rstrip("/") == "https://api.openai.com/v1"

    @staticmethod
    def _is_deepseek(base_url: Optional[str]) -> bool:
        """Whether ``base_url`` targets DeepSeek's hosted API."""
        if not base_url:
            return False
        hostname = urlparse(base_url).hostname or ""
        return hostname == "api.deepseek.com" or hostname.endswith(".deepseek.com")

    @classmethod
    def _build_extra_body(
        cls,
        base_url: Optional[str],
        disable_thinking: bool,
        reasoning_effort: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """Build the provider-specific ``extra_body`` used to disable reasoning.

        Providers differ in how reasoning is turned off. DeepSeek Chat
        Completions requires ``thinking.type=disabled``; vLLM/Qwen honours
        ``chat_template_kwargs.enable_thinking=false``; and some other routers
        require ``reasoning_effort='none'``. None of these provider-specific
        fields apply to the official OpenAI server.
        """
        if base_url is None or cls._is_official_openai(base_url):
            return None
        if cls._is_deepseek(base_url):
            if disable_thinking or reasoning_effort == "none":
                return {"thinking": {"type": "disabled"}}
            if reasoning_effort:
                return {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": reasoning_effort,
                }
            return None
        if reasoning_effort:
            return {"reasoning_effort": reasoning_effort}
        if disable_thinking:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None

    # ── subclass hooks ──────────────────────────────────────────────────────--

    @abstractmethod
    def warmup(self) -> None:
        """Issue a cheap request so the model/connection is ready before serving."""
        ...

    @abstractmethod
    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        """Return a ``(system, user) -> text`` fn used to compact long histories."""
        ...

    def prewarm(self) -> None:
        """Refresh an idle provider connection without generating billable text."""
        now = perf_counter()
        if now - self._last_connection_use_s < CONNECTION_PROBE_INTERVAL_S:
            return
        if not self._connection_probe_lock.acquire(blocking=False):
            return
        try:
            now = perf_counter()
            if now - self._last_connection_use_s < CONNECTION_PROBE_INTERVAL_S:
                return
            started_at_s = perf_counter()
            self.client.with_options(
                max_retries=0,
                timeout=httpx.Timeout(3.0, connect=2.0),
            ).models.list()
            self._last_connection_use_s = perf_counter()
            logger.info("Hosted LLM HTTP connection refreshed in %.3fs", self._last_connection_use_s - started_at_s)
        except Exception as exc:
            logger.warning("Hosted LLM HTTP connection refresh failed; the next request will reconnect: %s", exc)
        finally:
            self._connection_probe_lock.release()

    def maintain_connection(self) -> None:
        """Keep an idle telephony lane's HTTP transport ready for admission."""
        self.prewarm()

    def _wait_for_connection_prewarm(self) -> None:
        if self.prewarm_wait_s <= 0 or not self._connection_probe_lock.locked():
            return
        started_at_s = perf_counter()
        acquired = self._connection_probe_lock.acquire(timeout=self.prewarm_wait_s)
        if acquired:
            self._connection_probe_lock.release()
        waited_s = perf_counter() - started_at_s
        logger.debug("Waited %.3fs for hosted LLM connection prewarm", waited_s)

    @abstractmethod
    def _serialize(self, active_chat: Chat) -> Any:
        """Serialise the chat to the backend's request payload (input/messages)."""
        ...

    @abstractmethod
    def _request(self, api_input: Any, optional_kwargs: dict[str, Any]) -> Any:
        """Issue the create() call and return the response or stream."""
        ...

    @abstractmethod
    def _iter_stream_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Map a streaming response to normalised :data:`ProviderEvent`s."""
        ...

    @abstractmethod
    def _iter_response_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Map a non-streaming response to normalised :data:`ProviderEvent`s."""
        ...

    def _iter_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Dispatch to the stream/non-stream mapper. ``self.stream`` is the single
        source of truth (it set the request's ``stream=`` flag), so the response
        type always matches it."""
        if self.stream:
            yield from self._iter_stream_events(api_response)
        else:
            yield from self._iter_response_events(api_response)

    @abstractmethod
    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        """Build the per-request tools/tool_choice kwargs in the backend's shape."""
        ...

    # ── speculative-turn / cancellation gating ─────────────────────────────────

    def _turn_is_latest(self, turn_id: str | None, turn_revision: int | None) -> bool:
        return self.speculative_turns is None or self.speculative_turns.is_latest(turn_id, turn_revision)

    def _generation_is_stale(self, gen: int | None) -> bool:
        return gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)

    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)

    def _apply_config(
        self,
        chat: Chat,
        instructions: Optional[str],
        wants_audio: bool = True,
    ) -> None:
        if instructions:
            builder = build_voice_system_prompt if wants_audio else build_text_system_prompt
            full_instructions = builder(instructions)
            chat.add_item(make_system_message(full_instructions))

    # ── output helpers ──────────────────────────────────────────────────────--

    def _chunk(
        self,
        turn: _Turn,
        *,
        text: str = "",
        tools: list[ResponseFunctionToolCall] | None = None,
        language_code: Optional[str] = None,
    ) -> LLMResponseChunk:
        return LLMResponseChunk(
            text=text,
            language_code=language_code if language_code is not None else turn.language_code,
            tools=tools or [],
            runtime_config=turn.runtime_config,
            response=turn.response,
            turn_id=turn.turn_id,
            turn_revision=turn.turn_revision,
            speech_stopped_at_s=turn.speech_stopped_at_s,
            cancel_generation=turn.gen,
        )

    def _record_tool_call(self, state: _GenState, turn: _Turn, item: ResponseFunctionToolCall) -> Iterator[LLMOut]:
        """Emit a tool call, persisting it (and any assistant text seen so far)
        to history *before* it is forwarded to the client.

        The function_call must already exist in the conversation by the time the
        client returns its ``function_call_output``; otherwise a fast client
        races ahead of the deferred end-of-turn write-back and the output is
        rejected ("No function_call with call_id ... found"), which makes the
        model re-issue the same tool call. The call lands in ``_pending_tool_calls``
        (not serialized until its output pairs it), so eager recording is safe.

        Out-of-band turns never touch the default conversation, and a stale turn
        records nothing (it is not forwarded to the client either)."""
        state.tools.append(item)
        fc_item = RealtimeConversationItemFunctionCall(
            type="function_call",
            name=item.name,
            arguments=item.arguments,
            call_id=item.call_id,
            id=item.id,
            status=item.status,
        )
        if self._generation_is_stale(turn.gen) or not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
            logger.info("LLM generation cancelled (stale speculative turn)")
            return
        if not is_out_of_band(turn.response):
            # Flush assistant text accumulated before this call first (so history
            # order matches what the client received), then persist the call —
            # all before the chunk leaves for the client.
            chat = turn.runtime_config.chat
            for pending_item in state.pending:
                chat.add_item(pending_item)
            state.pending.clear()
            chat.add_item(fc_item)
        yield self._chunk(turn, tools=[item])

    # ── hedged requests ───────────────────────────────────────────────────────

    @staticmethod
    def _close_api_response(api_response: Any) -> None:
        close = getattr(api_response, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            logger.debug("Closing a hedged LLM response failed", exc_info=True)

    def _reap_hedge_losers(self, results: "Queue[_HedgeResult]", outstanding: int) -> None:
        """Drain and close the attempts that lost the race, off the hot path.

        A losing attempt is still blocked inside ``next(events)`` waiting on the
        provider, so it cannot be closed synchronously without re-introducing
        the very latency the hedge removed.
        """
        if outstanding <= 0:
            return

        def reap() -> None:
            for _ in range(outstanding):
                try:
                    result = results.get(timeout=self.request_timeout_s + 5.0)
                except Empty:
                    return
                self._close_api_response(result[1])

        Thread(target=reap, name="llm-hedge-reaper", daemon=True).start()

    def _request_hedged(
        self,
        api_input: Any,
        optional_kwargs: dict[str, Any],
        turn: _Turn,
    ) -> tuple[Any, Iterator[ProviderEvent]]:
        """Race up to two identical completions and return the first to speak.

        Time-to-first-token on a hosted provider has a much heavier tail than
        median (a cold route, a slow node, an SDK retry). Waiting it out puts
        seconds of silence in front of the reply, so once ``request_hedge_after_s``
        has passed with no first event a second request is issued and whichever
        produces a token first is consumed. Completions have no server-side
        side effects, so the loser is simply closed.
        """
        results: Queue[_HedgeResult] = Queue()

        def attempt(index: int) -> None:
            api_response: Any = None
            try:
                api_response = self._request(api_input, optional_kwargs)
                events = self._iter_events(api_response)
                first = next(events, None)
                results.put((index, api_response, events, first, None))
            except BaseException as exc:  # noqa: BLE001 - relayed to the caller thread
                results.put((index, api_response, None, None, exc))

        def spawn(index: int) -> None:
            Thread(target=attempt, args=(index,), name=f"llm-hedge-{index}", daemon=True).start()

        started_at_s = perf_counter()
        deadline_s = started_at_s + self.request_timeout_s
        spawn(0)
        outstanding = 1
        hedged = False
        errors: list[BaseException] = []

        while outstanding > 0:
            timeout_s = self.request_hedge_after_s if not hedged else max(0.0, deadline_s - perf_counter())
            try:
                index, api_response, events, first, exc = results.get(timeout=timeout_s)
            except Empty:
                if hedged:
                    break
                hedged = True
                outstanding += 1
                logger.info(
                    "Hedging LLM request after %.0fms (turn=%s rev=%s)",
                    self.request_hedge_after_s * 1000,
                    turn.turn_id,
                    turn.turn_revision,
                )
                spawn(1)
                continue

            outstanding -= 1
            if exc is not None:
                errors.append(exc)
                self._close_api_response(api_response)
                if not hedged:
                    # The primary failed before the hedge timer elapsed; retry
                    # now rather than surfacing the error to the caller.
                    hedged = True
                    outstanding += 1
                    logger.info(
                        "Retrying failed LLM request (turn=%s rev=%s): %s",
                        turn.turn_id,
                        turn.turn_revision,
                        exc,
                    )
                    spawn(1)
                continue

            if hedged:
                logger.info(
                    "Hedged LLM request attempt %d won in %.3fs (turn=%s rev=%s)",
                    index,
                    perf_counter() - started_at_s,
                    turn.turn_id,
                    turn.turn_revision,
                )
            self._reap_hedge_losers(results, outstanding)
            return api_response, _prepend_event(first, events)

        if errors:
            raise errors[0]
        raise TimeoutError(
            f"No LLM response within {self.request_timeout_s:.1f}s (turn={turn.turn_id} rev={turn.turn_revision})."
        )

    def _open_stream(
        self,
        api_input: Any,
        optional_kwargs: dict[str, Any],
        turn: _Turn,
    ) -> tuple[Any, Iterator[ProviderEvent]]:
        if self.request_hedge_after_s > 0:
            return self._request_hedged(api_input, optional_kwargs, turn)
        api_response = self._request(api_input, optional_kwargs)
        return api_response, self._iter_events(api_response)

    # ── consumption ─────────────────────────────────────────────────────────--

    def _consume_streaming(
        self,
        events: Iterator[ProviderEvent],
        state: _GenState,
        turn: _Turn,
        request_started_at_s: float | None = None,
    ) -> Iterator[LLMOut]:
        cancelled = False
        printable_text = ""
        sentence_batch: list[str] = []
        first_flush_done = False
        early_flush_done = False
        first_token_logged = False

        def _flush(batch: list[str]) -> Iterator[LLMOut]:
            if not batch:
                return
            if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                logger.info("LLM generation cancelled (stale speculative turn)")
                return
            # Sentences keep the whitespace that separated them from the
            # previous flush; normalise it so a chunk never opens with a space.
            flushed = " ".join(piece for piece in (part.strip() for part in batch) if piece)
            if not flushed:
                return
            logger.info(
                "Streaming LLM sentence (turn=%s rev=%s): %s",
                turn.turn_id,
                turn.turn_revision,
                flushed if len(flushed) <= 80 else f"{flushed[:80]}…",
            )
            yield self._chunk(turn, text=flushed)

        def _batch_limit() -> int:
            return 1 if not first_flush_done else self.stream_batch_sentences

        for event in events:
            if self._generation_is_stale(turn.gen) or not self._turn_is_latest(turn.turn_id, turn.turn_revision):
                logger.info("LLM generation cancelled (interruption)")
                cancelled = True
                break

            if isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens
            elif isinstance(event, AssistantMessage):
                state.pending.append(
                    RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=event.content)
                )
            elif isinstance(event, ToolCall):
                # Flush any pending spoken text before emitting the tool call.
                if printable_text.strip():
                    sentence_batch.append(printable_text.strip())
                    printable_text = ""
                if sentence_batch:
                    if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                        logger.info("LLM generation cancelled (stale speculative turn)")
                        cancelled = True
                        break
                    yield from _flush(sentence_batch)
                    sentence_batch = []
                    first_flush_done = True
                yield from self._record_tool_call(state, turn, event.item)
            elif isinstance(event, TextDelta):
                if (
                    request_started_at_s is not None
                    and not first_token_logged
                    and event.text
                ):
                    first_token_logged = True
                    logger.info(
                        "LLM first token in %.3fs (turn=%s rev=%s)",
                        perf_counter() - request_started_at_s,
                        turn.turn_id,
                        turn.turn_revision,
                    )
                    if turn.speech_stopped_at_s is not None:
                        logger.info(
                            "Last speech detected to LLM first token: %.3fs (turn=%s rev=%s)",
                            perf_counter() - turn.speech_stopped_at_s,
                            turn.turn_id,
                            turn.turn_revision,
                        )
                if not turn.wants_audio:
                    # Text-only: forward verbatim. Keep every character (no
                    # remove_unspeechable, which strips TTS-unfriendly symbols) and
                    # don't sentence-split (sent_tokenize collapses newlines/markdown).
                    state.clean_text += event.text
                    if event.text:
                        if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                            logger.info("LLM generation cancelled (stale speculative turn)")
                            cancelled = True
                            break
                        yield self._chunk(turn, text=event.text)
                    continue
                new_text = remove_unspeechable(event.text)
                state.clean_text += new_text
                printable_text += new_text
                if (
                    not early_flush_done
                    and not first_flush_done
                    and not sentence_batch
                    and self.stream_first_chunk_lookahead_chars > 0
                ):
                    # Speak the opening clause without waiting for the sentence
                    # to terminate: the remaining tokens of that sentence are
                    # then generated while the listener is already hearing it.
                    head, printable_text = split_first_spoken_unit(
                        printable_text,
                        self.stream_first_chunk_lookahead_chars,
                    )
                    if head:
                        if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                            logger.info("LLM generation cancelled (stale speculative turn)")
                            cancelled = True
                            break
                        early_flush_done = True
                        yield from _flush([head.strip()])
                complete, printable_text = split_spoken_units(printable_text)
                for sentence in complete:
                    sentence_batch.append(sentence)
                    if len(sentence_batch) >= _batch_limit():
                        if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                            logger.info("LLM generation cancelled (stale speculative turn)")
                            cancelled = True
                            break
                        yield from _flush(sentence_batch)
                        sentence_batch = []
                        first_flush_done = True
                if cancelled:
                    break

        if not cancelled:
            if printable_text.strip():
                sentence_batch.append(printable_text.strip())
            if sentence_batch:
                if self._generation_is_stale(turn.gen):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    logger.debug(f"Clean text: {state.clean_text}")
                    yield from _flush(sentence_batch)
            logger.info(f"Tools: {state.tools}")
            logger.info(
                "Streaming LLM finished (turn=%s rev=%s in=%d out=%d)",
                turn.turn_id,
                turn.turn_revision,
                state.input_tokens,
                state.output_tokens,
            )

    def _consume_nonstreaming(self, events: Iterator[ProviderEvent], state: _GenState, turn: _Turn) -> Iterator[LLMOut]:
        if self._generation_is_stale(turn.gen) or not self._turn_is_latest(turn.turn_id, turn.turn_revision):
            logger.info("LLM generation cancelled (interruption)")
            return
        for event in events:
            if isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens
            elif isinstance(event, AssistantMessage):
                state.pending.append(
                    RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=event.content)
                )
            elif isinstance(event, ToolCall):
                yield from self._record_tool_call(state, turn, event.item)
            elif isinstance(event, TextDelta):
                # Text-only keeps every character verbatim; audio strips
                # TTS-unfriendly symbols via remove_unspeechable.
                spoken = event.text if not turn.wants_audio else remove_unspeechable(event.text)
                state.clean_text += spoken
                out = spoken if not turn.wants_audio else spoken.strip()
                if (
                    out
                    and not self._generation_is_stale(turn.gen)
                    and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
                ):
                    yield self._chunk(turn, text=out)
        logger.debug(f"Clean text: {state.clean_text}")
        logger.info(f"Tools: {state.tools}")

    # ── orchestration ─────────────────────────────────────────────────────────

    def _generate(
        self,
        active_chat: Chat,
        original_chat: Chat,
        turn: _Turn,
        optional_kwargs: dict[str, Any],
    ) -> Iterator[LLMOut]:
        api_response: Any = None
        events: Iterator[ProviderEvent] = iter(())
        state = _GenState()
        error_message: str | None = None
        api_input = self._serialize(active_chat)
        # Images the model actually sees this turn; only these are stripped on
        # write-back, so an image a fast client injects mid-generation for the
        # next turn survives (it is not in this serialized snapshot).
        consumed_image_ids = active_chat.image_message_ids()
        if not api_input:
            # Nothing to send: empty `instructions` and no `input` (in the response,
            # the default conversation, or the out-of-band context). The provider
            # would reject this; fail with a clear message instead of an opaque error.
            error_message = "Cannot generate a response: no instructions and no input were provided."

        try:
            request_started_at_s = perf_counter()
            if error_message is None:
                logger.info(
                    "Streaming LLM start (turn=%s rev=%s stream=%s)",
                    turn.turn_id,
                    turn.turn_revision,
                    self.stream,
                )
                self._wait_for_connection_prewarm()
                api_response, events = self._open_stream(api_input, optional_kwargs, turn)
                self._last_connection_use_s = perf_counter()
            if api_response is not None:
                if self.stream:
                    yield from self._consume_streaming(events, state, turn, request_started_at_s)
                else:
                    yield from self._consume_nonstreaming(events, state, turn)
        except httpx.ReadTimeout:
            logger.warning(
                "OpenAI API read timed out after %.1fs; ending the current response",
                self.request_timeout_s,
            )
            if not self._generation_is_stale(turn.gen) and self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                # Canned apology carries no language_code (mirrors the prior handlers).
                yield LLMResponseChunk(
                    text="Wow I'm a bit slow today, could you repeat that?",
                    runtime_config=turn.runtime_config,
                    response=turn.response,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                    speech_stopped_at_s=turn.speech_stopped_at_s,
                    cancel_generation=turn.gen,
                )
        except Exception as exc:
            # Any other generation failure must still terminate the response: record
            # the error and fall through to the EndOfResponse below. Without this the
            # exception would escape process() and no EndOfResponse would be emitted,
            # leaving st.in_response stuck and locking every subsequent response.
            logger.exception("LLM generation failed; ending the current response")
            if error_message is None:
                error_message = f"Language model generation failed: {exc}"
        finally:
            if api_response is not None and hasattr(api_response, "close"):
                try:
                    api_response.close()
                except Exception:
                    pass

        if not is_out_of_band(turn.response):
            # Images are one-turn inputs. Remove exactly the image-bearing
            # messages captured in this request even when the provider rejects
            # or the response is cancelled; otherwise one bad image poisons
            # every later text-only turn.
            original_chat.strip_images(consumed_image_ids)

        if (
            error_message is None
            and not self._generation_is_stale(turn.gen)
            and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
        ):
            # Out-of-band responses emit output and usage but never write back to the
            # default conversation (their context was a throwaway chat).
            if not is_out_of_band(turn.response):
                # Tool calls (and any assistant text preceding them) were already
                # written eagerly in _record_tool_call; only trailing items remain.
                for item in state.pending:
                    original_chat.add_item(item)
                original_chat.trim_if_needed(self.compactor)
            if state.input_tokens or state.output_tokens:
                yield TokenUsage(
                    input_tokens=state.input_tokens,
                    output_tokens=state.output_tokens,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                )
        yield EndOfResponse(
            turn_id=turn.turn_id, turn_revision=turn.turn_revision, cancel_generation=turn.gen, error=error_message
        )

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        """Process a language model request and yield LLMResponseChunks."""
        runtime_config = request.runtime_config
        response = request.response
        turn_id = request.turn_id
        turn_revision = request.turn_revision
        speech_stopped_at_s = request.speech_stopped_at_s
        if not self._turn_is_latest(turn_id, turn_revision):
            logger.info("Skipping stale LLM request for turn=%s rev=%s", turn_id, turn_revision)
            yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision)
            return

        original_chat = runtime_config.chat
        if is_out_of_band(response):
            try:
                active_chat = build_active_chat(original_chat, response)
            except ChatItemError as exc:
                logger.info("Out-of-band response rejected: %s", exc)
                yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision, error=str(exc))
                return
        else:
            active_chat = original_chat.copy()
        language_code = request.language_code
        instructions = (
            response.instructions if response and response.instructions else runtime_config.session.instructions
        ) or ""
        req_tools = response.tools if response and response.tools else runtime_config.session.tools
        req_tool_choice = (
            response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
        )
        wants_audio = response_wants_audio(response)
        self._apply_config(active_chat, instructions, wants_audio)
        language_code, lang_name = resolve_auto_language(language_code)
        if lang_name and self.enable_lang_prompt:
            active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

        optional_kwargs = self._build_optional_kwargs(req_tools, req_tool_choice)

        # CancelScope.is_stale(gen) is checked when the stream iterator advances; a
        # blocked read inside httpx cannot be aborted by cancel_scope.cancel() from
        # the websocket router. Mitigations: request_timeout_s / ReadTimeout.
        gen = self.cancel_scope.generation if self.cancel_scope else None

        turn = _Turn(
            language_code=language_code,
            gen=gen,
            runtime_config=runtime_config,
            response=response,
            turn_id=turn_id,
            turn_revision=turn_revision,
            speech_stopped_at_s=speech_stopped_at_s,
            wants_audio=wants_audio,
        )
        yield from self._generate(active_chat, original_chat, turn, optional_kwargs)

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: LLMOut) -> bool:
        return isinstance(output, LLMResponseChunk) and self.last_time > self.min_time_to_debug

    def cleanup(self) -> None:
        if not self._owns_http_client:
            return
        close_client = getattr(self.client, "close", None)
        if callable(close_client):
            close_client()
            return
        close_http = getattr(self._http_client, "close", None)
        if callable(close_http):
            close_http()
