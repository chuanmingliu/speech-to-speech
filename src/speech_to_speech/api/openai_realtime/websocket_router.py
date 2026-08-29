import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from threading import Thread
from typing import Any, Callable, TypeVar

import numpy as np
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from openai.types.realtime import (
    ConversationItemCreateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    OutputAudioBufferClearEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
    SessionUpdateEvent,
)

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit, SessionState
from speech_to_speech.api.openai_realtime.service import (
    PIPELINE_SAMPLE_RATE,
    build_error_event,
)
from speech_to_speech.api.openai_realtime.transports import (
    SessionTransport,
    WebSocketTransport,
    log_realtime_event,
    send_ws_event,
    tel_log,
)
from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PartialTranscriptionEvent,
    PipelineEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.log_context import pipeline_log_ctx
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput

# aiortc (the 'webrtc' extra) is optional. Import it here, at module load,
# rather than lazily in the calls endpoint: the av/cryptography C extensions
# take up to a second to load cold, which would block the shared event loop —
# and every live conversation's audio — on the first WebRTC handshake.
try:
    from aiortc import RTCPeerConnection

    from speech_to_speech.api.openai_realtime.webrtc_session import (
        WebRTCSession,
        rtc_configuration_from_env,
    )

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

logger = logging.getLogger(__name__)
MAX_AUDIO_BATCH_BYTES = 6400
DEFAULT_CONNECTION_MAINTENANCE_INTERVAL_S = 30.0
# How long the release path waits for SESSION_END to propagate through the
# handler chain back to output_queue before warning that the unit is stuck.
# Tests monkeypatch this to a small value since their fixtures usually skip
# the real handler chain.
SESSION_END_DRAIN_TIMEOUT_S = 10.0
# Past this, the unit is quarantined: its service session is unregistered
# (closing the chat so late handler output can't mutate or bill it), but the
# unit stays unclaimable. Releasing it instead would let a new client claim a
# unit whose handlers may still emit the previous session's output (e.g. a
# transcript, which carries no session identity and would be appended to the
# new session's conversation) — a cross-session leak. If SESSION_END does
# eventually drain, the chain has proven itself clean and the unit returns to
# the pool; a dead handler keeps it quarantined forever, visible in /v1/pool
# as "stuck".
SESSION_END_QUARANTINE_TIMEOUT_S = 180.0
QItem = TypeVar("QItem")


def _keep_audio_sentinel(item: Any) -> bool:
    # SESSION_END must survive barge-in flushes of output_queue: dropping it
    # would leave the release path waiting forever for the drain signal.
    return _is_audio_done(item) or is_control_message(item, SESSION_END.kind)


def _keep_user_text_event(item: Any) -> bool:
    return isinstance(
        item,
        (SpeechStoppedEvent, PartialTranscriptionEvent, TranscriptionCompletedEvent, TokenUsageEvent),
    )


def _audio_payload(item: Any) -> Any:
    return item.audio if isinstance(item, AudioOutput) else item


def _audio_generation(item: Any) -> int | None:
    return item.cancel_generation if isinstance(item, AudioOutput) else None


def _flush_queue(q: Queue[QItem], *, preserve: Callable[[QItem], bool] | None = None) -> None:
    """Drain a queue, optionally preserving items matching *preserve*.

    Preserved items are re-inserted at the **front** of the queue
    (atomically under the queue's mutex) so they are processed before
    anything a pipeline thread may have enqueued during the drain.
    """
    preserved: list[QItem] = []
    while True:
        try:
            item = q.get_nowait()
            if preserve and preserve(item):
                preserved.append(item)
        except Empty:
            break
    if preserved:
        with q.mutex:
            for item in reversed(preserved):
                q.queue.appendleft(item)
            q.not_empty.notify(len(preserved))


async def _drain_pending_response_events(
    transport: SessionTransport | None,
    unit: PipelineUnit,
    session_id: str | None,
) -> None:
    if session_id is None:
        return

    preserved: list[Any] = []
    drained_assistant = 0
    drained_usage = 0
    drain_assistant_events = True
    try:
        while True:
            try:
                item = unit.text_output_queue.get_nowait()
            except Empty:
                break
            # Usage is accounting-only, so keep the old whole-queue drain behavior.
            # Assistant events are client-visible response output and stop at the
            # first non-response boundary to preserve normal text-event ordering.
            if isinstance(item, TokenUsageEvent):
                unit.service.dispatch_pipeline_event(session_id, item)
                drained_usage += 1
            elif drain_assistant_events and isinstance(item, AssistantTextEvent):
                drained_assistant += 1
                if _generation_is_discardable(unit, item.cancel_generation):
                    continue
                events = unit.service.dispatch_pipeline_event(session_id, item)
                if transport is not None and events:
                    await transport.send_events(events)
            else:
                preserved.append(item)
                drain_assistant_events = False
    finally:
        if preserved:
            with unit.text_output_queue.mutex:
                for item in reversed(preserved):
                    unit.text_output_queue.queue.appendleft(item)
                unit.text_output_queue.not_empty.notify(len(preserved))

    if drained_assistant or drained_usage:
        logger.debug(
            "Pipeline %d: drained %d assistant event(s) and %d token usage event(s) before response completion",
            unit.index,
            drained_assistant,
            drained_usage,
        )


def _clean_unit(unit: PipelineUnit, preserve: Callable[[Any], bool] | None = None) -> None:
    """Cancel in-flight work and flush queues for a single pipeline unit.

    All four pipeline queues are drained — input audio, transcript-to-LM,
    LM-to-TTS output, and the text-event side channel — so pending work from
    a released session cannot be picked up by handlers and leak into the next
    session that claims this unit. SESSION_END is enqueued by the route
    handler *after* this returns to serve as the soft reset signal for
    stateful handlers.
    """
    unit.cancel_scope.cancel()
    _flush_queue(unit.input_queue)
    _flush_queue(unit.text_prompt_queue)
    _flush_queue(unit.output_queue, preserve=preserve)
    _flush_queue(unit.text_output_queue, preserve=preserve)
    unit.response_playing.clear()
    unit.cancel_scope.reset()
    unit.should_listen.set()


def _run_prewarm(index: int, handler: Any) -> None:
    pipeline_log_ctx.set(index)
    try:
        handler.prewarm()
    except Exception:
        logger.exception("Pipeline %d: %s prewarm failed", index, type(handler).__name__)


def _prewarm_unit(unit: PipelineUnit) -> None:
    """Refresh idle hosted-provider connections without blocking the event loop."""
    for handler in unit.handlers:
        if not callable(getattr(handler, "prewarm", None)):
            continue
        Thread(
            target=_run_prewarm,
            args=(unit.index, handler),
            name=f"pipeline-{unit.index}-{type(handler).__name__}-prewarm",
            daemon=True,
        ).start()


def _run_connection_maintenance(index: int, handler: Any) -> None:
    pipeline_log_ctx.set(index)
    try:
        handler.maintain_connection()
    except Exception:
        logger.exception(
            "Pipeline %d: %s connection maintenance failed",
            index,
            type(handler).__name__,
        )


def _maintain_idle_unit(unit: PipelineUnit) -> None:
    """Refresh non-billable provider transports for an unclaimed pool lane."""
    if unit.session is not None:
        return
    for handler in unit.handlers:
        if not callable(getattr(handler, "maintain_connection", None)):
            continue
        Thread(
            target=_run_connection_maintenance,
            args=(unit.index, handler),
            name=f"pipeline-{unit.index}-{type(handler).__name__}-maintenance",
            daemon=True,
        ).start()


def _to_audio_bytes(chunk: Any) -> bytes:
    chunk = _audio_payload(chunk)
    if isinstance(chunk, PipelineControlMessage):
        raise TypeError(f"unexpected control message on audio output queue: {chunk!r}")
    if isinstance(chunk, np.ndarray) or hasattr(chunk, "tobytes"):
        return chunk.tobytes()
    return chunk


def _is_audio_done(item: Any) -> bool:
    payload = _audio_payload(item)
    return isinstance(payload, bytes) and payload == AUDIO_RESPONSE_DONE


def _is_pipeline_end(item: Any) -> bool:
    payload = _audio_payload(item)
    return isinstance(payload, bytes) and payload == PIPELINE_END


def _generation_is_discardable(unit: PipelineUnit, generation: int | None) -> bool:
    """Whether output tagged with *generation* should be dropped.

    A generation is discardable if it has been superseded (``is_stale``) or if the
    cancel scope is in its post-cancel discard window and this is not the current
    live generation. Shared by audio and assistant-text so the two paths stay in
    lockstep: dropping text whenever ``discarding`` is set (without this generation
    check) silently swallows the transcript of a fresh response when ``discarding``
    lingers — e.g. a superseded speculative turn whose TTS never emitted an
    AUDIO_RESPONSE_DONE sentinel, so response_done() never cleared the flag.
    """
    if generation is not None and unit.cancel_scope.is_stale(generation):
        return True
    if unit.cancel_scope.discarding and generation != unit.cancel_scope.generation:
        return True
    return False


def _should_discard_audio(unit: PipelineUnit, item: Any) -> bool:
    return _generation_is_discardable(unit, _audio_generation(item))


def _opening_from_instructions(text: str) -> str:
    import re

    if not text:
        return ""
    tagged = re.search(r"Opening line:[\s\S]*?turn:\s*\n([^\n]+)", text, re.I)
    if tagged and tagged.group(1).strip():
        return tagged.group(1).strip()[:280]
    labeled = re.search(r"Opening line:\s*\n([^\n]+)", text, re.I)
    if labeled and labeled.group(1).strip():
        return labeled.group(1).strip()[:280]
    return ""


def _opening_from_session(session: Any) -> str:
    return _opening_from_instructions(str(getattr(session, "instructions", None) or ""))


def _tel(st: Any, event: str, **kv: Any) -> None:
    call = str(getattr(st, "tel_call", "") or "")
    session = str(getattr(st, "session_id", "") or "")[:12]
    tel_log("s2s", event, t0=getattr(st, "connected_at_s", None), call=call or session, session=session, **kv)


def _conn_state(unit: PipelineUnit, session_id: str | None) -> Any | None:
    if not session_id:
        return None
    try:
        return unit.service._state(session_id)
    except KeyError:
        return None


def _holds_input_until_answer(st: Any) -> bool:
    if st.answered or st.opening_kicked:
        return False
    return bool(st.opening_line or _opening_from_session(st.runtime_config.session))


def _holds_output_until_answer(st: Any) -> bool:
    return (not st.answered) and bool(st.opening_line or st.opening_warming)


def _session_gone(unit: PipelineUnit, session_id: str) -> bool:
    sess = unit.session
    return sess is None or sess.session_id != session_id or sess.released_at is not None


def _pcm16_bytes(block: Any) -> bytes:
    if isinstance(block, (bytes, bytearray)):
        return bytes(block)
    if isinstance(block, np.ndarray):
        arr = np.ascontiguousarray(block)
        if arr.dtype != np.int16:
            clipped = np.clip(arr.astype(np.float64), -1.0, 1.0)
            arr = (clipped * 32767.0).astype(np.int16)
        return arr.tobytes()
    if hasattr(block, "tobytes"):
        return block.tobytes()
    return b""


def _minimax_opening_pcm(text: str, *, cancelled: Callable[[], bool]) -> bytes:
    api_key = os.getenv("MINIMAX_TTS_API_KEY") or os.getenv("MINIMAX_API_KEY")
    voice_id = os.getenv("MINIMAX_TTS_VOICE_ID")
    if not text or not api_key or not voice_id or cancelled():
        return b""
    from speech_to_speech.TTS.minimax_tts_handler import (  # noqa: PLC0415
        MiniMaxStreamingClient,
        MiniMaxTTSConfig,
        _resolved_minimax_speed,
    )

    cfg = MiniMaxTTSConfig(
        api_key=api_key,
        voice_id=voice_id,
        model=os.getenv("MINIMAX_TTS_MODEL", "speech-2.8-turbo"),
        endpoint=os.getenv("MINIMAX_TTS_ENDPOINT", "wss://api.minimax.io/ws/v1/t2a_v2"),
        language_boost=os.getenv("MINIMAX_TTS_LANGUAGE_BOOST", "auto"),
        speed=_resolved_minimax_speed(None),
    )
    client = MiniMaxStreamingClient(cfg)
    buf = bytearray()
    try:
        client.start(cancelled=cancelled)
        for block in client.synthesize(text, cancelled=cancelled):
            buf.extend(_pcm16_bytes(block))
        for block in client.finish(cancelled=cancelled):
            buf.extend(_pcm16_bytes(block))
    finally:
        try:
            client.close(graceful=True)
        except Exception:
            pass
    return bytes(buf)


def _start_opening_cache(unit: PipelineUnit, session_id: str, line: str) -> None:
    st = unit.service._state(session_id)
    text = (line or st.opening_line or _opening_from_session(st.runtime_config.session)).strip()[:280]
    if not text:
        return
    st.opening_line = text
    if st.opening_warming or st.opening_kicked:
        return
    st.opening_warming = True
    _tel(st, "opening_cache_start")

    def run() -> None:
        try:
            pcm = _minimax_opening_pcm(text, cancelled=lambda: _session_gone(unit, session_id))
            if not pcm or _session_gone(unit, session_id):
                st.opening_warming = False
                _tel(st, "opening_cache_empty")
                return
            step = 512 * 2
            for i in range(0, len(pcm), step):
                if _session_gone(unit, session_id):
                    return
                unit.output_queue.put(AudioOutput(audio=pcm[i : i + step]))
            unit.output_queue.put(AUDIO_RESPONSE_DONE)
            st.opening_ready = True
            st.opening_warming = False
            _tel(st, "opening_cached", bytes=len(pcm))
        except Exception:
            logger.exception("Opening MiniMax cache failed")
            st.opening_warming = False
            _tel(st, "opening_cache_fail")

    Thread(target=run, daemon=True, name="opening-cache").start()


def _stash_opening_audio(session: SessionState, unit: PipelineUnit, session_id: str, audio_chunk: Any) -> bool:
    st = _conn_state(unit, session_id)
    if st is None or not _holds_output_until_answer(st):
        return False
    if _is_audio_done(audio_chunk):
        session.held_opening_done = True
        st.opening_ready = True
        return True
    if _is_pipeline_end(audio_chunk) or is_control_message(audio_chunk):
        return False
    if _should_discard_audio(unit, audio_chunk):
        return True
    pcm = _to_audio_bytes(audio_chunk)
    if pcm:
        session.held_opening.append(pcm)
        st.opening_ready = True
        if not st.opening_hold_logged:
            st.opening_hold_logged = True
            _tel(st, "opening_held", bytes=len(pcm))
    return True


def _next_held_opening(session: SessionState, session_id: str, unit: PipelineUnit) -> Any:
    st = _conn_state(unit, session_id)
    if st is None or not st.answered:
        return None
    if session.held_opening:
        return session.held_opening.pop(0)
    if session.held_opening_done:
        session.held_opening_done = False
        return AUDIO_RESPONSE_DONE
    return None


async def _kick_opening_if_needed(unit: PipelineUnit, session_id: str, transport: SessionTransport) -> None:
    st = unit.service._state(session_id)
    st.answered = True
    if st.opening_kicked:
        return
    if st.opening_ready or st.opening_warming:
        st.opening_kicked = True
        _tel(st, "opening_flush", cached=1)
        return
    line = st.opening_line or _opening_from_session(st.runtime_config.session)
    if not line:
        return
    st.opening_kicked = True
    _tel(st, "opening_kick_llm")
    item_event = unit.service.parse_client_event(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"The callee answered. Speak now. Say exactly this sentence and nothing else: {line}",
                    }
                ],
            },
        }
    )
    if item_event is not None:
        events = unit.service.handle_conversation_item_create(session_id, item_event)
        if events:
            await transport.send_events(events)
    create_event = unit.service.parse_client_event({"type": "response.create"})
    if create_event is not None:
        result = unit.service.handle_response_create(session_id, create_event)
        if result:
            if result.type != "error":
                unit.cancel_scope.new_response()
            await transport.send_events([result])


def _safe_unregister(unit: PipelineUnit, session_id: str) -> None:
    try:
        unit.service.unregister(session_id)
    except Exception:
        logger.exception(f"Pipeline {unit.index}: unregister failed for session {session_id}")


async def _release_unit_after_drain(unit: PipelineUnit, session: Any, session_id: str) -> None:
    """Wait for SESSION_END to propagate, then release the unit.

    Runs in its own asyncio task so the route handler's finally block can return
    immediately. The unit stays unavailable for new claims (unit.session != None)
    until SESSION_END travels all the way through the handler chain back to
    output_queue — observed by the send loop, which sets session.drained.

    Past SESSION_END_QUARANTINE_TIMEOUT_S (a wedged or dead handler thread) the
    unit is quarantined, NOT released: still-running handlers could emit the old
    session's output (transcripts carry no session identity) into whichever
    session claimed the unit next, and a dead handler would make the unit accept
    clients it can never serve. The session is unregistered right away so late
    output can't mutate or bill the closed conversation; the unit itself only
    returns to the pool if SESSION_END eventually drains, proving the chain is
    clean. Operators can spot quarantined units in `/v1/pool` (state "stuck").
    """
    elapsed = 0.0
    warned = False
    try:
        while not session.drained.is_set():
            await asyncio.sleep(0.05)
            elapsed += 0.05
            if not warned and elapsed >= SESSION_END_DRAIN_TIMEOUT_S:
                logger.warning(
                    f"Pipeline {unit.index}: SESSION_END not drained after {elapsed:.1f}s — "
                    f"unit will remain unavailable until handlers finish (session {session_id})"
                )
                warned = True
            if session.quarantined_at is None and elapsed >= SESSION_END_QUARANTINE_TIMEOUT_S:
                session.quarantined_at = time.monotonic()
                _safe_unregister(unit, session_id)
                logger.error(
                    f"Pipeline {unit.index}: SESSION_END still not drained after {elapsed:.0f}s — "
                    f"quarantining unit until the handler chain drains (session {session_id})"
                )
    finally:
        # Runs when the drain completed (chain proven clean) or the task is
        # cancelled at shutdown. Release unconditionally: even if unregister
        # raises, the unit must not stay claimed forever.
        try:
            _safe_unregister(unit, session_id)
        finally:
            unit.session = None
        recovered = " after quarantine" if session.quarantined_at is not None else ""
        logger.info(f"Pipeline {unit.index} released{recovered} (session {session_id} ended)")


# Strong references to in-flight drain-and-release tasks (asyncio only
# holds tasks weakly); each task removes itself on completion.
_release_tasks: set[asyncio.Task[None]] = set()


def _release_session(unit: PipelineUnit, session_id: str) -> None:
    """Start the release of a unit after its client disconnected.

    Shared by the WebSocket route's finally block and the WebRTC session's
    close callback. Marks the session as released, resets the unit, enqueues
    SESSION_END, and spawns the drain-and-release task — the unit stays
    claimed until SESSION_END propagates back to output_queue.
    """
    old_session = unit.session
    if old_session is None:
        # Already released (e.g. duplicate close callbacks racing).
        return
    old_session.released_at = time.monotonic()
    _clean_unit(unit)
    # Tag SESSION_END with this session's id so that, after a force
    # release, a late arrival can't satisfy the next session's drain.
    unit.input_queue.put(PipelineControlMessage(SESSION_END.kind, session_id=session_id))
    task = asyncio.create_task(_release_unit_after_drain(unit, old_session, session_id))
    _release_tasks.add(task)
    task.add_done_callback(_release_tasks.discard)


async def _dispatch_client_event(
    unit: PipelineUnit,
    session_id: str,
    raw: dict[str, Any],
    transport: SessionTransport,
    *,
    transport_kind: str = "websocket",
) -> None:
    """Parse and apply one client event, replying over *transport*.

    Shared by both transports; ``transport_kind`` gates the events whose
    validity depends on how audio travels: ``input_audio_buffer.append`` is
    WebSocket-only (WebRTC audio arrives on the media track), and
    ``output_audio_buffer.clear`` is WebRTC-only (over WebSocket the unplayed
    audio sits client-side).
    """
    service = unit.service
    event = service.parse_client_event(raw)
    if event is not None:
        log_realtime_event("in", event)
    if event is None:
        await transport.send_events(
            [service.make_error(f"Unknown or invalid event: {raw.get('type')}", "unknown_or_invalid_event")]
        )
        return

    if isinstance(event, InputAudioBufferAppendEvent):
        if transport_kind == "webrtc":
            await transport.send_events(
                [
                    service.make_error(
                        "In WebRTC mode audio arrives via the media track; input_audio_buffer.append is not supported.",
                        "invalid_event_for_transport",
                    )
                ]
            )
            return
        st = service._state(session_id)
        if _holds_input_until_answer(st):
            return
        chunks = service.handle_audio_append(session_id, event)
        rt_cfg = st.runtime_config
        for chunk in chunks:
            unit.input_queue.put((chunk, rt_cfg))

    elif isinstance(event, InputAudioBufferCommitEvent):
        err = service.handle_audio_commit(session_id)
        if err:
            await transport.send_events([err])

    elif isinstance(event, OutputAudioBufferClearEvent):
        if transport_kind != "webrtc":
            await transport.send_events(
                [
                    service.make_error(
                        "output_audio_buffer.clear is only supported on the WebRTC transport.",
                        "invalid_event_for_transport",
                    )
                ]
            )
            return
        _flush_queue(unit.output_queue, preserve=_keep_audio_sentinel)
        transport.discard_pending_audio()

    elif isinstance(event, SessionUpdateEvent):
        err = service.handle_session_update(session_id, event)
        if err:
            await transport.send_events([err])
            return
        st = service._state(session_id)
        line = _opening_from_session(st.runtime_config.session)
        if line:
            st.opening_line = line
            _tel(st, "session_update", opening=1)
            _start_opening_cache(unit, session_id, line)

    elif isinstance(event, ConversationItemCreateEvent):
        events = service.handle_conversation_item_create(session_id, event)
        if events:
            await transport.send_events(events)

    elif isinstance(event, ResponseCreateEvent):
        result = service.handle_response_create(session_id, event)
        if result:
            if result.type != "error":
                unit.cancel_scope.new_response()
            await transport.send_events([result])

    elif isinstance(event, ResponseCancelEvent):
        was_active = service._state(session_id).in_response
        if was_active:
            unit.cancel_scope.cancel()
        _flush_queue(unit.output_queue, preserve=_keep_audio_sentinel)
        _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
        transport.discard_pending_audio()
        events = service.handle_response_cancel(session_id)
        if events:
            await transport.send_events(events)
        unit.response_playing.clear()


def create_app(
    pool: list[PipelineUnit],
    stop_event: ThreadingEvent,
    *,
    connection_maintenance_interval_s: float | None = None,
) -> FastAPI:
    if connection_maintenance_interval_s is None:
        raw_interval = os.getenv(
            "PROVIDER_CONNECTION_MAINTENANCE_S",
            str(DEFAULT_CONNECTION_MAINTENANCE_INTERVAL_S),
        )
        try:
            connection_maintenance_interval_s = float(raw_interval)
        except ValueError:
            logger.warning(
                "Invalid PROVIDER_CONNECTION_MAINTENANCE_S=%r; using %.0fs",
                raw_interval,
                DEFAULT_CONNECTION_MAINTENANCE_INTERVAL_S,
            )
            connection_maintenance_interval_s = DEFAULT_CONNECTION_MAINTENANCE_INTERVAL_S
    connection_maintenance_interval_s = max(0.0, connection_maintenance_interval_s)

    async def _maintain_idle_pool() -> None:
        assert connection_maintenance_interval_s is not None
        while not stop_event.is_set():
            await asyncio.sleep(connection_maintenance_interval_s)
            for unit in pool:
                _maintain_idle_unit(unit)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One send loop per pipeline unit; each polls its own queues and forwards
        # to the websocket currently attached via unit.session.
        send_tasks = [asyncio.create_task(_send_loop_for(unit)) for unit in pool]
        maintenance_task: asyncio.Task[None] | None = None
        if connection_maintenance_interval_s > 0:
            for unit in pool:
                _maintain_idle_unit(unit)
            maintenance_task = asyncio.create_task(_maintain_idle_pool())
        yield
        if maintenance_task is not None:
            maintenance_task.cancel()
            try:
                await maintenance_task
            except asyncio.CancelledError:
                pass
        for task in send_tasks:
            task.cancel()
        for task in send_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for unit in pool:
            sess = unit.session
            if sess is not None and sess.transport is not None:
                try:
                    await sess.transport.close()
                except Exception:
                    pass

    app = FastAPI(lifespan=lifespan)

    def _service_info() -> dict[str, Any]:
        return {
            "service": "speech-to-speech",
            "realtime": "/v1/realtime",
            "usage": "/v1/usage",
            "pool": "/v1/pool",
        }

    @app.get("/")
    async def root() -> dict[str, Any]:
        return _service_info()

    @app.get("/json/version")
    async def inspector_probe() -> dict[str, Any]:
        """Browser inspectors probe this Chrome DevTools path on the listen port."""
        return _service_info()

    def _claim_unit(transport: SessionTransport | None) -> PipelineUnit | None:
        """Atomically (between asyncio yield points) reserve the first idle unit.

        Creates a placeholder SessionState that the caller fills in with the
        session_id after RealtimeService.register(). The WebRTC route claims
        with transport=None and attaches the session object once constructed.
        """
        for unit in pool:
            if unit.session is None:
                unit.session = SessionState(transport=transport)
                return unit
        return None

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        transport = WebSocketTransport(ws)
        unit = _claim_unit(transport)
        if unit is None:
            logger.warning(f"Rejected connection: all {len(pool)} pipeline slots in use")
            # Stateless error event — rejection is not chargeable to any unit's usage metrics.
            await send_ws_event(
                ws,
                build_error_event(
                    f"All {len(pool)} session slots are in use. Disconnect an existing client first.",
                    error_type="session_limit_reached",
                ),
            )
            await ws.close(code=1008, reason="All session slots are in use")
            return

        pipeline_log_ctx.set(unit.index)
        # _claim_unit guarantees unit.session is not None for the returned unit.
        assert unit.session is not None
        # Everything after the claim runs inside try so the finally below always
        # releases the unit, even if session setup fails.
        session_id = ""
        try:
            session_id = unit.service.register()
            unit.session.session_id = session_id
            st = unit.service._state(session_id)
            st.tel_call = (ws.query_params.get("call") or "").strip()[:48]
            _tel(st, "ws_open", pipeline=unit.index, brain="local")
            logger.info(f"Client connected to pipeline {unit.index} (session {session_id})")

            # Defensive: drain edge queues and reset events so stale data from a
            # previous session that survived SESSION_END propagation doesn't leak.
            _clean_unit(unit)
            _prewarm_unit(unit)

            await send_ws_event(ws, unit.service.build_session_created(session_id))

            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                await _dispatch_client_event(unit, session_id, raw, transport)

        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected from pipeline {unit.index}")
        except Exception as e:
            logger.error(f"Client {session_id} on pipeline {unit.index} error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            if session_id:
                try:
                    st = unit.service._state(session_id)
                    _tel(st, "hangup", why="ws_close")
                except KeyError:
                    tel_log("s2s", "hangup", session=session_id[:12], why="ws_close")
            # Hold the session reference: the send loop's snapshot will still resolve
            # to this object until we clear unit.session, so any handler output that
            # arrives during the drain window is sent to the now-closed ws (silently
            # dropped) instead of leaking to whichever client claims this unit next.
            # _release_session spawns the drain-and-release as a separate task so
            # this finally returns immediately. Awaiting here is unreliable: after
            # WebSocketDisconnect propagates, subsequent awaits in the same task
            # can be skipped/cancelled by Starlette's runner and never resume.
            _release_session(unit, session_id)

    @app.get("/v1/usage")
    async def usage_endpoint() -> dict[str, Any]:
        # Aggregate usage across the pool. Numeric fields sum; dict fields (e.g.
        # errors_by_type) merge with numeric leaves summed too, so per-unit error
        # counts don't get dropped by the first-unit's value.
        def _merge(into: dict[str, Any], src: dict[str, Any]) -> None:
            for k, v in src.items():
                if isinstance(v, (int, float)):
                    into[k] = into.get(k, 0) + v
                elif isinstance(v, dict):
                    sub = into.setdefault(k, {})
                    if isinstance(sub, dict):
                        _merge(sub, v)
                else:
                    into.setdefault(k, v)

        total: dict[str, Any] = {}
        for unit in pool:
            _merge(total, unit.service.get_usage())
        return total

    @app.get("/v1/pool")
    async def pool_endpoint() -> dict[str, Any]:
        now = time.monotonic()

        def _state(u: PipelineUnit) -> dict[str, Any]:
            s = u.session
            if s is None:
                return {"index": u.index, "state": "idle", "session_id": None}
            if s.released_at is None:
                return {"index": u.index, "state": "active", "session_id": s.session_id}
            # Drain wait gave up (quarantine timeout): the unit stays occupied
            # until SESSION_END actually drains — possibly forever if a handler
            # thread died. Surfaced distinctly so operators can act on it.
            if s.quarantined_at is not None:
                return {
                    "index": u.index,
                    "state": "stuck",
                    "session_id": s.session_id,
                    "draining_for_s": round(now - s.released_at, 2),
                    "stuck_for_s": round(now - s.quarantined_at, 2),
                }
            # released by client but SESSION_END hasn't drained yet → unit
            # is still occupied; surface elapsed time so operators can spot
            # stuck handlers.
            return {
                "index": u.index,
                "state": "draining",
                "session_id": s.session_id,
                "draining_for_s": round(now - s.released_at, 2),
            }

        return {
            "size": len(pool),
            "in_use": sum(1 for u in pool if u.session is not None),
            "units": [_state(u) for u in pool],
        }

    @app.post("/v1/opening")
    async def opening_endpoint(request: Request) -> dict[str, Any]:
        """Warm MiniMax opening PCM on Dial; flush it on SIP 200."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        greeting = str(body.get("greeting") or body.get("text") or "").strip()[:280]
        warm = bool(body.get("warm"))
        cached: list[dict[str, Any]] = []
        kicked: list[str] = []
        for unit in pool:
            session = unit.session
            if session is None or session.released_at is not None or not session.session_id:
                continue
            try:
                st = unit.service._state(session.session_id)
            except KeyError:
                continue
            if greeting:
                st.opening_line = greeting
            if warm:
                _start_opening_cache(unit, session.session_id, greeting or st.opening_line)
                cached.append(
                    {
                        "session": session.session_id,
                        "ready": st.opening_ready,
                        "bytes": sum(len(b) for b in session.held_opening),
                        "warming": st.opening_warming,
                    }
                )
                continue
            _tel(st, "pickup", ready=int(st.opening_ready))
            if session.transport is not None:
                await _kick_opening_if_needed(unit, session.session_id, session.transport)
            else:
                st.answered = True
            kicked.append(session.session_id)
            cached.append(
                {
                    "session": session.session_id,
                    "ready": st.opening_ready,
                    "bytes": sum(len(b) for b in session.held_opening),
                }
            )
        return {"ok": True, "cached": cached, "kicked": kicked, "warm": warm}

    @app.post("/v1/realtime/calls")
    async def webrtc_calls_endpoint(request: Request) -> Response:
        """WebRTC SDP handshake (OpenAI GA Realtime 'calls' endpoint).

        The client POSTs an SDP offer with Content-Type: application/sdp and
        receives an SDP answer. Audio then flows over WebRTC media tracks;
        events flow over the 'oai-events' data channel using the same JSON
        protocol as the WebSocket transport.
        """
        if not WEBRTC_AVAILABLE:
            return Response(
                content="WebRTC support requires the 'webrtc' extra: pip install 'speech-to-speech[webrtc]'",
                status_code=501,
                media_type="text/plain",
            )

        if "application/sdp" not in request.headers.get("content-type", ""):
            return Response(
                content="Content-Type must be application/sdp",
                status_code=415,
                media_type="text/plain",
            )
        offer_sdp = (await request.body()).decode("utf-8")

        # Claim with a placeholder transport; the send loop tolerates a
        # transport-less snapshot until the session object below is attached.
        unit = _claim_unit(None)
        if unit is None:
            logger.warning(f"Rejected WebRTC offer: all {len(pool)} pipeline slots in use")
            return Response(
                content=build_error_event(
                    f"All {len(pool)} session slots are in use. Disconnect an existing client first.",
                    error_type="session_limit_reached",
                ).model_dump_json(),
                status_code=503,
                media_type="application/json",
            )

        pipeline_log_ctx.set(unit.index)
        try:
            session_id = unit.service.register()
            assert unit.session is not None
            unit.session.session_id = session_id
            logger.info(f"WebRTC client claiming pipeline {unit.index} (session {session_id})")

            # Defensive: drain edge queues and reset events so stale data from a
            # previous session that survived SESSION_END propagation doesn't leak.
            _clean_unit(unit)
            _prewarm_unit(unit)
        except Exception as e:  # noqa: BLE001
            logger.error(f"WebRTC call setup failed (pipeline {unit.index}): {type(e).__name__}: {e}")
            # No transport or drain task exists yet, so undoing the claim
            # directly is the whole release.
            unit.session = None
            return Response(content="WebRTC session setup failed", status_code=500, media_type="text/plain")

        released = False

        def _on_closed() -> None:
            # close() is idempotent but can be reached from several aiortc
            # callbacks; release the unit exactly once.
            nonlocal released
            if released:
                return
            released = True
            logger.info(f"WebRTC client {session_id} disconnected from pipeline {unit.index}")
            _release_session(unit, session_id)

        async def _on_client_event(raw: dict[str, Any]) -> None:
            assert session is not None  # callbacks only fire after setup()
            await _dispatch_client_event(unit, session_id, raw, session, transport_kind="webrtc")

        def _on_audio(pcm: bytes) -> None:
            chunks = unit.service.append_pcm(session_id, pcm, PIPELINE_SAMPLE_RATE)
            if not chunks:
                return
            rt_cfg = unit.service._state(session_id).runtime_config
            for chunk in chunks:
                unit.input_queue.put((chunk, rt_cfg))

        async def _on_open() -> None:
            assert session is not None  # callbacks only fire after setup()
            await session.send_events([unit.service.build_session_created(session_id)])
            logger.info(f"WebRTC session.created sent (session {session_id})")

        # Any failure between the claim above and a successful negotiate()
        # must release the unit, or it stays occupied forever with no peer
        # attached — the connect watchdog only exists once negotiate() ran.
        session = None
        try:
            config = rtc_configuration_from_env()
            pc = RTCPeerConnection(configuration=config) if config is not None else RTCPeerConnection()
            session = WebRTCSession(
                pc,
                on_client_event=_on_client_event,
                on_audio=_on_audio,
                on_open=_on_open,
                on_closed=_on_closed,
            )
            session.setup()
            unit.session.transport = session
        except Exception as e:  # noqa: BLE001
            logger.error(f"WebRTC session setup failed (session {session_id}): {type(e).__name__}: {e}")
            if session is not None:
                await session.close()  # fires _on_closed → _release_session
            else:
                _on_closed()
            return Response(content="WebRTC session setup failed", status_code=500, media_type="text/plain")

        try:
            answer_sdp = await session.negotiate(offer_sdp)
        except Exception as e:  # noqa: BLE001
            logger.error(f"WebRTC negotiation failed (session {session_id}): {type(e).__name__}: {e}")
            await session.close()
            return Response(content="Invalid SDP offer", status_code=400, media_type="text/plain")

        logger.info(f"WebRTC SDP answer returned (session {session_id})")
        return Response(
            content=answer_sdp,
            status_code=201,
            media_type="application/sdp",
            headers={"Location": f"/v1/realtime/calls/{session_id}"},
        )

    @app.delete("/v1/realtime/calls/{call_id}")
    async def webrtc_hangup_endpoint(call_id: str) -> Response:
        """Hang up a WebRTC call — the Location URL advertised by the POST above."""
        for unit in pool:
            session = unit.session
            if (
                session is None
                or session.session_id != call_id
                or session.released_at is not None
                or session.transport is None
                or session.transport.kind != "webrtc"
            ):
                continue
            logger.info(f"WebRTC call {call_id} hung up via DELETE (pipeline {unit.index})")
            # close() fires the session's on_closed callback, which releases
            # the unit exactly once (idempotent with aiortc's own callbacks).
            await session.transport.close()
            return Response(status_code=200)
        return Response(content="Unknown call", status_code=404, media_type="text/plain")

    async def _send_loop_for(unit: PipelineUnit) -> None:
        """Per-pipeline send loop. Polls this unit's output queues and forwards
        to the transport currently attached via unit.session.

        Per-session scratch (pending_output_item) lives on SessionState, so it
        disappears together with the transport when the session is released —
        no stale sentinel can leak into the next claim.
        """
        pipeline_log_ctx.set(unit.index)
        while not stop_event.is_set():
            try:
                # Snapshot the session once per iteration; if the route releases the
                # unit mid-iteration, we continue against the prior snapshot which is
                # consistent (its transport is still valid until close() returns).
                session = unit.session
                transport = session.transport if session is not None else None
                session_id = session.session_id if session is not None else None

                # Text events first (speech_started cancels active response).
                try:
                    text_msg = unit.text_output_queue.get_nowait()
                    is_speech_start = isinstance(text_msg, SpeechStartedEvent)

                    was_in_response = False
                    was_response_pending = False
                    if is_speech_start and session_id:
                        st = unit.service._state(session_id)
                        was_in_response = st.in_response
                        was_response_pending = st.response_pending

                    if isinstance(text_msg, AssistantTextEvent) and _generation_is_discardable(
                        unit, text_msg.cancel_generation
                    ):
                        pass
                    elif transport is not None and isinstance(text_msg, PipelineEvent) and session_id:
                        events = unit.service.dispatch_pipeline_event(session_id, text_msg)
                        if events:
                            await transport.send_events(events)

                    if is_speech_start and session_id:
                        _prewarm_unit(unit)
                        active_cfg = unit.service._state(session_id).runtime_config
                        interrupt_enabled = text_msg.interrupt_response and (
                            active_cfg is None or active_cfg.interrupt_response_enabled
                        )
                        if interrupt_enabled and transport is not None:
                            # Flush even when no response is active: the WebRTC
                            # track can still hold unplayed audio from a response
                            # whose done-sentinel was already observed —
                            # finish_response() runs on the sentinel, not when
                            # playback completes. No-op over WebSocket.
                            transport.discard_pending_audio()
                        if was_in_response or was_response_pending:
                            if interrupt_enabled:
                                unit.cancel_scope.cancel()
                                unit.service._state(session_id).response_pending = False
                                _flush_queue(unit.output_queue, preserve=_keep_audio_sentinel)
                                _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
                                if unit.response_playing.is_set():
                                    unit.response_playing.clear()
                                logger.info(
                                    "Pipeline %d: speech during %s: cancelled, queue flushed",
                                    unit.index,
                                    "response" if was_in_response else "pending response",
                                )
                            else:
                                logger.info(
                                    f"Pipeline {unit.index}: speech during response: interrupt_response disabled, ignoring"
                                )
                except Empty:
                    pass

                try:
                    pulled_held = False
                    if session is not None and session_id:
                        held = _next_held_opening(session, session_id, unit)
                        if held is not None:
                            audio_chunk = held
                            pulled_held = True
                    if not pulled_held:
                        if session is not None and session.pending_output_item is not None:
                            audio_chunk = session.pending_output_item
                            session.pending_output_item = None
                        else:
                            audio_chunk = unit.output_queue.get_nowait()
                        if session is not None and session_id and _stash_opening_audio(
                            session, unit, session_id, audio_chunk
                        ):
                            continue

                    if _is_pipeline_end(audio_chunk):
                        await _drain_pending_response_events(transport, unit, session_id)
                        if transport is not None and session_id:
                            await transport.send_events(unit.service.finish_response(session_id))
                        break

                    if _is_audio_done(audio_chunk):
                        audio_generation = _audio_generation(audio_chunk)
                        if audio_generation is not None and unit.cancel_scope.is_stale(audio_generation):
                            if session_id:
                                unit.service._state(session_id).response_pending = False
                            unit.cancel_scope.response_done(audio_generation)
                            unit.should_listen.set()
                            logger.info(f"Pipeline {unit.index}: stale response complete, listening re-enabled")
                            continue
                        await _drain_pending_response_events(transport, unit, session_id)
                        if transport is not None and session_id:
                            await transport.send_events(unit.service.finish_response(session_id))
                        if session_id:
                            unit.service._state(session_id).response_pending = False
                        unit.response_playing.clear()
                        unit.cancel_scope.response_done(audio_generation)
                        unit.should_listen.set()
                        logger.info(f"Pipeline {unit.index}: response complete, listening re-enabled")
                        continue

                    # SESSION_END travels from input_queue through every handler to
                    # output_queue. Observing it here means the chain has fully reset;
                    # signal the release path so it can clear unit.session. A tag from
                    # another session means the emitting session was force-released —
                    # its late SESSION_END must not stand in for this session's drain.
                    if is_control_message(audio_chunk, SESSION_END.kind):
                        chunk_session_id = getattr(audio_chunk, "session_id", None)
                        if session is not None and chunk_session_id in (None, session.session_id):
                            session.drained.set()
                            logger.debug(f"Pipeline {unit.index}: SESSION_END drained")
                        continue

                    if is_control_message(audio_chunk):
                        continue

                    if _should_discard_audio(unit, audio_chunk):
                        continue

                    audio_chunk = _to_audio_bytes(audio_chunk)

                    audio_batch = bytearray(audio_chunk)
                    while not pulled_held and len(audio_batch) < MAX_AUDIO_BATCH_BYTES:
                        try:
                            next_chunk = unit.output_queue.get_nowait()
                        except Empty:
                            break

                        if (
                            _is_pipeline_end(next_chunk)
                            or _is_audio_done(next_chunk)
                            or is_control_message(next_chunk, SESSION_END.kind)
                        ):
                            # Only stash if we still have a session; otherwise drop it.
                            if session is not None:
                                session.pending_output_item = next_chunk
                            break

                        if _should_discard_audio(unit, next_chunk):
                            continue

                        next_audio = _to_audio_bytes(next_chunk)
                        if len(audio_batch) + len(next_audio) > MAX_AUDIO_BATCH_BYTES:
                            if session is not None:
                                session.pending_output_item = next_chunk
                            break
                        audio_batch.extend(next_audio)

                    if not unit.response_playing.is_set():
                        unit.response_playing.set()
                        unit.should_listen.set()

                    if transport is not None and session_id:
                        await transport.send_audio_chunk(unit.service, session_id, bytes(audio_batch))
                except Empty:
                    pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline {unit.index} send loop error: {e}")
                await asyncio.sleep(0.1)

    return app
