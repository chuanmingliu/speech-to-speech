from __future__ import annotations

import logging

from openai.types.realtime import (
    RealtimeErrorEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler

logger = logging.getLogger(__name__)


class SessionHandler(RealtimeBaseHandler):
    """Owns session lifecycle: config updates and session.created events."""

    @staticmethod
    def _requested_turn_detection(session):
        if "audio" not in session.model_fields_set or session.audio is None:
            return None
        audio_input = session.audio.input
        if (
            audio_input is None
            or "input" not in session.audio.model_fields_set
            or "turn_detection" not in audio_input.model_fields_set
        ):
            return None
        return audio_input.turn_detection

    def handle_session_update(
        self,
        conn_id: str,
        event: SessionUpdateEvent,
    ) -> SessionUpdatedEvent | RealtimeErrorEvent | None:
        """Apply session config changes.

        Only ``RealtimeSessionCreateRequest`` sessions are accepted;
        ``RealtimeTranscriptionSessionCreateRequest`` sessions not yet supported.
        Incoming fields are deep-merged into the existing session so that
        partial updates preserve previously-set values.
        """
        s = event.session
        if s is None:
            return None

        if isinstance(s, RealtimeTranscriptionSessionCreateRequest):
            return self.make_error(
                message="Only 'realtime' session type is supported; transcription sessions are not.",
                _type="invalid_session_type",
            )

        st = self._state(conn_id)
        requested_turn_detection = self._requested_turn_detection(s)
        if requested_turn_detection is None and "audio" in s.model_fields_set:
            audio = s.audio
            audio_input = audio.input if audio is not None else None
            explicitly_null = (
                audio_input is not None
                and audio is not None
                and "input" in audio.model_fields_set
                and "turn_detection" in audio_input.model_fields_set
            )
        else:
            explicitly_null = False
        if explicitly_null:
            # Audio is streamed directly into the VAD pipeline.  Until the
            # pipeline has a native commit-delimited input seam, accepting null
            # would falsely claim that client commit owns the turn boundary.
            return self.make_error(
                message="Manual input turns (turn_detection: null) are not supported by this service.",
                _type="manual_input_turns_not_supported",
            )
        if requested_turn_detection is not None and requested_turn_detection.type != "server_vad":
            return self.make_error(
                message="Only server_vad turn detection is supported by this service.",
                _type="semantic_vad_not_supported",
            )

        if not self._service.tools_enabled:
            s.tools = []
            s.tool_choice = "none"

        model = getattr(s, "model", None)
        if model is not None:
            logger.info(f"Session model set to: {model}")

        cfg = st.runtime_config
        current = cfg.session
        if current is None:
            cfg.session = s
        else:
            cfg.apply_session_update(s)
        logger.info("Session configuration updated")
        return SessionUpdatedEvent(
            type="session.updated",
            event_id=self._next_event_id(),
            session=cfg.session,
        )

    def build_session_created(self, conn_id: str) -> SessionCreatedEvent:
        """Build a SessionCreatedEvent populated with the current config."""
        cfg = self._state(conn_id).runtime_config
        session = cfg.session
        return SessionCreatedEvent(
            type="session.created",
            event_id=self._next_event_id(),
            session=session,
        )
