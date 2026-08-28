#!/usr/bin/env python3
"""Generate and optionally run a 100 × 10 synthetic Realtime latency benchmark."""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import logging
import os
import shutil
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import websockets

logger = logging.getLogger("synthetic_latency_benchmark")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = REPO_ROOT / "benchmarks" / "synthetic-conversations.json"
DEFAULT_AUDIO_DIR = Path("/tmp/speech-to-speech-latency-audio")
SCHEMA_VERSION = 1
SAMPLE_RATE_HZ = 16000
BYTES_PER_SAMPLE = 2

TIMESTAMP_KEYS = (
    "turn_started",
    "input_audio_finished",
    "audio_send_finished",
    "speech_started",
    "first_asr_partial",
    "speech_stopped",
    "asr_final",
    "response_created",
    "first_assistant_text",
    "first_audio",
    "audio_done",
    "response_done",
    "error",
)

LATENCY_KEYS = (
    "connection_ms",
    "input_audio_ms",
    "audio_send_ms",
    "turn_start_to_speech_start_ms",
    "speech_start_to_first_asr_partial_ms",
    "input_audio_end_to_speech_stop_ms",
    "speech_stop_to_asr_final_ms",
    "asr_final_to_response_created_ms",
    "asr_final_to_first_assistant_text_ms",
    "response_created_to_first_audio_ms",
    "first_assistant_text_to_first_audio_ms",
    "speech_stop_to_first_audio_ms",
    "first_audio_to_audio_done_ms",
    "speech_stop_to_response_done_ms",
    "turn_total_ms",
)

EVENT_TIMESTAMP_KEYS = {
    "input_audio_buffer.speech_started": "speech_started",
    "input_audio_buffer.speech_stopped": "speech_stopped",
    "conversation.item.input_audio_transcription.delta": "first_asr_partial",
    "conversation.item.input_audio_transcription.completed": "asr_final",
    "response.created": "response_created",
    "response.audio_transcript.delta": "first_assistant_text",
    "response.output_audio_transcript.delta": "first_assistant_text",
    "response.audio_transcript.done": "first_assistant_text",
    "response.output_audio_transcript.done": "first_assistant_text",
    "response.audio.delta": "first_audio",
    "response.output_audio.delta": "first_audio",
    "response.audio.done": "audio_done",
    "response.output_audio.done": "audio_done",
    "response.done": "response_done",
    "error": "error",
}


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: str
    prompt: str

    def to_dict(self) -> dict[str, str]:
        return {"turn_id": self.turn_id, "prompt": self.prompt}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ConversationTurn:
        return cls(turn_id=str(value["turn_id"]), prompt=str(value["prompt"]))


@dataclass(frozen=True)
class ConversationCase:
    case_id: str
    family: str
    variant: int
    tags: tuple[str, ...]
    turns: tuple[ConversationTurn, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "variant": self.variant,
            "tags": list(self.tags),
            "turns": [turn.to_dict() for turn in self.turns],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ConversationCase:
        return cls(
            case_id=str(value["case_id"]),
            family=str(value["family"]),
            variant=int(value["variant"]),
            tags=tuple(str(tag) for tag in value.get("tags", [])),
            turns=tuple(ConversationTurn.from_dict(turn) for turn in value["turns"]),
        )


@dataclass(frozen=True)
class FamilySpec:
    slug: str
    tags: tuple[str, ...]
    templates: tuple[str, ...]
    variants: tuple[dict[str, Any], ...]


def _family_specs() -> tuple[FamilySpec, ...]:
    travel_destinations = (
        "Kyoto",
        "Lisbon",
        "Vancouver",
        "Singapore",
        "Reykjavik",
        "Melbourne",
        "Seoul",
        "Barcelona",
        "Marrakesh",
        "Auckland",
    )
    support_products = (
        "Orion wireless headphones",
        "Nova fitness watch",
        "Atlas home router",
        "Luma smart lamp",
        "Pico action camera",
        "Nimbus e-reader",
        "Echo portable speaker",
        "Terra robot vacuum",
        "Vela tablet",
        "Comet mechanical keyboard",
    )
    restaurants = (
        "Juniper Kitchen",
        "Harbor Table",
        "Copper Spoon",
        "Willow Garden",
        "North Star Cafe",
        "Saffron House",
        "Blue Finch Bistro",
        "Maple Room",
        "Cedar Grill",
        "Olive Courtyard",
    )
    return_items = (
        "winter jacket",
        "coffee grinder",
        "desk chair",
        "running shoes",
        "carry-on suitcase",
        "standing lamp",
        "yoga mat",
        "rice cooker",
        "office monitor",
        "camping tent",
    )
    study_topics = (
        "photosynthesis",
        "quadratic equations",
        "the French Revolution",
        "supply and demand",
        "cell division",
        "probability",
        "plate tectonics",
        "computer networks",
        "Shakespearean tragedy",
        "basic statistics",
    )
    projects = (
        "mobile app launch",
        "customer onboarding redesign",
        "warehouse inventory migration",
        "community fundraising event",
        "website accessibility audit",
        "quarterly sales campaign",
        "office relocation",
        "data quality cleanup",
        "podcast pilot season",
        "employee mentoring program",
    )
    smart_devices = (
        "living-room thermostat",
        "front-door camera",
        "bedroom smart bulb",
        "kitchen voice display",
        "garage door controller",
        "garden irrigation hub",
        "hallway motion sensor",
        "office air purifier",
        "nursery night light",
        "basement leak detector",
    )
    meeting_topics = (
        "design review",
        "vendor negotiation",
        "quarterly planning",
        "incident retrospective",
        "candidate interview",
        "budget review",
        "research kickoff",
        "customer feedback call",
        "security assessment",
        "release readiness check",
    )
    budget_goals = (
        "build an emergency fund",
        "save for a bicycle",
        "reduce dining expenses",
        "pay down a credit card",
        "plan a modest vacation",
        "replace a laptop",
        "prepare for moving costs",
        "start a course fund",
        "lower monthly subscriptions",
        "save for home repairs",
    )
    languages = (
        "Spanish",
        "French",
        "Japanese",
        "German",
        "Italian",
        "Korean",
        "Portuguese",
        "Mandarin",
        "Dutch",
        "Swedish",
    )

    return (
        FamilySpec(
            slug="travel-planning",
            tags=("context", "planning", "constraints"),
            templates=(
                "Help me plan a {days}-day trip to {primary}.",
                "For the {primary} trip, keep my total budget near {budget} dollars.",
                "On this {primary} visit, I care most about {interest}.",
                "Which neighborhood in {primary} best fits that preference?",
                "Give me a simple first-morning plan for {primary}.",
                "Make the second day in {primary} less rushed than the first.",
                "Suggest one rainy-day alternative while I am in {primary}.",
                "What local food should I prioritize in {primary}?",
                "Summarize my {primary} plan in three short points.",
                "What should I book first for the {primary} trip?",
            ),
            variants=tuple(
                {
                    "primary": destination,
                    "days": 3 + index % 5,
                    "budget": 900 + index * 175,
                    "interest": (
                        "quiet historic streets",
                        "modern architecture",
                        "local food markets",
                        "easy nature walks",
                        "museums",
                    )[index % 5],
                }
                for index, destination in enumerate(travel_destinations)
            ),
        ),
        FamilySpec(
            slug="device-support",
            tags=("context", "troubleshooting", "correction"),
            templates=(
                "My {primary} has started {issue}; help me diagnose it.",
                "The {primary} problem began after yesterday's software update.",
                "I already restarted the {primary}, but {issue} continues.",
                "Which {primary} setting should I inspect first?",
                "Explain that {primary} check without technical jargon.",
                "The {primary} indicator is blue, not red; revise the diagnosis.",
                "Give me one safe reset step for the {primary}.",
                "What data should I back up before resetting the {primary}?",
                "Summarize everything we tried on the {primary}.",
                "When should I escalate the {primary} case to support?",
            ),
            variants=tuple(
                {
                    "primary": product,
                    "issue": (
                        "disconnecting every few minutes",
                        "losing battery unusually quickly",
                        "showing a blank screen",
                        "failing to respond",
                        "overheating during normal use",
                    )[index % 5],
                }
                for index, product in enumerate(support_products)
            ),
        ),
        FamilySpec(
            slug="restaurant-booking",
            tags=("context", "booking", "numeric"),
            templates=(
                "Help me plan a dinner at {primary} in {city}.",
                "For {primary}, I need a table for {party} people on {date}.",
                "Ask for a {time} reservation at {primary}.",
                "One guest at {primary} needs vegetarian options.",
                "Another {primary} guest cannot eat peanuts.",
                "Move the preferred {primary} time thirty minutes later.",
                "What should I confirm with {primary} about accessibility?",
                "Draft a concise booking request for {primary}.",
                "Read back every detail for the {primary} reservation.",
                "What is still unconfirmed about dinner at {primary}?",
            ),
            variants=tuple(
                {
                    "primary": restaurant,
                    "city": (
                        "Boston",
                        "Austin",
                        "Seattle",
                        "Denver",
                        "Chicago",
                        "Portland",
                        "Atlanta",
                        "San Diego",
                        "Nashville",
                        "Philadelphia",
                    )[index],
                    "party": 2 + index % 6,
                    "date": f"October {12 + index}",
                    "time": f"{6 + index % 3}:30 PM",
                }
                for index, restaurant in enumerate(restaurants)
            ),
        ),
        FamilySpec(
            slug="order-return",
            tags=("context", "customer-support", "identifier"),
            templates=(
                "I need help returning a {primary} from order {order_id}.",
                "The {primary} arrived {reason}, and I noticed it immediately.",
                "Order {order_id} was delivered three days ago.",
                "For the {primary}, I prefer a replacement instead of a refund.",
                "What evidence should I provide for order {order_id}?",
                "The {primary} packaging is open but still available.",
                "Explain the likely next step for the {primary} return.",
                "Draft a short support message mentioning order {order_id}.",
                "Summarize the agreed return plan for the {primary}.",
                "What deadline should I verify for order {order_id}?",
            ),
            variants=tuple(
                {
                    "primary": item,
                    "order_id": f"ORD-{7300 + index * 37}",
                    "reason": (
                        "with a cracked part",
                        "in the wrong size",
                        "missing an accessory",
                        "with visible stains",
                        "and does not power on",
                    )[index % 5],
                }
                for index, item in enumerate(return_items)
            ),
        ),
        FamilySpec(
            slug="study-tutor",
            tags=("context", "education", "recall"),
            templates=(
                "Teach me the basics of {primary} at a {level} level.",
                "Give me a concrete example of {primary}.",
                "Explain the hardest term in {primary} more simply.",
                "Ask me one short comprehension question about {primary}.",
                "For that {primary} question, my answer is partly correct; explain why.",
                "Connect {primary} to something from everyday life.",
                "Give me a common misconception about {primary}.",
                "Create a two-step practice problem on {primary}.",
                "Show a concise solution to the {primary} practice problem.",
                "Summarize the {primary} lesson as five memory cues.",
            ),
            variants=tuple(
                {
                    "primary": topic,
                    "level": ("middle-school", "high-school", "beginner college")[index % 3],
                }
                for index, topic in enumerate(study_topics)
            ),
        ),
        FamilySpec(
            slug="project-planning",
            tags=("context", "project", "prioritization"),
            templates=(
                "Help me structure a {primary} with a {weeks}-week deadline.",
                "The {primary} team has {team_size} people with mixed experience.",
                "Name the first milestone for the {primary}.",
                "List the biggest schedule risk for the {primary}.",
                "Reduce the initial {primary} scope by about twenty percent.",
                "Assign clear roles for the smaller {primary} plan.",
                "Suggest one measurable success criterion for the {primary}.",
                "Create a brief weekly check-in agenda for the {primary}.",
                "Summarize the current {primary} plan and its tradeoffs.",
                "What decision about the {primary} must happen today?",
            ),
            variants=tuple(
                {
                    "primary": project,
                    "weeks": 4 + index,
                    "team_size": 3 + index % 6,
                }
                for index, project in enumerate(projects)
            ),
        ),
        FamilySpec(
            slug="smart-home",
            tags=("context", "troubleshooting", "safety"),
            templates=(
                "My {primary} is {problem}; walk me through safe checks.",
                "The {primary} still appears online in its mobile app.",
                "Power cycling did not fix the {primary}.",
                "Which network detail matters most for the {primary}?",
                "Explain how to verify that detail on the {primary}.",
                "The {primary} uses a guest network; update your advice.",
                "Give me a low-risk next step for the {primary}.",
                "What symptom would make the {primary} unsafe to use?",
                "Summarize the {primary} troubleshooting sequence.",
                "What information should I give the {primary} manufacturer?",
            ),
            variants=tuple(
                {
                    "primary": device,
                    "problem": (
                        "dropping offline",
                        "sending delayed alerts",
                        "ignoring schedules",
                        "reporting the wrong status",
                        "restarting unexpectedly",
                    )[index % 5],
                }
                for index, device in enumerate(smart_devices)
            ),
        ),
        FamilySpec(
            slug="meeting-scheduling",
            tags=("context", "scheduling", "correction"),
            templates=(
                "Help me organize a {primary} for {day}.",
                "The {primary} needs {attendees} attendees and forty-five minutes.",
                "Propose a {time} start for the {primary}.",
                "For the {primary}, include ten minutes for questions.",
                "Move the {primary} from {day} to the following business day.",
                "One {primary} attendee is remote in another time zone.",
                "Draft a concise agenda for the {primary}.",
                "Write a short invitation description for the {primary}.",
                "Read back the updated {primary} schedule.",
                "What scheduling risk remains for the {primary}?",
            ),
            variants=tuple(
                {
                    "primary": topic,
                    "day": (
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                    )[index % 5],
                    "attendees": 3 + index,
                    "time": f"{9 + index % 5}:00 AM",
                }
                for index, topic in enumerate(meeting_topics)
            ),
        ),
        FamilySpec(
            slug="budget-coaching",
            tags=("context", "numeric", "planning"),
            templates=(
                "Help me {primary} over the next {months} months.",
                "For my goal to {primary}, I can set aside {amount} dollars monthly.",
                "Name the first expense category to review so I can {primary}.",
                "Give me a weekly spending limit that supports my plan to {primary}.",
                "I forgot an annual insurance bill; revise the plan to {primary}.",
                "Keep a small entertainment allowance while I {primary}.",
                "Suggest one automatic transfer for the goal to {primary}.",
                "What progress number should I track while I {primary}?",
                "Summarize the monthly plan to {primary}.",
                "What is the biggest assumption in the plan to {primary}?",
            ),
            variants=tuple(
                {
                    "primary": goal,
                    "months": 4 + index,
                    "amount": 120 + index * 35,
                }
                for index, goal in enumerate(budget_goals)
            ),
        ),
        FamilySpec(
            slug="language-practice",
            tags=("context", "language", "role-play"),
            templates=(
                "Practice a beginner {primary} conversation about {situation} with me.",
                "In {primary}, greet me for the {situation} role-play.",
                "Translate my reply into natural {primary} for the {situation}.",
                "Correct one likely pronunciation issue in that {primary} sentence.",
                "Ask me a follow-up question in {primary} about {situation}.",
                "Make the next {primary} question slightly more difficult.",
                "Explain one polite phrase used in {primary} during {situation}.",
                "Now continue the {situation} role-play only in {primary}.",
                "Summarize my two biggest {primary} mistakes from {situation}.",
                "End the {situation} practice with a short {primary} farewell.",
            ),
            variants=tuple(
                {
                    "primary": language,
                    "situation": (
                        "ordering breakfast",
                        "checking into a hotel",
                        "asking for directions",
                        "buying a train ticket",
                        "meeting a colleague",
                    )[index % 5],
                }
                for index, language in enumerate(languages)
            ),
        ),
    )


def build_cases() -> list[ConversationCase]:
    cases: list[ConversationCase] = []
    for family in _family_specs():
        for variant_index, variables in enumerate(family.variants, start=1):
            case_id = f"{family.slug}-{variant_index:02d}"
            turns = tuple(
                ConversationTurn(
                    turn_id=f"{case_id}-turn-{turn_index:02d}",
                    prompt=template.format(**variables),
                )
                for turn_index, template in enumerate(family.templates, start=1)
            )
            cases.append(
                ConversationCase(
                    case_id=case_id,
                    family=family.slug,
                    variant=variant_index,
                    tags=family.tags,
                    turns=turns,
                )
            )
    validate_cases(cases)
    return cases


def validate_cases(cases: Sequence[ConversationCase]) -> dict[str, int]:
    if len(cases) != 100:
        raise ValueError(f"Expected 100 conversation cases, got {len(cases)}.")
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Conversation case IDs must be unique.")
    minimum_turns = min((len(case.turns) for case in cases), default=0)
    if minimum_turns < 10:
        raise ValueError(f"Every conversation case needs at least 10 turns; minimum is {minimum_turns}.")
    turn_ids = [turn.turn_id for case in cases for turn in case.turns]
    if len(set(turn_ids)) != len(turn_ids):
        raise ValueError("Conversation turn IDs must be globally unique.")
    if any(not turn.prompt.strip() for case in cases for turn in case.turns):
        raise ValueError("Conversation prompts must not be empty.")
    family_count = len({case.family for case in cases})
    if family_count != 10:
        raise ValueError(f"Expected 10 scenario families, got {family_count}.")
    return {
        "case_count": len(cases),
        "turn_count": len(turn_ids),
        "minimum_turns_per_case": minimum_turns,
        "family_count": family_count,
    }


def write_corpus(path: Path, cases: Sequence[ConversationCase]) -> None:
    stats = validate_cases(cases)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": SCHEMA_VERSION,
        **stats,
        "cases": [case.to_dict() for case in cases],
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")


def load_corpus(path: Path) -> list[ConversationCase]:
    document = json.loads(path.read_text())
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported corpus schema {document.get('schema_version')!r}; expected {SCHEMA_VERSION}."
        )
    cases = [ConversationCase.from_dict(case) for case in document.get("cases", [])]
    validate_cases(cases)
    return cases


def _first(mapping: dict[str, float], key: str, value: float) -> None:
    mapping.setdefault(key, value)


def _milliseconds(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start) * 1000.0, 3)


class TurnLatencyRecorder:
    """Collect receive-time protocol milestones for one spoken turn."""

    def __init__(
        self,
        *,
        case_id: str,
        family: str,
        turn_index: int,
        turn_id: str,
        prompt: str,
        turn_started_s: float,
        connection_ms: float,
    ) -> None:
        self.case_id = case_id
        self.family = family
        self.turn_index = turn_index
        self.turn_id = turn_id
        self.prompt = prompt
        self.connection_ms = connection_ms
        self.timestamps: dict[str, float] = {"turn_started": turn_started_s}
        self.input_transcript = ""
        self._assistant_deltas: list[str] = []
        self._assistant_done_parts: list[str] = []
        self.response_id: str | None = None
        self.response_status: str | None = None
        self.error_code: str | None = None
        self.error: str | None = None
        self.event_count = 0
        self.audio_chunks = 0
        self.audio_bytes = 0

    def mark_input_audio_finished(self, at_s: float) -> None:
        _first(self.timestamps, "input_audio_finished", at_s)

    def mark_audio_send_finished(self, at_s: float) -> None:
        _first(self.timestamps, "audio_send_finished", at_s)

    def mark_error(self, code: str, message: str, at_s: float | None = None) -> None:
        if self.error is None:
            self.error_code = code
            self.error = message
        _first(self.timestamps, "error", at_s if at_s is not None else time.perf_counter())

    def observe(self, event: dict[str, Any], received_at_s: float) -> None:
        self.event_count += 1
        event_type = str(event.get("type") or "")
        timestamp_key = EVENT_TIMESTAMP_KEYS.get(event_type)
        if timestamp_key:
            _first(self.timestamps, timestamp_key, received_at_s)

        if event_type == "conversation.item.input_audio_transcription.completed":
            self.input_transcript = str(event.get("transcript") or "")
        elif event_type in {
            "response.audio_transcript.delta",
            "response.output_audio_transcript.delta",
        }:
            delta = str(event.get("delta") or "")
            if delta:
                self._assistant_deltas.append(delta)
        elif event_type in {
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
        }:
            transcript = str(event.get("transcript") or "").strip()
            if transcript:
                self._assistant_done_parts.append(transcript)
            self.response_id = str(event.get("response_id") or self.response_id or "") or None
        elif event_type in {"response.audio.delta", "response.output_audio.delta"}:
            self.audio_chunks += 1
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                try:
                    self.audio_bytes += len(base64.b64decode(delta))
                except (ValueError, TypeError):
                    self.mark_error("invalid_audio_delta", "Audio delta was not valid base64.", received_at_s)
        elif event_type == "response.created":
            response = event.get("response") or {}
            self.response_id = str(response.get("id") or self.response_id or "") or None
        elif event_type == "response.done":
            response = event.get("response") or {}
            self.response_status = str(response.get("status") or "completed")
            self.response_id = str(response.get("id") or self.response_id or "") or None
        elif event_type == "error":
            error = event.get("error") or {}
            self.mark_error(
                str(error.get("code") or "realtime_error"),
                str(error.get("message") or "Realtime error"),
                received_at_s,
            )
        elif event_type == "_transport_error":
            self.mark_error(
                str(event.get("code") or "transport_error"),
                str(event.get("message") or "WebSocket transport error"),
                received_at_s,
            )

    def to_record(self) -> dict[str, Any]:
        ts = self.timestamps
        assistant_transcript = (
            " ".join(self._assistant_done_parts)
            if self._assistant_done_parts
            else "".join(self._assistant_deltas)
        )
        completed_at = ts.get("response_done") or ts.get("error") or ts.get("audio_send_finished")
        latency = {
            "connection_ms": round(self.connection_ms, 3),
            "input_audio_ms": _milliseconds(ts.get("turn_started"), ts.get("input_audio_finished")),
            "audio_send_ms": _milliseconds(ts.get("turn_started"), ts.get("audio_send_finished")),
            "turn_start_to_speech_start_ms": _milliseconds(ts.get("turn_started"), ts.get("speech_started")),
            "speech_start_to_first_asr_partial_ms": _milliseconds(
                ts.get("speech_started"), ts.get("first_asr_partial")
            ),
            "input_audio_end_to_speech_stop_ms": _milliseconds(
                ts.get("input_audio_finished"), ts.get("speech_stopped")
            ),
            "speech_stop_to_asr_final_ms": _milliseconds(ts.get("speech_stopped"), ts.get("asr_final")),
            "asr_final_to_response_created_ms": _milliseconds(
                ts.get("asr_final"), ts.get("response_created")
            ),
            "asr_final_to_first_assistant_text_ms": _milliseconds(
                ts.get("asr_final"), ts.get("first_assistant_text")
            ),
            "response_created_to_first_audio_ms": _milliseconds(
                ts.get("response_created"), ts.get("first_audio")
            ),
            "first_assistant_text_to_first_audio_ms": _milliseconds(
                ts.get("first_assistant_text"), ts.get("first_audio")
            ),
            "speech_stop_to_first_audio_ms": _milliseconds(
                ts.get("speech_stopped"), ts.get("first_audio")
            ),
            "first_audio_to_audio_done_ms": _milliseconds(ts.get("first_audio"), ts.get("audio_done")),
            "speech_stop_to_response_done_ms": _milliseconds(
                ts.get("speech_stopped"), ts.get("response_done")
            ),
            "turn_total_ms": _milliseconds(ts.get("turn_started"), completed_at),
        }
        relative_timestamps = {
            f"{key}_ms": _milliseconds(ts.get("turn_started"), ts.get(key))
            for key in TIMESTAMP_KEYS
        }
        success = (
            self.error is None
            and "response_done" in ts
            and self.response_status in (None, "completed")
        )
        return {
            "case_id": self.case_id,
            "family": self.family,
            "turn_index": self.turn_index,
            "turn_id": self.turn_id,
            "prompt": self.prompt,
            "success": success,
            "error_code": self.error_code,
            "error": self.error,
            "response_id": self.response_id,
            "response_status": self.response_status,
            "input_transcript": self.input_transcript,
            "assistant_transcript": assistant_transcript,
            "event_count": self.event_count,
            "audio_chunks": self.audio_chunks,
            "audio_bytes": self.audio_bytes,
            "timestamps_from_turn_start": relative_timestamps,
            "latency_ms": latency,
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile without values.")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _metric_stats(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p90": round(_percentile(values, 0.90), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "p99": round(_percentile(values, 0.99), 3),
        "max": round(max(values), 3),
    }


def summarize_records(
    records: Sequence[dict[str, Any]],
    *,
    planned_cases: int,
    planned_turns: int,
) -> dict[str, Any]:
    successful = sum(bool(record.get("success")) for record in records)
    errors = Counter(
        str(record.get("error_code") or "unknown_error")
        for record in records
        if not record.get("success")
    )
    metric_names = sorted(
        {
            metric_name
            for record in records
            for metric_name in (record.get("latency_ms") or {})
        }
    )
    metrics: dict[str, dict[str, float | int]] = {}
    for metric_name in metric_names:
        values = [
            float(value)
            for record in records
            if (value := (record.get("latency_ms") or {}).get(metric_name)) is not None
        ]
        if values:
            metrics[metric_name] = _metric_stats(values)
    return {
        "planned_cases": planned_cases,
        "planned_turns": planned_turns,
        "records": len(records),
        "successful_turns": successful,
        "failed_turns": len(records) - successful,
        "success_rate": round(successful / len(records), 6) if records else 0.0,
        "errors_by_code": dict(sorted(errors.items())),
        "metrics_ms": metrics,
    }


def _selected_cases(
    cases: Sequence[ConversationCase],
    *,
    case_limit: int | None,
    turn_limit: int | None,
) -> list[ConversationCase]:
    selected = list(cases[:case_limit] if case_limit is not None else cases)
    if turn_limit is not None:
        selected = [replace(case, turns=case.turns[:turn_limit]) for case in selected]
    return selected


def _audio_cache_path(audio_dir: Path, prompt: str, voice: str | None) -> Path:
    digest = hashlib.sha256(
        json.dumps({"prompt": prompt, "voice": voice}, sort_keys=True).encode()
    ).hexdigest()[:24]
    return audio_dir / f"{digest}.wav"


def _synthesize_prompt(prompt: str, output_path: Path, voice: str | None) -> None:
    command = [
        "say",
        prompt,
        "-o",
        str(output_path),
        "--file-format=WAVE",
        "--data-format=LEI16@16000",
    ]
    if voice:
        command.extend(["--voice", voice])
    subprocess.run(command, check=True, capture_output=True)


def prepare_audio(
    cases: Sequence[ConversationCase],
    *,
    audio_dir: Path,
    voice: str | None,
) -> dict[str, Path]:
    if shutil.which("say") is None:
        raise RuntimeError("Audio preparation requires the macOS `say` command.")
    audio_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    turns = [turn for case in cases for turn in case.turns]
    synthesized = 0
    for index, turn in enumerate(turns, start=1):
        path = _audio_cache_path(audio_dir, turn.prompt, voice)
        if not path.exists():
            _synthesize_prompt(turn.prompt, path, voice)
            synthesized += 1
        paths[turn.turn_id] = path
        if index % 100 == 0 or index == len(turns):
            logger.info(
                "Audio preparation: %d/%d turns ready (%d newly synthesized)",
                index,
                len(turns),
                synthesized,
            )
    return paths


def _load_pcm16(path: Path) -> bytes:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_rate != SAMPLE_RATE_HZ:
        mono = resample_poly(mono, SAMPLE_RATE_HZ, source_rate)
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


async def _stream_audio(
    websocket: Any,
    pcm: bytes,
    recorder: TurnLatencyRecorder,
    *,
    chunk_ms: int,
    trailing_silence_ms: int,
) -> None:
    chunk_bytes = SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * chunk_ms // 1000
    for start in range(0, len(pcm), chunk_bytes):
        chunk = pcm[start : start + chunk_bytes].ljust(chunk_bytes, b"\0")
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )
        await asyncio.sleep(chunk_ms / 1000.0)
    recorder.mark_input_audio_finished(time.perf_counter())
    silence = base64.b64encode(b"\0" * chunk_bytes).decode("ascii")
    for _ in range(max(1, trailing_silence_ms // chunk_ms)):
        await websocket.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": silence})
        )
        await asyncio.sleep(chunk_ms / 1000.0)
    recorder.mark_audio_send_finished(time.perf_counter())


async def _receive_events(websocket: Any, event_queue: asyncio.Queue[tuple[float, dict[str, Any]]]) -> None:
    try:
        while True:
            raw = await websocket.recv()
            received_at = time.perf_counter()
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise ValueError("Realtime event must be a JSON object.")
            await event_queue.put((received_at, event))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await event_queue.put(
            (
                time.perf_counter(),
                {
                    "type": "_transport_error",
                    "code": type(exc).__name__,
                    "message": str(exc),
                },
            )
        )


async def _run_turn(
    websocket: Any,
    event_queue: asyncio.Queue[tuple[float, dict[str, Any]]],
    *,
    case: ConversationCase,
    turn: ConversationTurn,
    turn_index: int,
    audio_path: Path,
    connection_ms: float,
    chunk_ms: int,
    trailing_silence_ms: int,
    timeout_s: float,
) -> dict[str, Any]:
    # Decode cached audio before starting the measured turn. Corpus preparation
    # and local disk/codec work must not inflate provider latency.
    pcm = await asyncio.to_thread(_load_pcm16, audio_path)
    recorder = TurnLatencyRecorder(
        case_id=case.case_id,
        family=case.family,
        turn_index=turn_index,
        turn_id=turn.turn_id,
        prompt=turn.prompt,
        turn_started_s=time.perf_counter(),
        connection_ms=connection_ms,
    )
    sender = asyncio.create_task(
        _stream_audio(
            websocket,
            pcm,
            recorder,
            chunk_ms=chunk_ms,
            trailing_silence_ms=trailing_silence_ms,
        )
    )
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                recorder.mark_error("response_timeout", f"No response.done within {timeout_s:.1f}s.")
                break
            try:
                received_at, event = await asyncio.wait_for(
                    event_queue.get(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                recorder.mark_error("response_timeout", f"No response.done within {timeout_s:.1f}s.")
                break
            recorder.observe(event, received_at)
            if event.get("type") in {"response.done", "error", "_transport_error"}:
                break
        try:
            await asyncio.wait_for(sender, timeout=max(2.0, trailing_silence_ms / 1000.0 + 1.0))
        except Exception as exc:
            recorder.mark_error("audio_send_error", str(exc))
    finally:
        if not sender.done():
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass
    return recorder.to_record()


async def _run_case(
    case: ConversationCase,
    *,
    url: str,
    headers: list[tuple[str, str]],
    audio_paths: dict[str, Path],
    chunk_ms: int,
    trailing_silence_ms: int,
    timeout_s: float,
    turn_gap_ms: int,
) -> list[dict[str, Any]]:
    connect_started = time.perf_counter()
    records: list[dict[str, Any]] = []
    async with websockets.connect(
        url,
        max_size=2**24,
        additional_headers=headers or None,
    ) as websocket:
        first_raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
        connection_ms = (time.perf_counter() - connect_started) * 1000.0
        first = json.loads(first_raw)
        if first.get("type") != "session.created":
            recorder = TurnLatencyRecorder(
                case_id=case.case_id,
                family=case.family,
                turn_index=1,
                turn_id=case.turns[0].turn_id,
                prompt=case.turns[0].prompt,
                turn_started_s=time.perf_counter(),
                connection_ms=connection_ms,
            )
            error = first.get("error") or {}
            recorder.mark_error(
                str(error.get("code") or "session_start_failed"),
                str(error.get("message") or f"Unexpected first event {first.get('type')!r}."),
            )
            return [recorder.to_record()]

        event_queue: asyncio.Queue[tuple[float, dict[str, Any]]] = asyncio.Queue()
        receiver = asyncio.create_task(_receive_events(websocket, event_queue))
        try:
            for turn_index, turn in enumerate(case.turns, start=1):
                record = await _run_turn(
                    websocket,
                    event_queue,
                    case=case,
                    turn=turn,
                    turn_index=turn_index,
                    audio_path=audio_paths[turn.turn_id],
                    connection_ms=connection_ms,
                    chunk_ms=chunk_ms,
                    trailing_silence_ms=trailing_silence_ms,
                    timeout_s=timeout_s,
                )
                records.append(record)
                first_audio_ms = record["latency_ms"].get("speech_stop_to_first_audio_ms")
                logger.info(
                    "Case %s turn %d/%d: %s, speech-stop→first-audio=%s",
                    case.case_id,
                    turn_index,
                    len(case.turns),
                    "ok" if record["success"] else f"failed ({record['error_code']})",
                    f"{first_audio_ms:.1f}ms" if first_audio_ms is not None else "n/a",
                )
                if not record["success"]:
                    break
                if turn_gap_ms > 0:
                    await asyncio.sleep(turn_gap_ms / 1000.0)
        finally:
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass
    return records


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat = {
        key: value
        for key, value in record.items()
        if key not in {"timestamps_from_turn_start", "latency_ms"}
    }
    flat.update(record.get("timestamps_from_turn_start") or {})
    flat.update(record.get("latency_ms") or {})
    return flat


def _write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    flattened = [_flatten_record(record) for record in records]
    fieldnames = sorted({key for record in flattened for key in record})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def _write_summary_markdown(path: Path, summary: dict[str, Any], config: dict[str, Any]) -> None:
    lines = [
        "# Synthetic latency benchmark summary",
        "",
        f"- URL: `{config['url']}`",
    ]
    if config.get("stopped_early"):
        lines.extend(
            [
                "- Run state: stopped early at user request",
                f"- Fully checkpointed cases: {config.get('completed_cases', 0)}",
            ]
        )
    lines.extend(
        [
        f"- Cases planned: {summary['planned_cases']}",
        f"- Turns planned: {summary['planned_turns']}",
        f"- Turns recorded: {summary['records']}",
        f"- Successful turns: {summary['successful_turns']}",
        f"- Failed turns: {summary['failed_turns']}",
        f"- Success rate: {summary['success_rate'] * 100:.2f}%",
        "",
        "## Latency metrics",
        ]
    )
    for name, values in summary["metrics_ms"].items():
        lines.append(
            f"- `{name}`: p50={values['p50']:.3f} ms, p95={values['p95']:.3f} ms, "
            f"p99={values['p99']:.3f} ms, mean={values['mean']:.3f} ms, n={values['count']}"
        )
    if summary["errors_by_code"]:
        lines.extend(("", "## Errors"))
        for code, count in summary["errors_by_code"].items():
            lines.append(f"- `{code}`: {count}")
    path.write_text("\n".join(lines) + "\n")


def write_results(
    results_dir: Path,
    records: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    planned_cases: int,
    planned_turns: int,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "turns.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    _write_csv(results_dir / "turns.csv", records)
    summary = summarize_records(
        records,
        planned_cases=planned_cases,
        planned_turns=planned_turns,
    )
    summary_document = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        **summary,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary_document, ensure_ascii=False, indent=2) + "\n"
    )
    _write_summary_markdown(results_dir / "summary.md", summary, config)
    return summary_document


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cases = _selected_cases(
        load_corpus(args.cases),
        case_limit=args.case_limit,
        turn_limit=args.turn_limit,
    )
    audio_paths = await asyncio.to_thread(
        prepare_audio,
        cases,
        audio_dir=args.audio_dir,
        voice=args.voice,
    )
    token = os.getenv(args.bearer_token_env) if args.bearer_token_env else None
    headers = [("Authorization", f"Bearer {token}")] if token else []
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_limited(case: ConversationCase) -> list[dict[str, Any]]:
        async with semaphore:
            try:
                logger.info("Starting case %s (%d turns)", case.case_id, len(case.turns))
                case_records = await _run_case(
                    case,
                    url=args.url,
                    headers=headers,
                    audio_paths=audio_paths,
                    chunk_ms=args.chunk_ms,
                    trailing_silence_ms=args.trailing_silence_ms,
                    timeout_s=args.turn_timeout,
                    turn_gap_ms=args.turn_gap_ms,
                )
                logger.info(
                    "Finished case %s: %d/%d turns successful",
                    case.case_id,
                    sum(bool(record["success"]) for record in case_records),
                    len(case.turns),
                )
                return case_records
            except Exception as exc:
                recorder = TurnLatencyRecorder(
                    case_id=case.case_id,
                    family=case.family,
                    turn_index=1,
                    turn_id=case.turns[0].turn_id,
                    prompt=case.turns[0].prompt,
                    turn_started_s=time.perf_counter(),
                    connection_ms=0.0,
                )
                recorder.mark_error(type(exc).__name__, str(exc))
                return [recorder.to_record()]
            finally:
                # The backend releases a pipeline only after SESSION_END has
                # propagated through every handler. Hold this concurrency slot
                # briefly so the next case doesn't race that release.
                if args.case_gap_ms > 0:
                    await asyncio.sleep(args.case_gap_ms / 1000.0)

    records: list[dict[str, Any]] = []
    args.results_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.results_dir / "turns.partial.jsonl"
    with partial_path.open("w") as partial:
        for completed in asyncio.as_completed([run_limited(case) for case in cases]):
            case_records = await completed
            records.extend(case_records)
            for record in case_records:
                partial.write(json.dumps(record, ensure_ascii=False) + "\n")
            partial.flush()

    config = {
        "url": args.url,
        "corpus": str(args.cases),
        "audio_dir": str(args.audio_dir),
        "case_limit": args.case_limit,
        "turn_limit": args.turn_limit,
        "concurrency": args.concurrency,
        "chunk_ms": args.chunk_ms,
        "trailing_silence_ms": args.trailing_silence_ms,
        "turn_timeout_s": args.turn_timeout,
        "turn_gap_ms": args.turn_gap_ms,
        "case_gap_ms": args.case_gap_ms,
        "voice": args.voice,
        "bearer_token_env": args.bearer_token_env,
    }
    summary = write_results(
        args.results_dir,
        records,
        config=config,
        planned_cases=len(cases),
        planned_turns=sum(len(case.turns) for case in cases),
    )
    partial_path.unlink(missing_ok=True)
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Write the deterministic 100-case corpus.")
    generate.add_argument("--output", type=Path, default=DEFAULT_CORPUS_PATH)

    validate = subparsers.add_parser("validate", help="Validate a corpus without provider calls.")
    validate.add_argument("--cases", type=Path, default=DEFAULT_CORPUS_PATH)

    prepare = subparsers.add_parser("prepare", help="Synthesize and cache local microphone WAVs only.")
    prepare.add_argument("--cases", type=Path, default=DEFAULT_CORPUS_PATH)
    prepare.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    prepare.add_argument("--case-limit", type=_positive_int)
    prepare.add_argument("--turn-limit", type=_positive_int)
    prepare.add_argument("--voice")

    run = subparsers.add_parser("run", help="Explicitly run selected cases against a live backend.")
    run.add_argument("--cases", type=Path, default=DEFAULT_CORPUS_PATH)
    run.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime")
    run.add_argument("--results-dir", type=Path, required=True)
    run.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    run.add_argument("--case-limit", type=_positive_int)
    run.add_argument("--turn-limit", type=_positive_int)
    run.add_argument("--concurrency", type=_positive_int, default=1)
    run.add_argument("--chunk-ms", type=_positive_int, default=20)
    run.add_argument("--trailing-silence-ms", type=_positive_int, default=1000)
    run.add_argument("--turn-timeout", type=float, default=30.0)
    run.add_argument("--turn-gap-ms", type=int, default=100)
    run.add_argument("--case-gap-ms", type=int, default=300)
    run.add_argument("--voice")
    run.add_argument(
        "--bearer-token-env",
        default="REALTIME_API_KEY",
        help="Environment variable containing an optional bearer token; empty disables auth.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        cases = build_cases()
        write_corpus(args.output, cases)
        stats = validate_cases(cases)
        print(
            f"Wrote {stats['case_count']} cases and {stats['turn_count']} turns to {args.output}"
        )
        return 0
    if args.command == "validate":
        stats = validate_cases(load_corpus(args.cases))
        print(
            f"Valid corpus: {stats['case_count']} cases, {stats['turn_count']} turns, "
            f"{stats['family_count']} families"
        )
        return 0
    if args.command == "prepare":
        cases = _selected_cases(
            load_corpus(args.cases),
            case_limit=args.case_limit,
            turn_limit=args.turn_limit,
        )
        paths = prepare_audio(cases, audio_dir=args.audio_dir, voice=args.voice)
        print(f"Prepared {len(paths)} turn WAVs under {args.audio_dir}")
        return 0
    if args.command == "run":
        summary = asyncio.run(run_benchmark(args))
        print(
            f"Recorded {summary['records']} turns with "
            f"{summary['success_rate'] * 100:.2f}% success under {args.results_dir}"
        )
        return 0 if summary["failed_turns"] == 0 else 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
