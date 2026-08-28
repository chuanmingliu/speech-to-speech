import asyncio
import base64
import json
import time
from collections import Counter

import pytest
import scripts.synthetic_latency_benchmark as benchmark
from scripts.synthetic_latency_benchmark import (
    TurnLatencyRecorder,
    build_cases,
    load_corpus,
    summarize_records,
    validate_cases,
    write_corpus,
    write_results,
)


def test_build_cases_creates_100_coherent_ten_turn_conversations():
    cases = build_cases()

    assert len(cases) == 100
    assert sum(len(case.turns) for case in cases) == 1000
    assert len({case.case_id for case in cases}) == 100
    assert set(Counter(case.family for case in cases).values()) == {10}
    assert all(len(case.turns) >= 10 for case in cases)
    assert all(len({turn.turn_id for turn in case.turns}) == len(case.turns) for case in cases)
    assert all(turn.prompt.strip() for case in cases for turn in case.turns)
    assert len({turn.prompt for case in cases for turn in case.turns}) >= 900


def test_corpus_round_trip_is_deterministic(tmp_path):
    path = tmp_path / "cases.json"
    cases = build_cases()

    write_corpus(path, cases)
    first = path.read_text()
    loaded = load_corpus(path)
    write_corpus(path, loaded)

    assert path.read_text() == first
    assert validate_cases(loaded) == {
        "case_count": 100,
        "turn_count": 1000,
        "minimum_turns_per_case": 10,
        "family_count": 10,
    }
    document = json.loads(first)
    assert document["schema_version"] == 1
    assert document["case_count"] == 100


def test_latency_recorder_captures_stage_boundaries_and_all_sentence_transcripts():
    recorder = TurnLatencyRecorder(
        case_id="travel-01",
        family="travel",
        turn_index=1,
        turn_id="travel-01-turn-01",
        prompt="Plan a trip.",
        turn_started_s=100.0,
        connection_ms=42.0,
    )
    recorder.mark_input_audio_finished(100.8)
    recorder.mark_audio_send_finished(101.4)
    recorder.observe({"type": "input_audio_buffer.speech_started"}, 100.3)
    recorder.observe(
        {"type": "conversation.item.input_audio_transcription.delta", "delta": "Plan"},
        100.5,
    )
    recorder.observe({"type": "input_audio_buffer.speech_stopped"}, 101.0)
    recorder.observe(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Plan a trip.",
        },
        101.12,
    )
    recorder.observe({"type": "response.created", "response": {"id": "resp_1"}}, 101.2)
    recorder.observe(
        {
            "type": "response.output_audio_transcript.done",
            "response_id": "resp_1",
            "transcript": "First sentence.",
        },
        101.5,
    )
    recorder.observe(
        {
            "type": "response.output_audio_transcript.done",
            "response_id": "resp_1",
            "transcript": "Second sentence.",
        },
        101.7,
    )
    audio = base64.b64encode(b"\x00\x01\x02\x03").decode()
    recorder.observe({"type": "response.output_audio.delta", "delta": audio}, 101.72)
    recorder.observe({"type": "response.output_audio.delta", "delta": audio}, 101.8)
    recorder.observe({"type": "response.output_audio.done"}, 102.0)
    recorder.observe(
        {"type": "response.done", "response": {"status": "completed"}},
        102.1,
    )

    record = recorder.to_record()

    assert record["success"] is True
    assert record["input_transcript"] == "Plan a trip."
    assert record["assistant_transcript"] == "First sentence. Second sentence."
    assert record["response_id"] == "resp_1"
    assert record["audio_chunks"] == 2
    assert record["audio_bytes"] == 8
    assert record["latency_ms"]["speech_stop_to_asr_final_ms"] == pytest.approx(120)
    assert record["latency_ms"]["asr_final_to_first_assistant_text_ms"] == pytest.approx(380)
    assert record["latency_ms"]["first_assistant_text_to_first_audio_ms"] == pytest.approx(220)
    assert record["latency_ms"]["speech_stop_to_first_audio_ms"] == pytest.approx(720)
    assert record["latency_ms"]["turn_total_ms"] == pytest.approx(2100)


def test_latency_recorder_preserves_missing_metrics_and_error():
    recorder = TurnLatencyRecorder(
        case_id="support-01",
        family="support",
        turn_index=2,
        turn_id="support-01-turn-02",
        prompt="It still fails.",
        turn_started_s=50.0,
        connection_ms=10.0,
    )
    recorder.observe(
        {"type": "error", "error": {"code": "provider_error", "message": "upstream failed"}},
        50.4,
    )
    recorder.mark_error("audio_send_error", "secondary failure", 50.5)

    record = recorder.to_record()

    assert record["success"] is False
    assert record["error_code"] == "provider_error"
    assert record["error"] == "upstream failed"
    assert record["latency_ms"]["speech_stop_to_first_audio_ms"] is None


def test_summary_reports_latency_percentiles_and_failures():
    records = [
        {
            "success": True,
            "error_code": None,
            "latency_ms": {"speech_stop_to_first_audio_ms": value},
        }
        for value in (100.0, 200.0, 300.0, 400.0)
    ]
    records.append(
        {
            "success": False,
            "error_code": "timeout",
            "latency_ms": {"speech_stop_to_first_audio_ms": None},
        }
    )

    summary = summarize_records(records, planned_cases=1, planned_turns=5)

    assert summary["records"] == 5
    assert summary["successful_turns"] == 4
    assert summary["failed_turns"] == 1
    assert summary["errors_by_code"] == {"timeout": 1}
    metric = summary["metrics_ms"]["speech_stop_to_first_audio_ms"]
    assert metric["count"] == 4
    assert metric["mean"] == 250.0
    assert metric["p50"] == 250.0
    assert metric["p95"] == pytest.approx(385.0)


def test_result_writer_emits_jsonl_csv_and_summaries(tmp_path):
    record = {
        "case_id": "travel-01",
        "family": "travel",
        "turn_index": 1,
        "turn_id": "travel-01-turn-01",
        "prompt": "Plan a trip.",
        "success": True,
        "error_code": None,
        "error": None,
        "response_id": "resp_1",
        "response_status": "completed",
        "input_transcript": "Plan a trip.",
        "assistant_transcript": "Sure.",
        "event_count": 8,
        "audio_chunks": 2,
        "audio_bytes": 1024,
        "timestamps_from_turn_start": {"speech_stopped_ms": 1000.0},
        "latency_ms": {"speech_stop_to_first_audio_ms": 750.0},
    }

    summary = write_results(
        tmp_path,
        [record],
        config={"url": "ws://localhost/v1/realtime"},
        planned_cases=1,
        planned_turns=1,
    )

    assert summary["successful_turns"] == 1
    assert json.loads((tmp_path / "turns.jsonl").read_text())["turn_id"] == record["turn_id"]
    assert "speech_stop_to_first_audio_ms" in (tmp_path / "turns.csv").read_text()
    assert json.loads((tmp_path / "summary.json").read_text())["records"] == 1
    assert "p50=750.000 ms" in (tmp_path / "summary.md").read_text()


@pytest.mark.asyncio
async def test_run_turn_uses_fake_realtime_events_without_network(tmp_path, monkeypatch):
    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(json.loads(payload))

    case = build_cases()[0]
    turn = case.turns[0]
    websocket = FakeWebSocket()
    events = asyncio.Queue()
    monkeypatch.setattr(benchmark, "_load_pcm16", lambda _path: b"\0" * 32)

    async def emit_events():
        await asyncio.sleep(0.005)
        for event in (
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": turn.prompt,
            },
            {
                "type": "response.output_audio_transcript.done",
                "transcript": "Synthetic answer.",
            },
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(b"\0\1").decode(),
            },
            {"type": "response.output_audio.done"},
            {"type": "response.done", "response": {"status": "completed"}},
        ):
            await events.put((time.perf_counter(), event))

    record, _ = await asyncio.gather(
        benchmark._run_turn(
            websocket,
            events,
            case=case,
            turn=turn,
            turn_index=1,
            audio_path=tmp_path / "unused.wav",
            connection_ms=5.0,
            chunk_ms=1,
            trailing_silence_ms=1,
            timeout_s=1.0,
        ),
        emit_events(),
    )

    assert record["success"] is True
    assert record["assistant_transcript"] == "Synthetic answer."
    assert record["audio_bytes"] == 2
    assert len(websocket.sent) == 2
    assert all(message["type"] == "input_audio_buffer.append" for message in websocket.sent)
