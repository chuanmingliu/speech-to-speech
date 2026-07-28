import json
import os

import pytest
from scripts.run_custom_services_test_app import (
    choose_backend_port,
    load_env_file,
    profile_with_port,
    require_environment,
)


def test_load_env_file_parses_simple_values_without_overwriting_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            (
                "# comment",
                "PLAIN=value",
                'DOUBLE_QUOTED="hello world"',
                "export SINGLE_QUOTED='voice-id'",
                "EXISTING=replacement",
            )
        )
    )
    monkeypatch.setenv("EXISTING", "original")

    load_env_file(env_file)

    assert os.environ["PLAIN"] == "value"
    assert os.environ["DOUBLE_QUOTED"] == "hello world"
    assert os.environ["SINGLE_QUOTED"] == "voice-id"
    assert os.environ["EXISTING"] == "original"


def test_load_env_file_rejects_malformed_entries(tmp_path):
    env_file = tmp_path / ".env.local"
    env_file.write_text("NOT_AN_ASSIGNMENT")

    with pytest.raises(ValueError, match="Invalid environment entry"):
        load_env_file(env_file)


def test_require_environment_sets_openai_alias(monkeypatch):
    values = {
        "DEEPSEEK_API_KEY": "deepseek",
        "TENCENT_ASR_SECRET_ID": "id",
        "TENCENT_ASR_SECRET_KEY": "key",
        "MINIMAX_TTS_API_KEY": "minimax",
        "MINIMAX_TTS_VOICE_ID": "voice",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    require_environment()

    assert os.environ["OPENAI_API_KEY"] == "deepseek"


def test_require_environment_reports_missing_names(monkeypatch):
    for name in (
        "DEEPSEEK_API_KEY",
        "TENCENT_ASR_SECRET_ID",
        "TENCENT_ASR_SECRET_KEY",
        "MINIMAX_TTS_API_KEY",
        "MINIMAX_TTS_VOICE_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        require_environment()


def test_choose_backend_port_returns_a_free_port():
    assert choose_backend_port(0) > 0


def test_profile_with_port_overrides_ws_port(tmp_path):
    source = tmp_path / "profile.json"
    source.write_text('{"mode": "realtime", "ws_port": 8765}')

    generated = profile_with_port(source, 18765)
    try:
        assert json.loads(generated.read_text())["ws_port"] == 18765
    finally:
        generated.unlink()
