from __future__ import annotations

import os
import subprocess
import sys


def test_import_does_not_load_optional_deepfilter():
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import speech_to_speech.VAD.vad_handler; print('df' in sys.modules)",
        ],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"
    assert "DeepFilterNet not available" not in result.stderr


def test_deepfilter_import_supports_current_torchaudio():
    from speech_to_speech.VAD.vad_handler import _load_deepfilter

    enhance, init_df = _load_deepfilter()

    assert callable(enhance)
    assert callable(init_df)
