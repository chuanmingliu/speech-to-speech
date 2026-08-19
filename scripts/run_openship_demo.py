#!/usr/bin/env python3
"""Start the Tencent/DeepSeek/MiniMax realtime backend and the browser demo.

This is the production entrypoint for OpenShip (and any other container host).
It differs from ``scripts/run_custom_services_test_app.py`` in three ways:

- Listens on ``0.0.0.0`` so the platform edge can reach the demo.
- Honours ``PORT`` (OpenShip / most PaaS inject this).
- Proxies Realtime WebSocket through the demo (``S2S_SAME_ORIGIN``) so the
  browser can use ``wss://<public-host>/v1/realtime`` instead of dialing
  loopback port 8765, which is not reachable from the user's machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_custom_services_test_app import (
    DEFAULT_ENV_FILE,
    DEFAULT_PROFILE,
    load_env_file,
    profile_with_port,
    require_environment,
    stop_process,
    wait_for_port,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PORT = int(os.environ.get("S2S_BACKEND_PORT", "8765"))


def main() -> int:
    env_file = Path(os.environ.get("S2S_ENV_FILE", DEFAULT_ENV_FILE))
    if env_file.exists():
        load_env_file(env_file)
    require_environment()

    app_port = int(os.environ.get("PORT", os.environ.get("S2S_APP_PORT", "7860")))
    profile_path = Path(os.environ.get("S2S_PROFILE", DEFAULT_PROFILE))
    runtime_profile = profile_with_port(profile_path, BACKEND_PORT)
    backend_ws = f"ws://127.0.0.1:{BACKEND_PORT}/v1/realtime"

    environment = os.environ.copy()
    environment["S2S_BACKEND_WS"] = backend_ws
    environment["S2S_SAME_ORIGIN"] = "1"
    environment["OPENAI_API_KEY"] = environment.get("OPENAI_API_KEY") or environment["DEEPSEEK_API_KEY"]
    # Internal only — /api/config rewrites this to the public host when
    # S2S_SAME_ORIGIN=1. Keep WebRTC off in that mode.
    environment["SPEECH_TO_SPEECH_URL"] = backend_ws

    backend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "speech_to_speech.s2s_pipeline",
            str(runtime_profile),
        ],
        cwd=REPO_ROOT,
        env=environment,
    )
    demo: subprocess.Popen[bytes] | None = None

    try:
        print(f"Starting speech-to-speech backend on {backend_ws} ...")
        wait_for_port(backend, BACKEND_PORT, "speech-to-speech backend")

        demo = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "--app-dir",
                "demo",
                "server:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(app_port),
            ],
            cwd=REPO_ROOT,
            env=environment,
        )
        wait_for_port(demo, app_port, "browser demo")
        print(f"Demo listening on 0.0.0.0:{app_port}")

        while True:
            backend_code = backend.poll()
            demo_code = demo.poll()
            if backend_code is not None:
                raise RuntimeError(f"speech-to-speech backend exited with code {backend_code}.")
            if demo_code is not None:
                raise RuntimeError(f"browser demo exited with code {demo_code}.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping ...")
        return 0
    finally:
        if demo is not None:
            stop_process(demo)
        stop_process(backend)
        runtime_profile.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
