#!/usr/bin/env python3
"""OpenShip production entrypoint: HTTP UI on $PORT, realtime backend on loopback.

OpenShip's contract (see openship.io/docs/troubleshooting/deployments):

- It starts ``startCommand`` and waits ~45s for a TCP accept on ``port``
  (or ``PORT`` in the environment). The default FastAPI guess is 8000.
- The process must bind ``0.0.0.0``, not ``127.0.0.1``.
- Secrets come from the project env in the dashboard, not from ``.env``.

This script therefore brings the demo up on PORT first (so the health check
passes), then starts the Tencent/DeepSeek/MiniMax realtime server on 8765
and proxies ``/v1/realtime`` to it.
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
    REQUIRED_ENV,
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

    # OpenShip injects PORT to match the value on the Configuration tab.
    app_port = int(os.environ.get("PORT", "8000"))
    backend_ws = f"ws://127.0.0.1:{BACKEND_PORT}/v1/realtime"
    environment = os.environ.copy()
    environment["S2S_BACKEND_WS"] = backend_ws
    environment["S2S_SAME_ORIGIN"] = "1"
    if environment.get("DEEPSEEK_API_KEY") and not environment.get("OPENAI_API_KEY"):
        environment["OPENAI_API_KEY"] = environment["DEEPSEEK_API_KEY"]
    environment["SPEECH_TO_SPEECH_URL"] = backend_ws

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
    backend: subprocess.Popen[bytes] | None = None
    runtime_profile = None

    try:
        print(f"Demo listening on 0.0.0.0:{app_port} (OpenShip health check)", flush=True)
        wait_for_port(demo, app_port, "browser demo", timeout_s=30.0)

        missing = [name for name in REQUIRED_ENV if not environment.get(name)]
        if missing:
            print(
                "Realtime backend not started; set these in OpenShip project env: "
                + ", ".join(missing),
                flush=True,
            )
        else:
            os.environ.update(
                {key: environment[key] for key in REQUIRED_ENV if environment.get(key)}
            )
            if environment.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = environment["OPENAI_API_KEY"]
            require_environment()
            profile_path = Path(os.environ.get("S2S_PROFILE", DEFAULT_PROFILE))
            runtime_profile = profile_with_port(profile_path, BACKEND_PORT)
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
            print(f"Starting speech-to-speech backend on {backend_ws} ...", flush=True)
            wait_for_port(backend, BACKEND_PORT, "speech-to-speech backend")

        while True:
            if demo.poll() is not None:
                raise RuntimeError(f"browser demo exited with code {demo.returncode}.")
            if backend is not None and backend.poll() is not None:
                raise RuntimeError(f"speech-to-speech backend exited with code {backend.returncode}.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping ...", flush=True)
        return 0
    finally:
        if backend is not None:
            stop_process(backend)
        stop_process(demo)
        if runtime_profile is not None:
            runtime_profile.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
