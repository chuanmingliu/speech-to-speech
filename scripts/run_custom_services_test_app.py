#!/usr/bin/env python3
"""Launch the custom-services realtime backend and bundled browser voice app."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / ".env.local"
DEFAULT_PROFILE = REPO_ROOT / "configs" / "tencent-deepseek-minimax.json"
REQUIRED_ENV = (
    "DEEPSEEK_API_KEY",
    "TENCENT_ASR_APP_ID",
    "TENCENT_ASR_SECRET_ID",
    "TENCENT_ASR_SECRET_KEY",
    "MINIMAX_TTS_API_KEY",
    "MINIMAX_TTS_VOICE_ID",
)


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without executing shell syntax."""
    if not path.exists():
        raise FileNotFoundError(f"Environment file does not exist: {path}")
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Invalid environment entry at {path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def require_environment() -> None:
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required variables: {', '.join(missing)}")
    os.environ.setdefault("OPENAI_API_KEY", os.environ["DEEPSEEK_API_KEY"])


def wait_for_port(process: subprocess.Popen[bytes], port: int, label: str, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"{label} exited during startup with code {return_code}.")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {label} on port {port}.")


def port_is_available(port: int) -> bool:
    """Return whether a local TCP port can be bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def choose_backend_port(preferred_port: int) -> int:
    """Use the preferred port when possible, otherwise ask the OS for a free port."""
    if preferred_port and port_is_available(preferred_port):
        return preferred_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("0.0.0.0", 0))
        return int(probe.getsockname()[1])


def profile_with_port(profile_path: Path, port: int) -> Path:
    """Create a temporary profile with the selected realtime server port."""
    profile = json.loads(profile_path.read_text())
    profile["ws_port"] = port
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="speech-to-speech-custom-",
        suffix=".json",
        delete=False,
    )
    with handle:
        json.dump(profile, handle)
    return Path(handle.name)


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--backend-port", type=int, default=8765)
    parser.add_argument("--app-port", type=int, default=7860)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    require_environment()

    backend_port = choose_backend_port(args.backend_port)
    runtime_profile = profile_with_port(args.profile, backend_port)
    realtime_url = f"ws://127.0.0.1:{backend_port}/v1/realtime"
    environment = os.environ.copy()
    environment["SPEECH_TO_SPEECH_URL"] = realtime_url
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
        if backend_port != args.backend_port:
            print(f"Port {args.backend_port} is busy; using free port {backend_port}.")
        print(f"Starting custom-services backend on {realtime_url} ...")
        wait_for_port(backend, backend_port, "speech-to-speech backend")

        demo = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "--app-dir",
                "demo",
                "server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.app_port),
            ],
            cwd=REPO_ROOT,
            env=environment,
        )
        wait_for_port(demo, args.app_port, "browser test app")

        print()
        print(f"Test app ready: http://127.0.0.1:{args.app_port}")
        print("Click the center orb, allow microphone access, and speak.")
        print("Press Ctrl-C here to stop both processes.")

        while True:
            backend_code = backend.poll()
            demo_code = demo.poll()
            if backend_code is not None:
                raise RuntimeError(f"speech-to-speech backend exited with code {backend_code}.")
            if demo_code is not None:
                raise RuntimeError(f"browser test app exited with code {demo_code}.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping test app ...")
        return 0
    finally:
        if demo is not None:
            stop_process(demo)
        stop_process(backend)
        runtime_profile.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
