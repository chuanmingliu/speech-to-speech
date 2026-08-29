"""Per-pipeline logging context.

Each isolated pipeline unit sets the contextvar at thread / asyncio-task entry,
and every log record emitted from that thread/task is automatically tagged with
a `[pipeline N] ` prefix via PipelineLogFilter. Non-pipeline-bound code emits
records with an empty prefix.

Works for both threading (handler threads) and asyncio (websocket route, send loops)
because contextvars are per-thread and per-task.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from queue import Full, Queue
from threading import Thread
from typing import Any, Optional
from urllib.request import Request, urlopen

pipeline_log_ctx: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("pipeline_index", default=None)
logger = logging.getLogger(__name__)
_INGEST_Q: Queue[bytes] = Queue(maxsize=500)
_INGEST_STARTED = False


def _ingest_url() -> str:
    return os.getenv("TEL_MONITOR_URL", "http://127.0.0.1:8768/ingest")


def _ingest_worker() -> None:
    url = _ingest_url()
    while True:
        payload = _INGEST_Q.get()
        try:
            urlopen(Request(url, data=payload, method="POST", headers={"content-type": "application/json"}), timeout=0.2)
        except Exception:
            pass


def _ensure_ingest() -> None:
    global _INGEST_STARTED
    if _INGEST_STARTED:
        return
    _INGEST_STARTED = True
    Thread(target=_ingest_worker, name="tel-ingest", daemon=True).start()


def tel_log(component: str, event: str, *, t0: float | None = None, **kv: Any) -> None:
    """Essential telephony trace. Grep docker logs for ``[tel]``. Also POSTs to the local monitor."""
    elapsed = 0
    if t0 is not None:
        elapsed = int(max(0.0, (time.monotonic() - t0) * 1000))
    extra = " ".join(f"{k}={v}" for k, v in kv.items() if v is not None and v != "")
    logger.info("[tel] t+%dms %s %s%s", elapsed, component, event, f" {extra}" if extra else "")
    body = {"at": int(time.time() * 1000), "tMs": elapsed, "t_ms": elapsed, "component": component, "event": event}
    body.update({k: v for k, v in kv.items() if isinstance(v, (str, int, float, bool))})
    try:
        _ensure_ingest()
        _INGEST_Q.put_nowait(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    except Full:
        pass
    except Exception:
        pass


class PipelineLogFilter(logging.Filter):
    """Inject `pipeline_prefix` into every LogRecord based on the current contextvar."""

    def filter(self, record: logging.LogRecord) -> bool:
        idx = pipeline_log_ctx.get()
        record.pipeline_prefix = f"[pipeline {idx}] " if idx is not None else ""
        return True
