// @ts-check
/**
 * Monitor + named test-case panel for the local browser demo.
 *
 * Owns the slide-in lab dock: connection, current-turn timings, session
 * percentiles, SLO lamps (green/red/unmeasured), the protocol event log,
 * and Start/Stop grading for C1/C3/C4/C6/C7 plus a latency turn.
 */

import { $, escHtml } from "./dom.js";
import {
  SLO,
  TEST_CASES,
  SessionMetrics,
  formatMs,
} from "./metrics.js";

const OPEN_CLASS = "lab-open";

export class LabView {
  constructor() {
    this.metrics = new SessionMetrics();
    /** @type {HTMLElement} */
    this.root = $("#lab-panel");
    /** @type {HTMLButtonElement} */
    this.toggleBtn = $("#lab-btn");
    /** @type {HTMLButtonElement} */
    this.closeBtn = $("#lab-panel-close");
    /** @type {HTMLElement} */
    this.connEl = $("#lab-conn");
    /** @type {HTMLElement} */
    this.urlEl = $("#lab-url");
    /** @type {HTMLElement} */
    this.turnEl = $("#lab-turn");
    /** @type {HTMLElement} */
    this.sessionEl = $("#lab-session");
    /** @type {HTMLElement} */
    this.lampsEl = $("#lab-lamps");
    /** @type {HTMLElement} */
    this.logEl = $("#lab-log");
    /** @type {HTMLElement} */
    this.casesEl = $("#lab-cases");
    /** @type {HTMLElement} */
    this.caseHint = $("#lab-case-hint");
    /** @type {HTMLButtonElement} */
    this.startBtn = $("#lab-case-start");
    /** @type {HTMLButtonElement} */
    this.stopBtn = $("#lab-case-stop");
    /** @type {HTMLElement} */
    this.verdictEl = $("#lab-verdict");
    /** @type {string} */
    this.selectedCase = TEST_CASES[0].id;
    this._open = false;
    this._url = "";

    this.toggleBtn.addEventListener("click", () => {
      if (this._open) this.close();
      else this.open();
    });
    this.closeBtn.addEventListener("click", () => this.close());
    this.startBtn.addEventListener("click", () => this._startSelected());
    this.stopBtn.addEventListener("click", () => this._stopSelected());

    this._renderCases();
    this._syncCaseHint();
    this.render();
  }

  /** Open the dock. Used by `/lab` and the toolbar button. */
  open() {
    this._open = true;
    this.root.classList.add("open");
    this.root.removeAttribute("hidden");
    document.body.classList.add(OPEN_CLASS);
    this.toggleBtn.setAttribute("aria-pressed", "true");
    this.render();
  }

  close() {
    this._open = false;
    this.root.classList.remove("open");
    document.body.classList.remove(OPEN_CLASS);
    this.toggleBtn.setAttribute("aria-pressed", "false");
  }

  get isOpen() {
    return this._open;
  }

  /** @param {string} url */
  setUrl(url) {
    this._url = url || "";
    this.render();
  }

  /** @param {string} status */
  noteStatus(status) {
    this.metrics.noteStatus(status);
    this.render();
  }

  /** @param {number} rms */
  noteInputLevel(rms) {
    this.metrics.noteInputLevel(rms);
  }

  /**
   * @param {{ name: string; t?: number; tool?: string; source?: string; barging?: boolean }} event
   */
  noteProtocol(event) {
    this.metrics.noteProtocol({
      name: /** @type {any} */ (event.name),
      t: event.t ?? performance.now(),
      tool: event.tool,
      source: event.source,
      barging: event.barging,
    });
    this.render();
  }

  resetSession() {
    const keepCase = this.metrics.activeCase;
    this.metrics.reset();
    if (keepCase) this.metrics.startCase(keepCase);
    this.render();
  }

  _startSelected() {
    this.metrics.startCase(this.selectedCase);
    this.startBtn.disabled = true;
    this.stopBtn.disabled = false;
    this.verdictEl.textContent = "Recording. Perform the steps, then Stop.";
    this.verdictEl.dataset.verdict = "running";
    this._syncCaseHint();
    this.render();
  }

  _stopSelected() {
    const verdict = this.metrics.stopCase();
    this.startBtn.disabled = false;
    this.stopBtn.disabled = true;
    if (verdict) {
      this.verdictEl.textContent = `${verdict.verdict}: ${verdict.reason}`;
      this.verdictEl.dataset.verdict = verdict.verdict;
    }
    this._syncCaseHint();
    this.render();
  }

  _renderCases() {
    this.casesEl.replaceChildren();
    for (const c of TEST_CASES) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "lab-case-btn";
      btn.dataset.id = c.id;
      btn.textContent = c.title;
      btn.setAttribute("aria-pressed", c.id === this.selectedCase ? "true" : "false");
      btn.addEventListener("click", () => {
        if (this.metrics.activeCase) return;
        this.selectedCase = c.id;
        this._syncCaseHint();
        this._renderCases();
      });
      this.casesEl.appendChild(btn);
    }
  }

  _syncCaseHint() {
    const c = TEST_CASES.find((x) => x.id === this.selectedCase);
    this.caseHint.textContent = c ? c.instructions : "";
    this.startBtn.disabled = !!this.metrics.activeCase;
    this.stopBtn.disabled = !this.metrics.activeCase;
  }

  render() {
    const snap = this.metrics.snapshot();
    const conn = connectionLabel(snap.connection);
    this.connEl.textContent = conn.text;
    this.connEl.dataset.state = conn.state;
    this.urlEl.textContent = this._url || "unmeasured";

    this.turnEl.innerHTML = [
      metricRow("TTFA", snap.current.ttfa, "client-measured"),
      metricRow("Stop", snap.current.stop, "client-measured"),
      metricRow("Onset", snap.current.onset, "client-measured"),
      metricRow("Hangover", snap.current.hangover, "client-measured"),
    ].join("");

    this.sessionEl.innerHTML = [
      sessionRow("TTFA", snap.session.ttfa),
      sessionRow("Stop", snap.session.stop),
      sessionRow("Onset", snap.session.onset),
      sessionRow("Hangover", snap.session.hangover),
    ].join("");

    this.lampsEl.innerHTML = [
      lamp("TTFA p50", snap.lamps.ttfaP50, `≤${SLO.ttfa.p50}`),
      lamp("TTFA p95", snap.lamps.ttfaP95, `≤${SLO.ttfa.p95}`),
      lamp("TTFA hard", snap.lamps.ttfaHard, `≤${SLO.ttfa.hard}`),
      lamp("Stop p50", snap.lamps.stopP50, `≤${SLO.stop.p50}`),
      lamp("Stop p95", snap.lamps.stopP95, `≤${SLO.stop.p95}`),
      lamp("Onset p50", snap.lamps.onsetP50, `≤${SLO.onset.p50}`),
      lamp("Onset cap", snap.lamps.onsetCap, `≤${SLO.onset.cap}`),
    ].join("");

    if (!snap.log.length) {
      this.logEl.innerHTML = `<p class="lab-log-empty">No events yet. Tap the orb and talk.</p>`;
    } else {
      const rows = snap.log.slice(-40).reverse().map((e) => {
        const ms = clock(e.t);
        return `<div class="lab-log-row"><span class="lab-log-t">${escHtml(ms)}</span><span class="lab-log-n">${escHtml(e.name)}</span><span class="lab-log-l">${escHtml(e.label || "client-measured")}</span></div>`;
      });
      this.logEl.innerHTML = rows.join("");
    }
  }
}

/** Open `/lab` (or `?lab=1` / `#lab`) with the dock visible. */
export function shouldOpenLab() {
  try {
    const path = location.pathname.replace(/\/+$/, "") || "/";
    if (path === "/lab") return true;
    if (new URLSearchParams(location.search).has("lab")) return true;
    if (location.hash === "#lab") return true;
  } catch {
    // ignore
  }
  return false;
}

/** @param {string} status */
function connectionLabel(status) {
  switch (status) {
    case "connected":
    case "listening":
    case "user-speaking":
    case "processing":
    case "ai-speaking":
      return { text: status, state: "live" };
    case "connecting":
    case "creating-session":
    case "queued":
    case "your-turn":
      return { text: status, state: "wait" };
    case "error":
      return { text: "error", state: "err" };
    default:
      return { text: status || "idle", state: "idle" };
  }
}

/** @param {string} label @param {number | null} value @param {string} note */
function metricRow(label, value, note) {
  const shown = formatMs(value);
  const kind = value == null ? "unmeasured" : "value";
  return `<div class="lab-metric"><span class="lab-metric-k">${escHtml(label)}</span><span class="lab-metric-v ${kind}">${escHtml(shown)}</span><span class="lab-metric-n">${escHtml(note)}</span></div>`;
}

/** @param {string} label @param {{ n: number; p50: number | null; p95: number | null; max: number | null }} s */
function sessionRow(label, s) {
  const n = s.n ? `${s.n}` : "0";
  return `<div class="lab-metric"><span class="lab-metric-k">${escHtml(label)}</span><span class="lab-metric-v ${s.n ? "value" : "unmeasured"}">p50 ${escHtml(formatMs(s.p50))} · p95 ${escHtml(formatMs(s.p95))} · max ${escHtml(formatMs(s.max))}</span><span class="lab-metric-n">n=${escHtml(n)}</span></div>`;
}

/** @param {string} label @param {"green" | "red" | "unmeasured"} state @param {string} limit */
function lamp(label, state, limit) {
  return `<div class="lab-lamp" data-state="${escHtml(state)}"><span class="lab-lamp-dot" aria-hidden="true"></span><span class="lab-lamp-k">${escHtml(label)}</span><span class="lab-lamp-s">${escHtml(state)}</span><span class="lab-lamp-n">${escHtml(limit)} ms</span></div>`;
}

/** @param {number} t */
function clock(t) {
  const d = new Date(performance.timeOrigin + t);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  const ms = String(d.getMilliseconds()).padStart(3, "0");
  return `${hh}:${mm}:${ss}.${ms}`;
}
