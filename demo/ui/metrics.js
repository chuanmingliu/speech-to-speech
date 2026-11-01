// @ts-check
/**
 * Client-side turn timings for the demo lab panel.
 *
 * Every number is measured in the browser (`performance.now()`). Unknown
 * values stay `null` and render as "unmeasured" — this module never invents
 * a latency, percentile, or SLO pass.
 *
 * TTFA:     user_eos (speech_stopped) → first audible bot audio
 * stop:     client-measured barge-in onset (RMS while AI speaking) → flush
 * onset:    client-measured speech energy → speech_started
 * hangover: client-measured energy drop → speech_stopped
 */

/** Locked SLO ceilings (ms). Lamps compare measurements to these; they do
 *  not certify a session-level pass. */
export const SLO = Object.freeze({
  ttfa: Object.freeze({ p50: 700, p95: 1100, hard: 1200 }),
  stop: Object.freeze({ p50: 120, p95: 250 }),
  onset: Object.freeze({ p50: 64, cap: 70 }),
});

/** dBFS above which the client treats mic energy as speech (onset). */
const ONSET_DB = -38;
/** dBFS below which the client treats mic energy as silence (hangover start). */
const DROP_DB = -48;
/** Ignore an onset/drop pairing older than this (ms). */
const PAIR_WINDOW_MS = 2500;

/**
 * Linear-interpolated percentile. Empty input → `null` (unmeasured).
 * @param {number[]} values
 * @param {number} p 0–100
 * @returns {number | null}
 */
export function percentile(values, p) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const idx = (sorted.length - 1) * (p / 100);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/**
 * @param {number | null | undefined} value
 * @param {number} limit
 * @returns {"green" | "red" | "unmeasured"}
 */
export function lampFor(value, limit) {
  if (value == null || !Number.isFinite(value)) return "unmeasured";
  return value <= limit ? "green" : "red";
}

/** @param {number | null | undefined} value */
export function formatMs(value) {
  if (value == null || !Number.isFinite(value)) return "unmeasured";
  return `${Math.round(value)} ms`;
}

/** @param {number} rms */
export function rmsToDb(rms) {
  if (!(rms > 0)) return -120;
  return 20 * Math.log10(rms);
}

/**
 * @typedef {"speech_started" | "user_eos" | "cancel" | "flush" | "first_audio" | "toolcall"} ProtocolName
 *
 * @typedef {Object} ProtocolEvent
 * @property {ProtocolName} name
 * @property {number} t
 * @property {string} [tool]
 * @property {string} [source]
 * @property {boolean} [barging]
 * @property {string} [label]
 *
 * @typedef {Object} CaseDef
 * @property {string} id
 * @property {string} title
 * @property {string} instructions
 */

/** @type {readonly CaseDef[]} */
export const TEST_CASES = Object.freeze([
  {
    id: "c1",
    title: "C1 barge-in real",
    instructions:
      "Tap the orb, wait until the bot is talking, then speak over it. The bot should stop. Pass/fail is whether cancel/flush fired — not whether stop latency meets the SLO.",
  },
  {
    id: "c3",
    title: "C3 false-barge",
    instructions:
      "While the bot is talking, make a brief click or cough (well under 100 ms). The bot should keep talking. A speech_started or cancel during this run is a fail.",
  },
  {
    id: "c4",
    title: "C4 hangover confirm",
    instructions:
      "Say a short phrase, then stay quiet. Confirm speech_stopped arrives after the silence hangover (min_silence_ms is 64 ms; this is not the 384 ms turn-commit bar).",
  },
  {
    id: "c6",
    title: "C6 silence nudge",
    instructions:
      "Speak one short phrase, then stay silent. The turn should commit on silence (the nudge) and the bot should reply. Extra speech_started events during that silence are a fail. Sitting in silence with no false onset also passes.",
  },
  {
    id: "c7",
    title: "C7 get_time",
    instructions:
      "Ask “what time is it?” The assistant should call the get_time tool and speak the answer. Pass is the tool call plus a reply — not TTFA.",
  },
  {
    id: "latency",
    title: "Latency turn",
    instructions:
      "Say a short phrase and wait for the first bot audio. This records TTFA (user end-of-speech → first audible audio). The lamp is observational; this case never auto-claims an SLO pass.",
  },
]);

/**
 * Session + current-turn timing store. Feed it `protocol` events, input
 * levels, and connection status; read `snapshot()` for the panel.
 */
export class SessionMetrics {
  constructor() {
    this.reset();
  }

  reset() {
    /** @type {number[]} */
    this.ttfa = [];
    /** @type {number[]} */
    this.stop = [];
    /** @type {number[]} */
    this.onset = [];
    /** @type {number[]} */
    this.hangover = [];
    /** @type {ProtocolEvent[]} */
    this.log = [];
    this.current = emptyTurn();
    this.connection = "idle";
    this.aiSpeaking = false;
    /** @type {string | null} */
    this.activeCase = null;
    this.caseStartedAt = 0;
    /** @type {ProtocolEvent[]} */
    this.caseEvents = [];
    /** @type {ReturnType<typeof emptyCaseSamples>} */
    this.caseSamples = emptyCaseSamples();
    /** @type {{ verdict: "pass" | "fail" | "incomplete"; reason: string } | null} */
    this.lastVerdict = null;
    this._inClientSpeech = false;
    this._pendingOnset = null;
    this._pendingDrop = null;
    this._pendingBargeOnset = null;
    this._pendingEos = null;
    this._firstAudioForEos = false;
    this._sawAudiblePlayback = false;
  }

  /** @param {string} status */
  noteStatus(status) {
    this.connection = status;
    this.aiSpeaking = status === "ai-speaking";
    if (status === "idle" || status === "closed" || status === "error") {
      this.current = emptyTurn();
      this._inClientSpeech = false;
      this._pendingOnset = null;
      this._pendingDrop = null;
      this._pendingBargeOnset = null;
      this._pendingEos = null;
      this._firstAudioForEos = false;
      this._sawAudiblePlayback = false;
    }
  }

  /**
   * Client-measured energy tracker (mic RMS from the capture worklet).
   * @param {number} rms
   * @param {number} [t]
   */
  noteInputLevel(rms, t = performance.now()) {
    const db = rmsToDb(rms);
    if (db >= ONSET_DB) {
      if (!this._inClientSpeech) {
        this._inClientSpeech = true;
        this._pendingOnset = t;
        this.current.clientOnset = t;
        if (this.aiSpeaking) this._pendingBargeOnset = t;
      }
      this._pendingDrop = null;
    } else if (db <= DROP_DB && this._inClientSpeech) {
      this._inClientSpeech = false;
      this._pendingDrop = t;
      this.current.clientEnergyDrop = t;
    }
  }

  /**
   * @param {ProtocolEvent} event
   */
  noteProtocol(event) {
    const t = event.t;
    const labeled = {
      ...event,
      label: event.label || "client-measured",
    };
    this.log.push(labeled);
    if (this.log.length > 80) this.log.shift();
    if (this.activeCase) this.caseEvents.push(labeled);

    switch (event.name) {
      case "speech_started": {
        this.current.speechStarted = t;
        const onset = this._pendingOnset;
        if (onset != null && t - onset <= PAIR_WINDOW_MS) {
          const ms = t - onset;
          this.onset.push(ms);
          this.current.onset = ms;
          if (this.activeCase) this.caseSamples.onset.push(ms);
        }
        if (event.barging || this.aiSpeaking) {
          this.current.barging = true;
        }
        this._pendingOnset = null;
        this._pendingEos = null;
        this._firstAudioForEos = false;
        this._sawAudiblePlayback = false;
        break;
      }
      case "user_eos": {
        this.current.userEos = t;
        const drop = this._pendingDrop;
        if (drop != null && t - drop <= PAIR_WINDOW_MS) {
          const ms = t - drop;
          this.hangover.push(ms);
          this.current.hangover = ms;
          if (this.activeCase) this.caseSamples.hangover.push(ms);
        }
        this._pendingDrop = null;
        this._pendingEos = t;
        this._firstAudioForEos = false;
        this._sawAudiblePlayback = false;
        break;
      }
      case "flush":
      case "cancel": {
        if (event.name === "flush") this.current.flush = t;
        if (event.name === "cancel") this.current.cancel = t;
        const barge = this._pendingBargeOnset;
        if (barge != null && t - barge <= PAIR_WINDOW_MS) {
          const ms = t - barge;
          this.stop.push(ms);
          this.current.stop = ms;
          if (this.activeCase) this.caseSamples.stop.push(ms);
        }
        this._pendingBargeOnset = null;
        break;
      }
      case "first_audio": {
        // Prefer the playback-worklet "audible" mark; a delta that arrives
        // earlier is kept only until that mark (or used alone on WebRTC).
        if (event.source === "playback") {
          this._recordFirstAudio(t, "playback");
          this._sawAudiblePlayback = true;
        } else if (!this._sawAudiblePlayback && !this._firstAudioForEos) {
          this._recordFirstAudio(t, event.source || "delta");
        }
        break;
      }
      case "toolcall": {
        this.current.tool = event.tool || "";
        break;
      }
    }
  }

  /**
   * @param {number} t
   * @param {string} source
   */
  _recordFirstAudio(t, source) {
    this.current.firstAudio = t;
    this.current.firstAudioSource = source;
    const eos = this._pendingEos;
    if (eos != null && t - eos <= 30_000) {
      const ms = t - eos;
      if (!this._firstAudioForEos) {
        this.ttfa.push(ms);
        if (this.activeCase) this.caseSamples.ttfa.push(ms);
      } else if (source === "playback" && this.ttfa.length) {
        // Replace the delta-based sample with the audible one.
        this.ttfa[this.ttfa.length - 1] = ms;
        if (this.activeCase && this.caseSamples.ttfa.length) {
          this.caseSamples.ttfa[this.caseSamples.ttfa.length - 1] = ms;
        }
      }
      this.current.ttfa = ms;
      this._firstAudioForEos = true;
    }
  }

  /** @param {string} id */
  startCase(id) {
    this.activeCase = id;
    this.caseStartedAt = performance.now();
    this.caseEvents = [];
    this.caseSamples = emptyCaseSamples();
    this.lastVerdict = null;
    this.current = emptyTurn();
    this._pendingEos = null;
    this._firstAudioForEos = false;
    this._sawAudiblePlayback = false;
  }

  /**
   * Close the active case and grade it from recorded events only.
   * @returns {{ verdict: "pass" | "fail" | "incomplete"; reason: string } | null}
   */
  stopCase() {
    if (!this.activeCase) return null;
    const id = this.activeCase;
    const verdict = judgeCase(id, this.caseEvents, this.caseSamples);
    this.lastVerdict = verdict;
    this.activeCase = null;
    return verdict;
  }

  snapshot() {
    const ttfa = summarize(this.ttfa);
    const stop = summarize(this.stop);
    const onset = summarize(this.onset);
    const hangover = summarize(this.hangover);
    return {
      connection: this.connection,
      aiSpeaking: this.aiSpeaking,
      current: { ...this.current },
      session: { ttfa, stop, onset, hangover },
      lamps: {
        ttfaP50: lampFor(ttfa.p50, SLO.ttfa.p50),
        ttfaP95: lampFor(ttfa.p95, SLO.ttfa.p95),
        ttfaHard: lampFor(ttfa.max, SLO.ttfa.hard),
        stopP50: lampFor(stop.p50, SLO.stop.p50),
        stopP95: lampFor(stop.p95, SLO.stop.p95),
        onsetP50: lampFor(onset.p50, SLO.onset.p50),
        onsetCap: lampFor(onset.max, SLO.onset.cap),
      },
      log: this.log.slice(),
      activeCase: this.activeCase,
      lastVerdict: this.lastVerdict,
      caseSamples: {
        ttfa: this.caseSamples.ttfa.slice(),
        stop: this.caseSamples.stop.slice(),
        onset: this.caseSamples.onset.slice(),
        hangover: this.caseSamples.hangover.slice(),
      },
    };
  }
}

function emptyTurn() {
  return {
    clientOnset: /** @type {number | null} */ (null),
    clientEnergyDrop: /** @type {number | null} */ (null),
    speechStarted: /** @type {number | null} */ (null),
    userEos: /** @type {number | null} */ (null),
    firstAudio: /** @type {number | null} */ (null),
    firstAudioSource: "",
    flush: /** @type {number | null} */ (null),
    cancel: /** @type {number | null} */ (null),
    barging: false,
    tool: "",
    ttfa: /** @type {number | null} */ (null),
    stop: /** @type {number | null} */ (null),
    onset: /** @type {number | null} */ (null),
    hangover: /** @type {number | null} */ (null),
  };
}

function emptyCaseSamples() {
  return { ttfa: /** @type {number[]} */ ([]), stop: /** @type {number[]} */ ([]), onset: /** @type {number[]} */ ([]), hangover: /** @type {number[]} */ ([]) };
}

/** @param {number[]} values */
function summarize(values) {
  return {
    n: values.length,
    p50: percentile(values, 50),
    p95: percentile(values, 95),
    max: values.length ? Math.max(...values) : null,
  };
}

/**
 * Grade a named case from events collected between Start and Stop.
 * Behavioral pass/fail only — never an SLO certification.
 *
 * @param {string} id
 * @param {ProtocolEvent[]} events
 * @param {{ ttfa: number[]; stop: number[]; onset: number[]; hangover: number[] }} samples
 */
export function judgeCase(id, events, samples) {
  const names = new Set(events.map((e) => e.name));
  const started = names.has("speech_started");
  const eos = names.has("user_eos");
  const cancel = names.has("cancel") || names.has("flush");
  const audio = names.has("first_audio");
  const timeTool = events.some((e) => e.name === "toolcall" && e.tool === "get_time");

  if (id === "c1") {
    if (!started) {
      return { verdict: /** @type {const} */ ("incomplete"), reason: "No speech_started. Talk over the bot while it is speaking." };
    }
    if (!cancel) {
      return { verdict: /** @type {const} */ ("fail"), reason: "speech_started arrived but no cancel/flush — the bot may have kept talking." };
    }
    const stop = samples.stop[samples.stop.length - 1];
    const extra = stop != null ? ` Stop ${formatMs(stop)} (client-measured; not an SLO claim).` : " Stop latency unmeasured (no client onset while AI speaking).";
    return { verdict: /** @type {const} */ ("pass"), reason: `Cancel/flush observed.${extra}` };
  }

  if (id === "c3") {
    if (started || cancel) {
      return { verdict: /** @type {const} */ ("fail"), reason: "speech_started or cancel/flush fired — this brief noise counted as a barge-in." };
    }
    return { verdict: /** @type {const} */ ("pass"), reason: "No speech_started or cancel during the run." };
  }

  if (id === "c4") {
    if (!eos) {
      return { verdict: /** @type {const} */ ("incomplete"), reason: "No user_eos (speech_stopped). Say a phrase, then stay quiet." };
    }
    const hang = samples.hangover[samples.hangover.length - 1];
    if (hang == null) {
      return { verdict: /** @type {const} */ ("pass"), reason: "user_eos observed. Hangover unmeasured (no client energy drop paired with it)." };
    }
    return { verdict: /** @type {const} */ ("pass"), reason: `user_eos after hangover ${formatMs(hang)} (client-measured).` };
  }

  if (id === "c6") {
    if (!started && !eos && !audio) {
      return { verdict: /** @type {const} */ ("pass"), reason: "Silence held: no false speech_started, user_eos, or first_audio." };
    }
    if (eos && audio) {
      const eosT = events.find((e) => e.name === "user_eos")?.t ?? 0;
      const extraStart = events.some((e) => e.name === "speech_started" && e.t > eosT);
      if (extraStart) {
        return { verdict: /** @type {const} */ ("fail"), reason: "Silence after user_eos produced another speech_started." };
      }
      return { verdict: /** @type {const} */ ("pass"), reason: "Silence committed the turn; first_audio followed without a new onset." };
    }
    if (started && !eos) {
      return { verdict: /** @type {const} */ ("incomplete"), reason: "speech_started but no user_eos yet. Stay silent after you speak." };
    }
    return { verdict: /** @type {const} */ ("incomplete"), reason: "Need either a quiet hold (no events) or a spoken turn that commits on silence." };
  }

  if (id === "c7") {
    if (!timeTool) {
      return { verdict: /** @type {const} */ ("fail"), reason: "get_time was not called. Ask for the time while the get_time tool is enabled." };
    }
    if (!audio && !names.has("first_audio")) {
      return { verdict: /** @type {const} */ ("incomplete"), reason: "get_time ran, but no first_audio yet. Wait for the spoken answer." };
    }
    return { verdict: /** @type {const} */ ("pass"), reason: "get_time ran and a reply was heard." };
  }

  if (id === "latency") {
    if (!eos || !audio) {
      return { verdict: /** @type {const} */ ("incomplete"), reason: "Need user_eos and first_audio. Speak a short phrase, then wait." };
    }
    const ttfa = samples.ttfa[samples.ttfa.length - 1];
    return {
      verdict: /** @type {const} */ ("pass"),
      reason: `TTFA ${formatMs(ttfa)} (client-measured). Observational only — not an SLO pass.`,
    };
  }

  return { verdict: /** @type {const} */ ("incomplete"), reason: "Unknown case." };
}
