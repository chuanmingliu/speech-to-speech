import assert from "node:assert/strict";
import {
  SLO,
  percentile,
  lampFor,
  formatMs,
  judgeCase,
} from "../demo/ui/metrics.js";

assert.equal(SLO.ttfa.p50, 700);
assert.equal(SLO.ttfa.p95, 1100);
assert.equal(SLO.ttfa.hard, 1200);
assert.equal(SLO.stop.p50, 120);
assert.equal(SLO.stop.p95, 250);
assert.equal(SLO.onset.p50, 64);
assert.equal(SLO.onset.cap, 70);

assert.equal(percentile([], 50), null);
assert.equal(percentile([10], 50), 10);
assert.equal(percentile([10, 20, 30], 50), 20);
assert.equal(lampFor(null, 700), "unmeasured");
assert.equal(lampFor(500, 700), "green");
assert.equal(lampFor(800, 700), "red");
assert.equal(formatMs(null), "unmeasured");
assert.equal(formatMs(511.4), "511 ms");

const empty = { ttfa: [], stop: [], onset: [], hangover: [] };

const c1 = judgeCase(
  "c1",
  [
    { name: "speech_started", t: 1, barging: true },
    { name: "cancel", t: 2 },
    { name: "flush", t: 2 },
  ],
  { ...empty, stop: [90] },
);
assert.equal(c1.verdict, "pass");
assert.match(c1.reason, /Stop 90 ms/);
assert.doesNotMatch(c1.reason, /SLO pass/);

const c3 = judgeCase("c3", [], empty);
assert.equal(c3.verdict, "pass");

const c3fail = judgeCase("c3", [{ name: "speech_started", t: 1 }], empty);
assert.equal(c3fail.verdict, "fail");

const c4 = judgeCase("c4", [{ name: "user_eos", t: 1 }], { ...empty, hangover: [70] });
assert.equal(c4.verdict, "pass");

const c6quiet = judgeCase("c6", [], empty);
assert.equal(c6quiet.verdict, "pass");

const c7 = judgeCase(
  "c7",
  [
    { name: "toolcall", t: 1, tool: "get_time" },
    { name: "first_audio", t: 2 },
  ],
  empty,
);
assert.equal(c7.verdict, "pass");

const latency = judgeCase(
  "latency",
  [
    { name: "user_eos", t: 1 },
    { name: "first_audio", t: 2 },
  ],
  { ...empty, ttfa: [640] },
);
assert.equal(latency.verdict, "pass");
assert.match(latency.reason, /not an SLO pass/);

console.log("metrics.js cases ok");
