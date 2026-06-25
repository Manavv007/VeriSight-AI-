/* eval.mjs — replay a corpus of labeled traces and report detection metrics.
   Usage: node tools/eval.mjs [corpusDir]
   Exit code 0 if all targets met, 1 otherwise (so the agent loop can branch).
   Uses the CURRENT gaze-math thresholds, so editing gaze-math + re-running this
   is the autonomous tuning loop. */
import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { replayTrace } from "../js/detection-core.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const corpusDir = resolve(process.argv[2] || join(here, "..", "corpus"));

// Pass targets.
const TARGET = { maxFalsePositives: 0, minRecall: 1.0, maxLatencyMs: 1700, minDirAccuracy: 1.0 };

const files = readdirSync(corpusDir).filter((f) => f.endsWith(".json"));
if (files.length === 0) { console.error("No traces in", corpusDir); process.exit(1); }

let fpTraces = 0, fpTotal = 0;          // on-screen traces that wrongly flagged
let recallHit = 0, recallTotal = 0;     // off-screen traces detected
let dirHit = 0, dirTotal = 0;           // direction correct among detected
const latencies = [];
const rows = [];

for (const file of files) {
  const trace = JSON.parse(readFileSync(join(corpusDir, file), "utf8"));
  const { incidents } = replayTrace(trace);
  const exp = trace.expect || {};
  const hard = incidents.filter((i) => i.type === "offscreen");
  const faceInc = incidents.filter((i) => i.type === "facelost");

  let verdict = "ok", detail = "";
  if (!exp.flag) {
    fpTotal++;
    if (incidents.length > 0) { fpTraces++; verdict = "FALSE POSITIVE"; detail = incidents.map((i) => i.type + (i.direction ? "/" + i.direction : "")).join(","); }
  } else {
    recallTotal++;
    const wantFace = exp.type === "facelost";
    const got = wantFace ? faceInc : hard;
    if (got.length > 0) {
      recallHit++;
      const first = got[0];
      const lat = first.atMs - (exp.offFromMs || 0);
      latencies.push(lat);
      if (!wantFace) {
        dirTotal++;
        if (first.direction === exp.direction) dirHit++;
        else { verdict = "WRONG DIR"; detail = `got ${first.direction} want ${exp.direction}`; }
      }
      if (lat > TARGET.maxLatencyMs) { verdict = "SLOW"; detail = `latency ${lat}ms`; }
    } else { verdict = "MISSED"; }
  }
  rows.push({ label: trace.label, expect: exp.flag ? (exp.type || "flag:" + (exp.direction || "")) : "on-screen", incidents: incidents.length, verdict, detail });
}

// Report
const pad = (s, n) => String(s).padEnd(n);
console.log("\n" + pad("scenario", 18) + pad("expected", 14) + pad("inc", 5) + pad("verdict", 16) + "detail");
console.log("-".repeat(70));
for (const r of rows) console.log(pad(r.label, 18) + pad(r.expect, 14) + pad(r.incidents, 5) + pad(r.verdict, 16) + r.detail);

const recall = recallTotal ? recallHit / recallTotal : 1;
const dirAcc = dirTotal ? dirHit / dirTotal : 1;
const meanLat = latencies.length ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length) : 0;
const maxLat = latencies.length ? Math.max(...latencies) : 0;

console.log("\n=== metrics ===");
console.log(`false positives : ${fpTraces}/${fpTotal} on-screen traces flagged`);
console.log(`recall          : ${(recall * 100).toFixed(0)}% (${recallHit}/${recallTotal} off-screen detected)`);
console.log(`direction acc   : ${(dirAcc * 100).toFixed(0)}% (${dirHit}/${dirTotal})`);
console.log(`latency         : mean ${meanLat}ms, max ${maxLat}ms`);

const pass =
  fpTraces <= TARGET.maxFalsePositives &&
  recall >= TARGET.minRecall &&
  dirAcc >= TARGET.minDirAccuracy &&
  maxLat <= TARGET.maxLatencyMs;

console.log(`\nRESULT: ${pass ? "PASS ✅" : "FAIL ❌"}  (targets: FP<=${TARGET.maxFalsePositives}, recall>=${TARGET.minRecall * 100}%, dir>=${TARGET.minDirAccuracy * 100}%, latency<=${TARGET.maxLatencyMs}ms)`);
process.exit(pass ? 0 : 1);
