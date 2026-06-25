/* replay.mjs — replay one trace and print incidents + a level timeline.
   Usage: node tools/replay.mjs corpus/off-left.json
   Optional threshold overrides: --SOFT_YAW=20 --EYE_GAIN_X=55 etc. */
import { readFileSync } from "node:fs";
import { replayTrace } from "../js/detection-core.mjs";

const file = process.argv[2];
if (!file) { console.error("usage: node tools/replay.mjs <trace.json> [--KEY=VAL ...]"); process.exit(1); }

const overrides = {};
for (const a of process.argv.slice(3)) {
  const m = a.match(/^--([A-Za-z_]+)=(.+)$/);
  if (m) (overrides.thresholds ||= {})[m[1]] = Number(m[2]);
}

const trace = JSON.parse(readFileSync(file, "utf8"));
const { incidents, perFrame, finalBaseline } = replayTrace(trace, overrides);

console.log(`trace: ${trace.label}  frames: ${trace.frames.length}  expect:`, trace.expect);
if (overrides.thresholds) console.log("overrides:", overrides.thresholds);

// compress level timeline into runs
let runs = [], cur = null;
for (const f of perFrame) {
  const key = f.level + (f.direction ? ":" + f.direction : "");
  if (!cur || cur.key !== key) { cur = { key, from: f.tMs, to: f.tMs }; runs.push(cur); }
  else cur.to = f.tMs;
}
console.log("\nlevel timeline:");
for (const r of runs) console.log(`  ${String(r.from).padStart(5)}–${String(r.to).padStart(5)}ms  ${r.key}`);

console.log("\nincidents:");
if (incidents.length === 0) console.log("  (none)");
for (const i of incidents) console.log(`  #${i.id} ${i.type}${i.direction ? "/" + i.direction : ""} @${i.atMs}ms conf=${i.confidence}`);

console.log("\nfinal baseline:", finalBaseline);
