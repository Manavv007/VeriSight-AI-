/* analyze.mjs — inspect real traces: combined-gaze ranges, levels, incidents.
   Usage: node tools/analyze.mjs corpus/real */
import { readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { createDetector } from "../js/detection-core.mjs";

const dir = resolve(process.argv[2] || "corpus/real");
const files = readdirSync(dir).filter((f) => f.endsWith(".json"));

// optional overrides: --INVERT_YAW=1 --HEAD_GATE_DEG=50 --SOFT_YAW=22 ...
const ov = {};
for (const a of process.argv.slice(3)) {
  const m = a.match(/^--([A-Za-z_]+)=(.+)$/);
  if (m) ov[m[1]] = m[2] === "true" ? true : m[2] === "false" ? false : Number(m[2]);
}
if (Object.keys(ov).length) console.log("overrides:", ov);

for (const file of files) {
  const t = JSON.parse(readFileSync(join(dir, file), "utf8"));
  const det = createDetector({ baseline: t.meta && t.meta.baseline, thresholds: ov });
  const b = (t.meta && t.meta.baseline) || { yaw: 0, gazeX: 0 };
  const cy = [], cp = [], dyaw = [], dgx = [];
  const levels = { none: 0, soft: 0, hard: 0 };
  let faceLostFrames = 0;
  for (const f of t.frames) {
    const r = det.process(f, f.tMs);
    if (!r.facePresent) { faceLostFrames++; continue; }
    cy.push(r.combinedYaw); cp.push(r.combinedPitch);
    dyaw.push(f.yaw - b.yaw); dgx.push(f.gazeX - b.gazeX);
    levels[r.level]++;
  }
  const stat = (a) => a.length ? { min: Math.min(...a).toFixed(1), max: Math.max(...a).toFixed(1), mean: (a.reduce((x, y) => x + y, 0) / a.length).toFixed(1) } : {};
  const inc = det.getIncidents();
  console.log(`\n=== ${t.label}  (expect ${JSON.stringify(t.expect)}) ===`);
  console.log(`frames ${t.frames.length}, faceLost ${faceLostFrames}`);
  console.log(`raw dHeadYaw `, stat(dyaw));
  console.log(`raw dGazeX   `, stat(dgx));
  console.log(`combinedYaw  `, stat(cy));
  console.log(`combinedPitch`, stat(cp));
  console.log(`levels`, levels);
  console.log(`incidents`, inc.map((i) => `${i.type}${i.direction ? "/" + i.direction : ""}@${i.atMs}ms`).join(", ") || "(none)");
}
