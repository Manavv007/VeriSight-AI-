/* gen-synthetic.mjs — generate labeled synthetic signal traces into ./corpus
   Usage: node tools/gen-synthetic.mjs
   Each trace mimics MediaPipe-derived per-frame signals for one scenario with a
   ground-truth label, so the eval loop has data without a recorded webcam.
   These are approximations (Layer 2); record real traces for ground-truth fidelity. */
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const corpus = join(here, "..", "corpus");
mkdirSync(corpus, { recursive: true });

const FPS = 30;
const DT = Math.round(1000 / FPS);
const noise = (a) => (Math.random() * 2 - 1) * a;

// Build frames: ramp from neutral to target over rampF frames, then hold holdF frames.
function ramped({ target = {}, rampF = 10, holdF = 50, facePresent = true, jitter = true }) {
  const frames = [];
  const keys = ["yaw", "pitch", "gazeX", "gazeY"];
  const tgt = Object.assign({ yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 }, target);
  let t = 0;
  for (let i = 0; i < rampF + holdF; i++) {
    const k = Math.min(1, i / Math.max(1, rampF));
    const f = { tMs: t, facePresent };
    for (const key of keys) {
      const j = jitter ? (key.startsWith("gaze") ? noise(0.02) : noise(0.8)) : 0;
      f[key] = tgt[key] * k + j;
    }
    frames.push(f);
    t += DT;
  }
  return frames;
}

// Slow linear drift (no hold) — for the posture-drift scenario.
function drift({ to = {}, frames: nF = 120 }) {
  const keys = ["yaw", "pitch", "gazeX", "gazeY"];
  const tgt = Object.assign({ yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 }, to);
  const out = [];
  let t = 0;
  for (let i = 0; i < nF; i++) {
    const k = i / (nF - 1);
    const f = { tMs: t, facePresent: true };
    for (const key of keys) f[key] = tgt[key] * k + (key.startsWith("gaze") ? noise(0.01) : noise(0.5));
    out.push(f); t += DT;
  }
  return out;
}

const rampMs = 10 * DT; // ~330ms ground-truth onset for ramped scenarios

const traces = [
  { label: "on-center", expect: { flag: false }, frames: ramped({ target: {}, holdF: 60 }) },
  { label: "on-edge-left", expect: { flag: false }, frames: ramped({ target: { gazeX: -0.4 }, holdF: 60 }) },
  { label: "on-edge-right", expect: { flag: false }, frames: ramped({ target: { gazeX: 0.4 }, holdF: 60 }) },
  { label: "on-read-bottom", expect: { flag: false }, frames: ramped({ target: { pitch: 6, gazeY: 0.3 }, holdF: 60 }) },
  { label: "off-left", expect: { flag: true, direction: "left", offFromMs: rampMs }, frames: ramped({ target: { gazeX: -0.9 }, holdF: 55 }) },
  { label: "off-right", expect: { flag: true, direction: "right", offFromMs: rampMs }, frames: ramped({ target: { gazeX: 0.9 }, holdF: 55 }) },
  { label: "off-left-head", expect: { flag: true, direction: "left", offFromMs: rampMs }, frames: ramped({ target: { yaw: 42 }, holdF: 55 }) }, // head-left = +raw yaw (MediaPipe); default INVERT_YAW → left
  { label: "off-down", expect: { flag: true, direction: "down", offFromMs: rampMs }, frames: ramped({ target: { pitch: 26, gazeY: 0.4 }, holdF: 55 }) },
  { label: "off-up", expect: { flag: true, direction: "up", offFromMs: rampMs }, frames: ramped({ target: { pitch: -36 }, holdF: 55 }) },
  { label: "face-lost", expect: { flag: true, type: "facelost", offFromMs: 0 }, frames: ramped({ target: {}, rampF: 0, holdF: 70, facePresent: false }) },
  { label: "drift-no-flag", expect: { flag: false }, frames: drift({ to: { yaw: 12, pitch: -8 }, frames: 130 }) },
];

for (const tr of traces) {
  const out = { label: tr.label, expect: tr.expect, meta: { baseline: { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 }, thresholds: null, synthetic: true }, frames: tr.frames };
  writeFileSync(join(corpus, tr.label + ".json"), JSON.stringify(out));
}
console.log(`Wrote ${traces.length} synthetic traces to ${corpus}`);
