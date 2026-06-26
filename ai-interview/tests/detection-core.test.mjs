/* Tests for detection-core.mjs — run with: node --test */
import { test } from "node:test";
import assert from "node:assert/strict";
import { createDetector } from "../js/detection-core.mjs";

const DT = 33;
// Feed a constant (or function) frame for durationMs; returns the detector.
function feed(det, frame, durationMs, startMs = 0) {
  let t = startMs;
  const end = startMs + durationMs;
  while (t <= end) {
    det.process(typeof frame === "function" ? frame(t) : frame, t);
    t += DT;
  }
  return det;
}

test("on-screen center → no incidents", () => {
  const det = createDetector();
  feed(det, { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0, facePresent: true }, 2000);
  assert.equal(det.getIncidents().length, 0);
});

test("soft zone (eyes -0.5) → no logged incident", () => {
  const det = createDetector();
  feed(det, { yaw: 0, pitch: 0, gazeX: -0.5, gazeY: 0, facePresent: true }, 2000); // ~-25 → soft
  assert.equal(det.getIncidents().length, 0);
});

test("hard look-left held > sustain → one offscreen/left incident", () => {
  const det = createDetector();
  feed(det, { yaw: 0, pitch: 0, gazeX: -0.9, gazeY: 0, facePresent: true }, 1600); // ~-45 → hard
  const inc = det.getIncidents();
  assert.equal(inc.length, 1);
  assert.equal(inc[0].type, "offscreen");
  assert.equal(inc[0].direction, "left");
});

test("hard but brief (< sustain) → no incident", () => {
  const det = createDetector();
  feed(det, { yaw: 0, pitch: 0, gazeX: -0.9, gazeY: 0, facePresent: true }, 600); // < AWAY_FLAG_MS 1000
  assert.equal(det.getIncidents().length, 0);
});

test("look down (pitch 26) → offscreen/down", () => {
  const det = createDetector();
  feed(det, { yaw: 0, pitch: 26, gazeX: 0, gazeY: 0.3, facePresent: true }, 1600);
  const inc = det.getIncidents();
  assert.ok(inc.length === 1 && inc[0].direction === "down", JSON.stringify(inc));
});

test("face lost held > 1.5s → facelost incident", () => {
  const det = createDetector();
  feed(det, { facePresent: false }, 1800);
  const inc = det.getIncidents();
  assert.ok(inc.length === 1 && inc[0].type === "facelost", JSON.stringify(inc));
});

test("slow head drift with eyes centred → no incident (adaptation)", () => {
  const det = createDetector();
  // yaw drifts 0 → 14 over 4s, gaze centred
  feed(det, (t) => ({ yaw: 14 * (t / 4000), pitch: 0, gazeX: 0, gazeY: 0, facePresent: true }), 4000);
  assert.equal(det.getIncidents().length, 0);
});

test("recovery: hard then back to centre stays a single incident", () => {
  const det = createDetector();
  feed(det, { yaw: 0, pitch: 0, gazeX: 0.9, gazeY: 0, facePresent: true }, 1600, 0);   // off-right
  feed(det, { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0, facePresent: true }, 1000, 1700);  // back on screen
  const inc = det.getIncidents();
  assert.ok(inc.length === 1 && inc[0].direction === "right", JSON.stringify(inc));
});

// ---- point-of-regard (PoR) mode via a calibrated gaze model ----
import { fitCalibration } from "../js/gaze-mapping.mjs";
const SCREEN = { X0: -15, W: 30, Y0: 0, H: 20 }, G_TRUE = 28, DEG = Math.PI / 180;
function look(Sx, Sy, pose) {
  const totalYaw = Math.atan2(Sx - pose.ex, pose.ez) / DEG;
  const totalPitch = Math.atan2(Sy - pose.ey, pose.ez) / DEG;
  return {
    headYaw: pose.headYaw, headPitch: pose.headPitch,
    gazeX: (totalYaw - pose.headYaw) / G_TRUE, gazeY: (totalPitch - pose.headPitch) / G_TRUE,
    tx: pose.ex, ty: pose.ey, tz: pose.ez,
  };
}
function frame(s) { return { yaw: s.headYaw, pitch: s.headPitch, gazeX: s.gazeX, gazeY: s.gazeY, tx: s.tx, ty: s.ty, tz: s.tz, facePresent: true }; }
function poRModel() {
  const grid = [0, 0.5, 1], samples = [];
  let k = 0;
  for (const ny of grid) for (const nx of grid) {
    const pose = { ex: (k % 3 - 1) * 3, ey: 10, ez: 50, headYaw: (k % 3 - 1) * 5, headPitch: 0 };
    const s = look(SCREEN.X0 + nx * SCREEN.W, SCREEN.Y0 + ny * SCREEN.H, pose);
    samples.push(Object.assign(s, { target: { x: nx, y: ny } }));
    k++;
  }
  return fitCalibration(samples);
}

test("PoR mode: gaze off the right edge → offscreen/right incident", () => {
  const det = createDetector({ poRModel: poRModel() });
  const pose = { ex: 0, ey: 10, ez: 50, headYaw: 0, headPitch: 0 };
  const off = frame(look(SCREEN.X0 + 1.4 * SCREEN.W, SCREEN.Y0 + 0.5 * SCREEN.H, pose)); // x≈1.4 → off right
  feed(det, off, 1600);
  const inc = det.getIncidents();
  assert.ok(inc.length === 1 && inc[0].type === "offscreen" && inc[0].direction === "right", JSON.stringify(inc));
});

test("PoR mode: on-screen gaze (even head-turned) → no incident", () => {
  const det = createDetector({ poRModel: poRModel() });
  // looking at on-screen centre but head turned hard the other way
  const onCentre = frame(look(SCREEN.X0 + 0.5 * SCREEN.W, SCREEN.Y0 + 0.5 * SCREEN.H, { ex: 8, ey: 10, ez: 50, headYaw: -18, headPitch: 0 }));
  feed(det, onCentre, 2000);
  assert.equal(det.getIncidents().length, 0);
});
