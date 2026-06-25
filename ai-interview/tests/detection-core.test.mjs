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
