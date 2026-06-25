/* Unit tests for gaze-math.mjs — run with: node --test */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  matrixToEuler, ratio, computeIrisGaze, computeBlendshapeGaze, decideLookingAway, computeCombined,
  deriveThresholdsFromCorners, adaptBaseline, makeSmoother, DEFAULT_THRESHOLDS, EYE,
} from "../js/gaze-math.mjs";

const DEG = Math.PI / 180;
const near = (a, b, eps = 1e-3) => Math.abs(a - b) <= eps;

// Build a column-major 16-float matrix from a row-major 3x3 rotation, optional scale.
function colMajor(R, s = 1) {
  // R is [[R00,R01,R02],[R10,R11,R12],[R20,R21,R22]]
  return [
    R[0][0] * s, R[1][0] * s, R[2][0] * s, 0,
    R[0][1] * s, R[1][1] * s, R[2][1] * s, 0,
    R[0][2] * s, R[1][2] * s, R[2][2] * s, 0,
    0, 0, 0, 1,
  ];
}
const Ry = (t) => [[Math.cos(t), 0, Math.sin(t)], [0, 1, 0], [-Math.sin(t), 0, Math.cos(t)]];
const Rx = (t) => [[1, 0, 0], [0, Math.cos(t), -Math.sin(t)], [0, Math.sin(t), Math.cos(t)]];
const Rz = (t) => [[Math.cos(t), -Math.sin(t), 0], [Math.sin(t), Math.cos(t), 0], [0, 0, 1]];

test("matrixToEuler: identity → zero angles", () => {
  const e = matrixToEuler([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  assert.ok(near(e.yaw, 0) && near(e.pitch, 0) && near(e.roll, 0));
});

test("matrixToEuler: yaw 30° about Y", () => {
  const e = matrixToEuler(colMajor(Ry(30 * DEG)));
  assert.ok(near(e.yaw, 30, 1e-2), `yaw=${e.yaw}`);
  assert.ok(near(e.pitch, 0, 1e-2) && near(e.roll, 0, 1e-2));
});

test("matrixToEuler: pitch 20° about X", () => {
  const e = matrixToEuler(colMajor(Rx(20 * DEG)));
  assert.ok(near(e.pitch, 20, 1e-2), `pitch=${e.pitch}`);
  assert.ok(near(e.yaw, 0, 1e-2) && near(e.roll, 0, 1e-2));
});

test("matrixToEuler: roll 15° about Z", () => {
  const e = matrixToEuler(colMajor(Rz(15 * DEG)));
  assert.ok(near(e.roll, 15, 1e-2), `roll=${e.roll}`);
  assert.ok(near(e.yaw, 0, 1e-2) && near(e.pitch, 0, 1e-2));
});

test("matrixToEuler: uniform scale does not distort angles", () => {
  const e = matrixToEuler(colMajor(Ry(30 * DEG), 2.5));
  assert.ok(near(e.yaw, 30, 1e-2), `yaw=${e.yaw}`);
});

test("matrixToEuler: invalid input → valid=false", () => {
  assert.equal(matrixToEuler(null).valid, false);
  assert.equal(matrixToEuler([1, 2, 3]).valid, false);
});

test("ratio: endpoints and midpoint", () => {
  assert.ok(near(ratio(0.5, 0.4, 0.6), 0.5));
  assert.ok(near(ratio(0.4, 0.4, 0.6), 0));
  assert.ok(near(ratio(0.6, 0.4, 0.6), 1));
  assert.equal(ratio(5, 3, 3), 0.5); // degenerate → centered
});

function makeLandmarks() {
  const lm = [];
  for (let i = 0; i < 478; i++) lm.push({ x: 0.5, y: 0.5, z: 0 });
  // set both eyes centered: outer=0.4, inner=0.6, top=0.4, bot=0.6, iris=0.5
  for (const k of ["a", "b"]) {
    const c = EYE[k];
    lm[c.hOuter] = { x: 0.4, y: 0.5 };
    lm[c.hInner] = { x: 0.6, y: 0.5 };
    lm[c.vTop] = { x: 0.5, y: 0.4 };
    lm[c.vBot] = { x: 0.5, y: 0.6 };
    lm[c.iris] = { x: 0.5, y: 0.5 };
  }
  return lm;
}

test("computeIrisGaze: centered → ~0,0", () => {
  const g = computeIrisGaze(makeLandmarks());
  assert.ok(g.valid && near(g.gazeX, 0) && near(g.gazeY, 0));
});

test("computeIrisGaze: iris toward inner corner → gazeX +1", () => {
  const lm = makeLandmarks();
  lm[EYE.a.iris] = { x: 0.6, y: 0.5 };
  lm[EYE.b.iris] = { x: 0.6, y: 0.5 };
  const g = computeIrisGaze(lm);
  assert.ok(near(g.gazeX, 1, 1e-6), `gazeX=${g.gazeX}`);
});

test("computeIrisGaze: iris toward lower lid → gazeY +1", () => {
  const lm = makeLandmarks();
  lm[EYE.a.iris] = { x: 0.5, y: 0.6 };
  lm[EYE.b.iris] = { x: 0.5, y: 0.6 };
  const g = computeIrisGaze(lm);
  assert.ok(near(g.gazeY, 1, 1e-6), `gazeY=${g.gazeY}`);
});

test("computeIrisGaze: too few landmarks → valid=false", () => {
  assert.equal(computeIrisGaze([{ x: 0, y: 0 }]).valid, false);
});

const base = { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 };
const sig = (o) => Object.assign({ facePresent: true, yaw: 0, pitch: 0, roll: 0, gazeX: 0, gazeY: 0 }, o);
// Sign-neutral cfg for logic tests written as "yaw+ = right" (the default now inverts yaw).
const NOINV = Object.assign({}, DEFAULT_THRESHOLDS, { INVERT_YAW: false });

test("decideLookingAway: neutral → level none", () => {
  assert.equal(decideLookingAway(sig({}), base).level, "none");
});

test("decideLookingAway: strong head yaw → hard, correct direction", () => {
  const r1 = decideLookingAway(sig({ yaw: 40 }), base, NOINV);
  assert.ok(r1.level === "hard" && r1.direction === "right", JSON.stringify(r1));
  assert.ok(r1.confidence >= 0.8);
  const r2 = decideLookingAway(sig({ yaw: -40 }), base, NOINV);
  assert.ok(r2.level === "hard" && r2.direction === "left");
});

test("decideLookingAway: INVERT_YAW (default) flips head-yaw sign", () => {
  // MediaPipe head-right = negative raw yaw; default inverts so it reads "right".
  assert.equal(decideLookingAway(sig({ yaw: -40 }), base).direction, "right");
  assert.equal(decideLookingAway(sig({ yaw: 40 }), base).direction, "left");
});

test("decideLookingAway: moderate head yaw → soft (possible) with mid confidence", () => {
  const r = decideLookingAway(sig({ yaw: 25 }), base, NOINV);
  assert.ok(r.level === "soft" && r.direction === "right", JSON.stringify(r));
  assert.ok(r.confidence >= 0.5 && r.confidence < 0.8, `conf=${r.confidence}`);
});

test("decideLookingAway: head pitch down/up → hard down/up", () => {
  assert.equal(decideLookingAway(sig({ pitch: 35 }), base).direction, "down");
  assert.equal(decideLookingAway(sig({ pitch: 35 }), base).level, "hard");
  assert.equal(decideLookingAway(sig({ pitch: -35 }), base).direction, "up");
});

test("decideLookingAway: down needs less tilt than up (asymmetric thresholds)", () => {
  // 22° down → hard (down hard = 20); 22° up → only soft (up hard = 30)
  const d = decideLookingAway(sig({ pitch: 22 }), base);
  assert.ok(d.direction === "down" && d.level === "hard", JSON.stringify(d));
  const u = decideLookingAway(sig({ pitch: -22 }), base);
  assert.ok(u.direction === "up" && u.level === "soft", JSON.stringify(u));
});

test("decideLookingAway: eyes-only (head still) reach hard at extreme", () => {
  const r = decideLookingAway(sig({ gazeX: 1 }), base); // 40*1 = 40 >= hard 36
  assert.ok(r.level === "hard" && r.direction === "right", JSON.stringify(r));
  const soft = decideLookingAway(sig({ gazeX: 0.6 }), base); // 24 → soft
  assert.equal(soft.level, "soft");
});

test("decideLookingAway: eyes-only down/up via combined pitch", () => {
  assert.equal(decideLookingAway(sig({ gazeY: 1 }), base).direction, "down"); // 30 >= hard 30
  assert.equal(decideLookingAway(sig({ gazeY: 1 }), base).level, "hard");
  assert.equal(decideLookingAway(sig({ gazeY: -1 }), base).direction, "up");
});

test("decideLookingAway: combined head+eye reinforce", () => {
  // head 15° right + eyes 0.6 right → 15 + 40*w*0.6; w=eyeWeight(15)=0.4 → 15+9.6=24.6 → soft,
  // and pushing eyes further crosses hard
  const r = decideLookingAway(sig({ yaw: 20, gazeX: 0.8 }), base, NOINV);
  assert.ok(r.level !== "none" && r.direction === "right", JSON.stringify(r));
});

test("decideLookingAway: head-gated eye (#3) — far head turn suppresses eye term", () => {
  // head exactly at gate (25°) → eye weight 0 → eyes can't add; combined = 25 → soft (not hard)
  const r = decideLookingAway(sig({ yaw: 25, gazeX: 1 }), base);
  assert.equal(r.level, "soft", JSON.stringify(r));
});

test("decideLookingAway: baseline-relative (off-axis neutral cancels)", () => {
  const b = { yaw: 20, pitch: 0, gazeX: 0, gazeY: 0 };
  assert.equal(decideLookingAway(sig({ yaw: 22 }), b).level, "none");
  assert.equal(decideLookingAway(sig({ yaw: 60 }), b).level, "hard");
});

test("decideLookingAway: no face → level none", () => {
  assert.equal(decideLookingAway(sig({ facePresent: false, yaw: 90 }), base).level, "none");
});

test("computeCombined: eye weight gates to 0 at HEAD_GATE_DEG", () => {
  const c = computeCombined(sig({ yaw: 25, gazeX: 1 }), base, NOINV);
  assert.ok(near(c.eyeWeightX, 0), `wX=${c.eyeWeightX}`);
  assert.ok(near(c.yaw, 25), `yaw=${c.yaw}`);
});

test("computeCombined: full eye weight near neutral head", () => {
  const c = computeCombined(sig({ yaw: 0, gazeX: 0.5 }), base, NOINV);
  assert.ok(near(c.eyeWeightX, 1));
  assert.ok(near(c.yaw, NOINV.EYE_GAIN_X * 0.5), `yaw=${c.yaw}`); // gain * weight * gaze
});

test("makeSmoother: moving average over window", () => {
  const s = makeSmoother(2);
  assert.equal(s.push(10), 10);
  assert.equal(s.push(20), 15);
  assert.equal(s.push(30), 25); // window drops the 10 → (20+30)/2
  s.reset();
  assert.equal(s.value(), 0);
});

test("adaptBaseline: single EMA step moves toward signal by alpha", () => {
  const b0 = { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 };
  const sigv = { yaw: 12, pitch: -10, gazeX: 0.5, gazeY: 0.4 };
  const b1 = adaptBaseline(b0, sigv, 0.5);
  assert.ok(near(b1.yaw, 6) && near(b1.pitch, -5) && near(b1.gazeX, 0.25) && near(b1.gazeY, 0.2));
});

test("adaptBaseline: gazeAlpha=0 freezes eye baseline, head still adapts", () => {
  // a sustained eye glance must NOT be absorbed (eye baseline stays put)
  let b = { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 };
  const look = { yaw: 0, pitch: 0, gazeX: 0.8, gazeY: 0.5 };
  for (let i = 0; i < 200; i++) b = adaptBaseline(b, look, 0.04, 0);
  assert.ok(near(b.gazeX, 0) && near(b.gazeY, 0), `eye drifted: ${JSON.stringify(b)}`);
  // head still converges
  let h = { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 };
  const rest = { yaw: 14, pitch: -9, gazeX: 0, gazeY: 0 };
  for (let i = 0; i < 200; i++) h = adaptBaseline(h, rest, 0.04, 0);
  assert.ok(near(h.yaw, 14, 0.2) && near(h.pitch, -9, 0.2), JSON.stringify(h));
});

test("adaptBaseline: converges toward a settled resting pose", () => {
  // simulate a stale baseline (0) while the user actually rests at yaw 12, pitch -11.
  let b = { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 };
  const rest = { yaw: 12, pitch: -11, gazeX: 0.47, gazeY: 0.46 };
  for (let i = 0; i < 120; i++) b = adaptBaseline(b, rest, 0.04); // ~5s at 24fps
  assert.ok(near(b.yaw, 12, 0.5) && near(b.pitch, -11, 0.5), JSON.stringify(b));
  // after convergence, the resting pose reads as on-screen
  const d = decideLookingAway({ facePresent: true, ...rest, roll: 0 }, b, DEFAULT_THRESHOLDS);
  assert.equal(d.level, "none");
});

// ---- blendshape eye gaze ----
function cats(obj) {
  return Object.entries(obj).map(([categoryName, score]) => ({ categoryName, score }));
}

test("computeBlendshapeGaze: neutral → ~0,0", () => {
  const g = computeBlendshapeGaze(cats({
    eyeLookOutRight: 0, eyeLookInLeft: 0, eyeLookOutLeft: 0, eyeLookInRight: 0,
    eyeLookUpLeft: 0, eyeLookUpRight: 0, eyeLookDownLeft: 0, eyeLookDownRight: 0,
  }));
  assert.ok(g.valid && near(g.gazeX, 0) && near(g.gazeY, 0));
});

test("computeBlendshapeGaze: looking right → gazeX positive, both eyes reinforce", () => {
  const g = computeBlendshapeGaze(cats({
    eyeLookOutRight: 0.8, eyeLookInLeft: 0.8, eyeLookOutLeft: 0, eyeLookInRight: 0,
    eyeLookUpLeft: 0, eyeLookUpRight: 0, eyeLookDownLeft: 0, eyeLookDownRight: 0,
  }));
  assert.ok(g.valid && near(g.gazeX, 0.8, 1e-6), `gazeX=${g.gazeX}`);
  assert.ok(near(g.gazeY, 0));
});

test("computeBlendshapeGaze: looking left → gazeX negative", () => {
  const g = computeBlendshapeGaze(cats({
    eyeLookOutRight: 0, eyeLookInLeft: 0, eyeLookOutLeft: 0.7, eyeLookInRight: 0.7,
    eyeLookUpLeft: 0, eyeLookUpRight: 0, eyeLookDownLeft: 0, eyeLookDownRight: 0,
  }));
  assert.ok(near(g.gazeX, -0.7, 1e-6), `gazeX=${g.gazeX}`);
});

test("computeBlendshapeGaze: looking down → gazeY positive", () => {
  const g = computeBlendshapeGaze(cats({
    eyeLookOutRight: 0, eyeLookInLeft: 0, eyeLookOutLeft: 0, eyeLookInRight: 0,
    eyeLookUpLeft: 0, eyeLookUpRight: 0, eyeLookDownLeft: 0.6, eyeLookDownRight: 0.6,
  }));
  assert.ok(near(g.gazeY, 0.6, 1e-6), `gazeY=${g.gazeY}`);
});

test("computeBlendshapeGaze: looking up → gazeY negative", () => {
  const g = computeBlendshapeGaze(cats({
    eyeLookOutRight: 0, eyeLookInLeft: 0, eyeLookOutLeft: 0, eyeLookInRight: 0,
    eyeLookUpLeft: 0.5, eyeLookUpRight: 0.5, eyeLookDownLeft: 0, eyeLookDownRight: 0,
  }));
  assert.ok(near(g.gazeY, -0.5, 1e-6), `gazeY=${g.gazeY}`);
});

test("computeBlendshapeGaze: missing categories → valid=false", () => {
  assert.equal(computeBlendshapeGaze([]).valid, false);
  assert.equal(computeBlendshapeGaze(cats({ jawOpen: 0.5 })).valid, false);
});

test("computeBlendshapeGaze: off-screen eye glance crosses default threshold", () => {
  const g = computeBlendshapeGaze(cats({
    eyeLookOutRight: 0.7, eyeLookInLeft: 0.7, eyeLookOutLeft: 0, eyeLookInRight: 0,
    eyeLookUpLeft: 0, eyeLookUpRight: 0, eyeLookDownLeft: 0, eyeLookDownRight: 0,
  }));
  const decision = decideLookingAway(sig({ gazeX: g.gazeX }), base);
  assert.ok(decision.level !== "none" && decision.direction === "right", JSON.stringify(decision));
});

// ---- 4-corner calibration (combined-gaze soft/hard derivation) ----
const FLOORS = { SOFT_YAW: 15, TOL_YAW: 8, SOFT_PITCH: 14, TOL_PITCH: 8 };
// head-only corners (gazeX/Y = 0) so combined == head deviation, easy to verify
const headCorners = (yawSpan, pitchSpan) => [
  { yaw: -yawSpan, pitch: -pitchSpan, gazeX: 0, gazeY: 0 },
  { yaw: yawSpan, pitch: -pitchSpan, gazeX: 0, gazeY: 0 },
  { yaw: -yawSpan, pitch: pitchSpan, gazeX: 0, gazeY: 0 },
  { yaw: yawSpan, pitch: pitchSpan, gazeX: 0, gazeY: 0 },
];

test("deriveThresholdsFromCorners: combined span × soft/hard margins", () => {
  const r = deriveThresholdsFromCorners(headCorners(20, 10), DEFAULT_THRESHOLDS, FLOORS, 1.1, 1.4);
  assert.ok(near(r.baseline.yaw, 0) && near(r.baseline.pitch, 0));
  assert.ok(near(r.thresholds.SOFT_YAW, 22, 1e-6), `SOFT_YAW=${r.thresholds.SOFT_YAW}`);   // 20*1.1
  assert.ok(near(r.thresholds.TOL_YAW, 8, 1e-6), `TOL_YAW=${r.thresholds.TOL_YAW}`);        // max(20*0.3,8)=8
  assert.ok(near(r.thresholds.SOFT_PITCH_DOWN, 14, 1e-6));                                   // max(10*1.1,14)=14
  assert.equal(r.thresholds.SOFT_PITCH_UP, r.thresholds.SOFT_PITCH_DOWN);
});

test("deriveThresholdsFromCorners: tiny movement clamps to floors", () => {
  const r = deriveThresholdsFromCorners(headCorners(1, 0.5), DEFAULT_THRESHOLDS, FLOORS, 1.1, 1.4);
  assert.equal(r.thresholds.SOFT_YAW, 15);
  assert.equal(r.thresholds.TOL_YAW, 8);
  assert.equal(r.thresholds.SOFT_PITCH_DOWN, 14);
});

test("deriveThresholdsFromCorners: off-axis seating → non-zero baseline, span preserved", () => {
  const corners = [
    { yaw: -5, pitch: 0, gazeX: 0, gazeY: 0 },
    { yaw: 35, pitch: 0, gazeX: 0, gazeY: 0 },
    { yaw: -5, pitch: 0, gazeX: 0, gazeY: 0 },
    { yaw: 35, pitch: 0, gazeX: 0, gazeY: 0 },
  ];
  const r = deriveThresholdsFromCorners(corners, DEFAULT_THRESHOLDS, FLOORS, 1.1, 1.4);
  assert.ok(near(r.baseline.yaw, 15), `baseYaw=${r.baseline.yaw}`);
  assert.ok(near(r.thresholds.SOFT_YAW, 22, 1e-6)); // dev=20 → 22
});

test("deriveThresholdsFromCorners: eye movement contributes to combined span", () => {
  // corners reached with eyes (head still): gazeX ±0.5 at full eye weight → 40*0.5 = 20 combined
  const corners = [
    { yaw: 0, pitch: 0, gazeX: -0.5, gazeY: 0 },
    { yaw: 0, pitch: 0, gazeX: 0.5, gazeY: 0 },
    { yaw: 0, pitch: 0, gazeX: -0.5, gazeY: 0 },
    { yaw: 0, pitch: 0, gazeX: 0.5, gazeY: 0 },
  ];
  const r = deriveThresholdsFromCorners(corners, DEFAULT_THRESHOLDS, FLOORS, 1.1, 1.4);
  const expectedMax = DEFAULT_THRESHOLDS.EYE_GAIN_X * 0.5; // gain * weight(1) * 0.5
  assert.ok(near(r.thresholds.SOFT_YAW, Math.max(expectedMax * 1.1, 15), 1e-6), `SOFT_YAW=${r.thresholds.SOFT_YAW}`);
});

test("deriveThresholdsFromCorners: fewer than 4 → null", () => {
  assert.equal(deriveThresholdsFromCorners([{ yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 }], DEFAULT_THRESHOLDS, FLOORS), null);
});
