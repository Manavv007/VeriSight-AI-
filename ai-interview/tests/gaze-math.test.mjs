/* Unit tests for gaze-math.mjs — run with: node --test */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  matrixToEuler, ratio, computeIrisGaze, computeBlendshapeGaze, decideLookingAway,
  deriveThresholdsFromCorners, makeSmoother, DEFAULT_THRESHOLDS, EYE,
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

test("decideLookingAway: neutral → not away", () => {
  assert.equal(decideLookingAway(sig({}), base).lookingAway, false);
});

test("decideLookingAway: head yaw beyond threshold → away", () => {
  const r1 = decideLookingAway(sig({ yaw: 25 }), base);
  assert.ok(r1.lookingAway && r1.direction === "right");
  const r2 = decideLookingAway(sig({ yaw: -25 }), base);
  assert.ok(r2.lookingAway && r2.direction === "left");
});

test("decideLookingAway: head pitch down → away down", () => {
  const r = decideLookingAway(sig({ pitch: 20 }), base);
  assert.ok(r.lookingAway && r.direction === "down");
});

test("decideLookingAway: eyes off-center (head still) → away", () => {
  const r = decideLookingAway(sig({ gazeX: 0.6 }), base);
  assert.ok(r.lookingAway && r.direction === "right");
});

test("decideLookingAway: eyes down → away down", () => {
  const r = decideLookingAway(sig({ gazeY: 0.5 }), base);
  assert.ok(r.lookingAway && r.direction === "down", JSON.stringify(r));
});

test("decideLookingAway: eyes up → away up", () => {
  const r = decideLookingAway(sig({ gazeY: -0.5 }), base);
  assert.ok(r.lookingAway && r.direction === "up", JSON.stringify(r));
});

test("decideLookingAway: head pitched up → away up", () => {
  const r = decideLookingAway(sig({ pitch: -16 }), base);
  assert.ok(r.lookingAway && r.direction === "up", JSON.stringify(r));
});

test("decideLookingAway: baseline-relative (offset baseline cancels)", () => {
  // user sits with head turned 20° as their neutral; same pose must NOT flag
  const b = { yaw: 20, pitch: 0, gazeX: 0, gazeY: 0 };
  assert.equal(decideLookingAway(sig({ yaw: 22 }), b).lookingAway, false);
  assert.equal(decideLookingAway(sig({ yaw: 42 }), b).lookingAway, true);
});

test("decideLookingAway: no face → not away", () => {
  assert.equal(decideLookingAway(sig({ facePresent: false, yaw: 90 }), base).lookingAway, false);
});

test("makeSmoother: moving average over window", () => {
  const s = makeSmoother(2);
  assert.equal(s.push(10), 10);
  assert.equal(s.push(20), 15);
  assert.equal(s.push(30), 25); // window drops the 10 → (20+30)/2
  s.reset();
  assert.equal(s.value(), 0);
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
  assert.ok(decision.lookingAway && decision.direction === "right", JSON.stringify(decision));
});

// ---- 4-corner calibration ----
const FLOORS = { YAW_DEG: 10, PITCH: 10, GAZE_X: 0.18, GAZE_Y_DOWN: 0.22 };

test("deriveThresholdsFromCorners: span × margin, centered baseline", () => {
  const corners = [
    { yaw: -20, pitch: -10, gazeX: -0.3, gazeY: -0.2 }, // top-left
    { yaw: 20, pitch: -10, gazeX: 0.3, gazeY: -0.2 },   // top-right
    { yaw: -20, pitch: 10, gazeX: -0.3, gazeY: 0.2 },   // bottom-left
    { yaw: 20, pitch: 10, gazeX: 0.3, gazeY: 0.2 },     // bottom-right
  ];
  const r = deriveThresholdsFromCorners(corners, FLOORS, 1.15);
  assert.ok(near(r.baseline.yaw, 0) && near(r.baseline.pitch, 0));
  assert.ok(near(r.thresholds.YAW_DEG, 23, 1e-6), `YAW=${r.thresholds.YAW_DEG}`);   // 20*1.15
  assert.ok(near(r.thresholds.GAZE_X, 0.345, 1e-6), `GX=${r.thresholds.GAZE_X}`);   // 0.3*1.15
  assert.ok(near(r.thresholds.PITCH_DOWN_DEG, 11.5, 1e-6));                          // 10*1.15
  assert.ok(near(r.thresholds.GAZE_Y_UP, r.thresholds.GAZE_Y_DOWN), "up==down span");
});

test("deriveThresholdsFromCorners: tiny movement clamps to floors", () => {
  const corners = [
    { yaw: 1, pitch: 0.5, gazeX: 0.02, gazeY: 0.01 },
    { yaw: -1, pitch: -0.5, gazeX: -0.02, gazeY: -0.01 },
    { yaw: 0.5, pitch: 1, gazeX: 0.01, gazeY: 0.02 },
    { yaw: -0.5, pitch: -1, gazeX: -0.01, gazeY: -0.02 },
  ];
  const r = deriveThresholdsFromCorners(corners, FLOORS, 1.15);
  assert.equal(r.thresholds.YAW_DEG, 10);
  assert.equal(r.thresholds.GAZE_X, 0.18);
  assert.equal(r.thresholds.GAZE_Y_DOWN, 0.22);
});

test("deriveThresholdsFromCorners: off-axis seating → non-zero baseline, span preserved", () => {
  // candidate sits turned ~15° right; corners centered on yaw=15, span ±20
  const corners = [
    { yaw: -5, pitch: 0, gazeX: 0, gazeY: 0 },
    { yaw: 35, pitch: 0, gazeX: 0, gazeY: 0 },
    { yaw: -5, pitch: 0, gazeX: 0, gazeY: 0 },
    { yaw: 35, pitch: 0, gazeX: 0, gazeY: 0 },
  ];
  const r = deriveThresholdsFromCorners(corners, FLOORS, 1.15);
  assert.ok(near(r.baseline.yaw, 15), `baseYaw=${r.baseline.yaw}`);
  assert.ok(near(r.thresholds.YAW_DEG, 23, 1e-6)); // dev=20 → 23
});

test("deriveThresholdsFromCorners: fewer than 4 → null", () => {
  assert.equal(deriveThresholdsFromCorners([{ yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 }], FLOORS), null);
});
