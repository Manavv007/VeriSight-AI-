/* Tests for gaze-mapping.mjs (regression PoR) — run with: node --test
   Synthetic forward geometry (eye in 3D, flat screen) generates calibration like
   the app's: 9 straight-head dots + a center head-movement pass. Verifies the fit
   LEARNS head-rotation independence (the reported "tilt head → false positive" bug)
   and stays bounded. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { fitCalibration, predictPoR, decideBoundary } from "../js/gaze-mapping.mjs";

const DEG = Math.PI / 180;
const near = (a, b, eps = 0.08) => Math.abs(a - b) <= eps;
const SCREEN = { X0: -15, W: 30, Y0: 0, H: 20 }, G_TRUE = 28;

function sample(Sx, Sy, pose) {
  const totalYaw = Math.atan2(Sx - pose.ex, pose.ez) / DEG;
  const totalPitch = Math.atan2(Sy - pose.ey, pose.ez) / DEG;
  return {
    headYaw: pose.headYaw, headPitch: pose.headPitch,
    gazeX: (totalYaw - pose.headYaw) / G_TRUE, gazeY: (totalPitch - pose.headPitch) / G_TRUE,
    tx: pose.ex, ty: pose.ey, tz: pose.ez,
    target: { x: (Sx - SCREEN.X0) / SCREEN.W, y: (Sy - SCREEN.Y0) / SCREEN.H },
  };
}
const cm = (nx, ny) => [SCREEN.X0 + nx * SCREEN.W, SCREEN.Y0 + ny * SCREEN.H];

// 9 dots looked at with a straight, still head.
function straightDots() {
  const g = [0, 0.5, 1], out = [];
  for (const ny of g) for (const nx of g) out.push(sample(...cm(nx, ny), { ex: 0, ey: 10, ez: 60, headYaw: 0, headPitch: 0 }));
  return out;
}
// Center fixation while the head rotates/translates (eyes compensate to stay centred).
function centerHeadMotion(n = 36) {
  const [Sx, Sy] = cm(0.5, 0.5), out = [];
  for (let i = 0; i < n; i++) {
    const pose = {
      ex: (i % 5 - 2) * 1.5, ey: 10 + (i % 3 - 1) * 1.5, ez: 60,
      headYaw: (i % 2 ? 1 : -1) * (4 + (i % 6) * 2.2),
      headPitch: (i % 3 - 1) * (5 + (i % 4) * 2.5),
    };
    out.push(sample(Sx, Sy, pose));
  }
  return out;
}
const calibration = () => [...straightDots(), ...centerHeadMotion()];

test("fitCalibration: low residual", () => {
  const m = fitCalibration(calibration());
  assert.ok(m.residual < 5e-3, `residual=${m.residual}`);
});

test("predictPoR: straight-head dots map back to targets", () => {
  const m = fitCalibration(calibration());
  for (const s of straightDots()) {
    const p = predictPoR(m, s);
    assert.ok(near(p.x, s.target.x) && near(p.y, s.target.y), `got ${JSON.stringify(p)} want ${JSON.stringify(s.target)}`);
  }
});

test("HEAD-ROTATION independence: centre look stays centred & bounded under head moves (the bug)", () => {
  const m = fitCalibration(calibration());
  const poses = [
    { ex: 0, ey: 10, ez: 60, headYaw: 0, headPitch: 0 },
    { ex: 0, ey: 10, ez: 60, headYaw: 14, headPitch: 0 },   // turn
    { ex: 0, ey: 10, ez: 60, headYaw: -14, headPitch: 0 },
    { ex: 0, ey: 10, ez: 60, headYaw: 0, headPitch: 13 },   // nod down
    { ex: 0, ey: 10, ez: 60, headYaw: 0, headPitch: -13 },  // nod up
    { ex: 0, ey: 10, ez: 60, headYaw: 10, headPitch: 10 },
  ];
  for (const pose of poses) {
    const p = predictPoR(m, sample(...cm(0.5, 0.5), pose));
    assert.ok(near(p.x, 0.5, 0.12) && near(p.y, 0.5, 0.12), `pose ${JSON.stringify(pose)} → ${JSON.stringify(p)}`);
    assert.equal(decideBoundary(p).level, "none", `pose ${JSON.stringify(pose)} → ${JSON.stringify(p)}`);
  }
});

test("on-screen edge with head turned the other way → still on screen", () => {
  const m = fitCalibration(calibration());
  const p = predictPoR(m, sample(...cm(0.1, 0.5), { ex: 4, ey: 10, ez: 60, headYaw: -12, headPitch: 0 }));
  assert.equal(decideBoundary(p).level, "none", JSON.stringify(p));
});

test("boundary: genuinely off-screen → flagged with direction", () => {
  const m = fitCalibration(calibration());
  const pose = { ex: 0, ey: 10, ez: 60, headYaw: 0, headPitch: 0 };
  const at = (nx, ny) => decideBoundary(predictPoR(m, sample(...cm(nx, ny), pose)));
  assert.equal(at(0.5, 0.5).level, "none");
  assert.equal(at(1.5, 0.5).direction, "right");
  assert.ok(at(1.5, 0.5).level === "hard");
  assert.equal(at(-0.5, 0.5).direction, "left");
  assert.equal(at(0.5, 1.5).direction, "down");
  assert.equal(at(0.5, -0.5).direction, "up");
});

test("predictPoR: output is bounded (no blow-up) under a large head move", () => {
  const m = fitCalibration(calibration());
  const p = predictPoR(m, sample(...cm(0.5, 0.5), { ex: 0, ey: 10, ez: 60, headYaw: 30, headPitch: 25 }));
  assert.ok(p.x >= -1 && p.x <= 2 && p.y >= -1 && p.y <= 2, JSON.stringify(p));
});
