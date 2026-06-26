/* Tests for gaze-mapping.mjs — run with: node --test
   Uses a synthetic forward geometry model (eye in 3D, flat screen at camera plane)
   to PROVE the point-of-regard is head/distance independent. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { fitCalibration, predictPoR, decideBoundary } from "../js/gaze-mapping.mjs";

const DEG = Math.PI / 180;
const near = (a, b, eps = 1e-2) => Math.abs(a - b) <= eps;

// Synthetic screen (cm, at camera plane z=0): X in [-15,15], Y in [0,20].
const SCREEN = { X0: -15, W: 30, Y0: 0, H: 20 };
const G_TRUE = 28; // true eye gain (deg per gaze unit)

// Build a sample: eye at pose {ex,ey,ez, headYaw, headPitch} looking at screen point (Sx,Sy) cm.
function sample(Sx, Sy, pose) {
  const { ex, ey, ez, headYaw, headPitch } = pose;
  const totalYaw = Math.atan2(Sx - ex, ez) / DEG;     // degrees
  const totalPitch = Math.atan2(Sy - ey, ez) / DEG;
  return {
    headYaw, headPitch,
    gazeX: (totalYaw - headYaw) / G_TRUE,
    gazeY: (totalPitch - headPitch) / G_TRUE,
    tx: ex, ty: ey, tz: ez,
    target: { x: (Sx - SCREEN.X0) / SCREEN.W, y: (Sy - SCREEN.Y0) / SCREEN.H },
  };
}
const screenCm = (nx, ny) => [SCREEN.X0 + nx * SCREEN.W, SCREEN.Y0 + ny * SCREEN.H];

// 9-point calibration, each captured at a slightly different head pose.
function calibrationSet() {
  const grid = [0, 0.5, 1];
  const out = [];
  let k = 0;
  for (const ny of grid) for (const nx of grid) {
    const [Sx, Sy] = screenCm(nx, ny);
    const pose = { ex: (k % 3 - 1) * 3, ey: 10 + (k % 2) * 2, ez: 48 + (k % 3) * 4, headYaw: (k % 3 - 1) * 6, headPitch: (k % 2) * 4 };
    out.push(sample(Sx, Sy, pose));
    k++;
  }
  return out;
}

test("fitCalibration: recovers eye-gain and low residual", () => {
  const model = fitCalibration(calibrationSet());
  assert.ok(model.residual < 1e-3, `residual=${model.residual}`);
  assert.ok(Math.abs(model.eyeGain - G_TRUE) <= 3, `eyeGain=${model.eyeGain}`);
});

test("predictPoR: calibration points map back to their targets", () => {
  const set = calibrationSet();
  const model = fitCalibration(set);
  for (const s of set) {
    const p = predictPoR(model, s);
    assert.ok(near(p.x, s.target.x) && near(p.y, s.target.y), `got ${JSON.stringify(p)} want ${JSON.stringify(s.target)}`);
  }
});

test("HEAD-POSITION independence: same on-screen point from different head poses → same PoR", () => {
  const model = fitCalibration(calibrationSet());
  const [Sx, Sy] = screenCm(0.5, 0.5); // screen centre
  const poses = [
    { ex: 0, ey: 10, ez: 50, headYaw: 0, headPitch: 0 },
    { ex: -8, ey: 8, ez: 50, headYaw: 12, headPitch: 0 },   // head shifted/turned left
    { ex: 9, ey: 12, ez: 50, headYaw: -15, headPitch: 3 },  // head shifted/turned right
  ];
  for (const pose of poses) {
    const p = predictPoR(model, sample(Sx, Sy, pose));
    assert.ok(near(p.x, 0.5, 0.04) && near(p.y, 0.5, 0.04), `pose ${JSON.stringify(pose)} → ${JSON.stringify(p)}`);
  }
});

test("DISTANCE independence: same on-screen point near vs far → same PoR", () => {
  const model = fitCalibration(calibrationSet());
  const [Sx, Sy] = screenCm(0.75, 0.25);
  const nearP = predictPoR(model, sample(Sx, Sy, { ex: 0, ey: 10, ez: 35, headYaw: 0, headPitch: 0 }));
  const farP = predictPoR(model, sample(Sx, Sy, { ex: 0, ey: 10, ez: 70, headYaw: 0, headPitch: 0 }));
  assert.ok(near(nearP.x, 0.75, 0.05) && near(farP.x, 0.75, 0.05), `near ${JSON.stringify(nearP)} far ${JSON.stringify(farP)}`);
  assert.ok(near(nearP.y, 0.25, 0.05) && near(farP.y, 0.25, 0.05));
});

test("boundary: on-screen centre → none; off right/left/up/down → flagged", () => {
  const model = fitCalibration(calibrationSet());
  const pose = { ex: 0, ey: 10, ez: 50, headYaw: 0, headPitch: 0 };
  const at = (nx, ny) => decideBoundary(predictPoR(model, sample(SCREEN.X0 + nx * SCREEN.W, SCREEN.Y0 + ny * SCREEN.H, pose)));
  assert.equal(at(0.5, 0.5).level, "none");
  assert.equal(at(1.4, 0.5).direction, "right");
  assert.ok(at(1.4, 0.5).level === "hard");
  assert.equal(at(-0.4, 0.5).direction, "left");
  assert.equal(at(0.5, -0.4).direction, "up");
  assert.equal(at(0.5, 1.4).direction, "down");
});

test("boundary: head turned but still on-screen → NOT flagged (the key fix)", () => {
  const model = fitCalibration(calibrationSet());
  // user looks at the on-screen left edge but with the head turned the other way
  const [Sx, Sy] = screenCm(0.05, 0.5);
  const d = decideBoundary(predictPoR(model, sample(Sx, Sy, { ex: 6, ey: 10, ez: 50, headYaw: -18, headPitch: 0 })));
  assert.equal(d.level, "none", JSON.stringify(d));
});
