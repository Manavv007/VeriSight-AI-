/* Tests for gaze-mapping.mjs (regression PoR) — run with: node --test
   Synthetic forward geometry (eye in 3D, flat screen) generates calibration like
   the app's: 9 straight-head dots + a center head-movement pass. Verifies the fit
   LEARNS head-rotation independence (the reported "tilt head → false positive" bug)
   and stays bounded. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { fitCalibration, predictPoR, decideBoundary, decidePoR, decideHead, DEFAULT_BOUNDARY, createRecenterState, recenterApply } from "../js/gaze-mapping.mjs";

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
// Head motion (rotation + translation + distance) while fixating a given target.
function headMotionAt(nx, ny, n = 18) {
  const [Sx, Sy] = cm(nx, ny), out = [];
  for (let i = 0; i < n; i++) {
    const pose = {
      ex: (i % 5 - 2) * 2, ey: 10 + (i % 3 - 1) * 2, ez: 50 + (i % 5) * 5, // distance 50–70
      headYaw: (i % 2 ? 1 : -1) * (4 + (i % 6) * 2.5),
      headPitch: (i % 3 - 1) * (5 + (i % 4) * 3),
    };
    out.push(sample(Sx, Sy, pose));
  }
  return out;
}
// App-like calibration: 9 straight dots + head-pose coverage at centre and 4 corners.
const calibration = () => [
  ...straightDots(),
  ...headMotionAt(0.5, 0.5), ...headMotionAt(0, 0), ...headMotionAt(1, 1),
  ...headMotionAt(1, 0), ...headMotionAt(0, 1),
];

test("fitCalibration: low residual", () => {
  const m = fitCalibration(calibration());
  assert.ok(m.residual < 2e-2, `residual=${m.residual}`);
});

test("HEAD-POSITION independence: on-screen from NEW head poses/distances → none (the bug)", () => {
  const m = fitCalibration(calibration());
  const poses = [
    { ex: 5, ey: 12, ez: 68, headYaw: 15, headPitch: 9 },    // shifted right, far, turned
    { ex: -6, ey: 8, ez: 52, headYaw: -14, headPitch: -8 },  // shifted left, near, turned
  ];
  for (const pose of poses) {
    for (const [nx, ny] of [[0.3, 0.3], [0.7, 0.7], [0.5, 0.5], [0.25, 0.8], [0.8, 0.25]]) {
      const r = decideBoundary(predictPoR(m, sample(...cm(nx, ny), pose)), DEFAULT_BOUNDARY, m.bounds);
      assert.equal(r.level, "none", `pose ${JSON.stringify(pose)} pt ${nx},${ny} → ${JSON.stringify(r)}`);
    }
  }
});

test("off-screen from a NEW head pose still flags", () => {
  const m = fitCalibration(calibration());
  const pose = { ex: 5, ey: 12, ez: 68, headYaw: 15, headPitch: 9 };
  const r = decideBoundary(predictPoR(m, sample(...cm(1.5, 0.5), pose)), DEFAULT_BOUNDARY, m.bounds);
  assert.ok(r.level !== "none" && r.direction === "right", JSON.stringify(r));
});

test("bounds padding widens the on-screen extent so corner overshoot stays none", () => {
  const padded = fitCalibration(calibration(), { boundsPad: 0.12 });
  const raw = fitCalibration(calibration(), { boundsPad: 0 });
  assert.ok(padded.bounds.xMin < raw.bounds.xMin && padded.bounds.xMax > raw.bounds.xMax, "x not padded");
  assert.ok(padded.bounds.yMin < raw.bounds.yMin && padded.bounds.yMax > raw.bounds.yMax, "y not padded");
  // A corner overshoot just past the raw edge stays within the padded bounds (not hard).
  const overshoot = { x: raw.bounds.xMin - 0.03, y: 0.5 };
  assert.notEqual(decideBoundary(overshoot, DEFAULT_BOUNDARY, padded.bounds).level, "hard");
  // A clearly off-screen point still flags with the correct direction.
  const off = { x: padded.bounds.xMin - 0.3, y: 0.5 };
  assert.equal(decideBoundary(off, DEFAULT_BOUNDARY, padded.bounds).direction, "left");
});


test("DISTANCE robustness: leaning far in/out (tz beyond calibration) does NOT blow up por", () => {
  const m = fitCalibration(calibration());
  // Calibration covered ~50–70cm; 30 and 95 are well outside → must NOT extrapolate-explode.
  for (const ez of [30, 95]) {
    const por = predictPoR(m, sample(...cm(0.5, 0.5), { ex: 0, ey: 10, ez, headYaw: 0, headPitch: 0 }));
    assert.ok(por.x > 0.2 && por.x < 0.8 && por.y > 0.2 && por.y < 0.8, `ez=${ez} por=${JSON.stringify(por)}`);
    const r = decideBoundary(por, DEFAULT_BOUNDARY, m.bounds);
    assert.equal(r.level, "none", `ez=${ez} → ${JSON.stringify(r)}`);
  }
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


// ---- head-deviation trigger (catches head-driven off-screen: up, and left/right via head turn) ----
test("head trigger: head-turn at centre (eyes compensate) flags via head, not PoR", () => {
  const m = fitCalibration(calibration());
  const s = sample(...cm(0.5, 0.5), { ex: 0, ey: 10, ez: 60, headYaw: 22, headPitch: 0 });
  assert.equal(decideBoundary(predictPoR(m, s)).level, "none"); // PoR alone: compensated → none
  const r = decidePoR(m, s);
  assert.equal(r.level, "hard");                                 // combined: head trigger fires
  assert.ok(r.direction === "left" || r.direction === "right", r.direction);
});

test("face tilted up but looking at centre → NOT flagged (no face-pitch trigger)", () => {
  const m = fitCalibration(calibration());
  const s = sample(...cm(0.5, 0.5), { ex: 0, ey: 10, ez: 60, headYaw: 0, headPitch: -24 });
  assert.equal(decidePoR(m, s).level, "none", JSON.stringify(decidePoR(m, s)));
});

test("iris above the screen → up via boundary-relative PoR", () => {
  const m = fitCalibration(calibration());
  const s = sample(...cm(0.5, -0.4), { ex: 0, ey: 10, ez: 60, headYaw: 0, headPitch: 0 });
  const r = decidePoR(m, s);
  assert.ok(r.level === "hard" && r.direction === "up", JSON.stringify(r));
});

test("head trigger: small on-screen head movement stays none", () => {
  const m = fitCalibration(calibration());
  const s = sample(...cm(0.5, 0.5), { ex: 0, ey: 10, ez: 60, headYaw: 8, headPitch: 6 });
  assert.equal(decidePoR(m, s).level, "none");
});

test("adaptive re-centering: slow drift while on-screen is absorbed (stays none)", () => {
  const st = createRecenterState();
  const bounds = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
  let flagged = false;
  // Raw PoR drifts 0.5→0.95 (x) and 0.5→0.10 (y) over 60 frames — like slowly leaning in.
  for (let i = 0; i <= 60; i++) {
    const raw = { x: 0.5 + 0.45 * (i / 60), y: 0.5 - 0.40 * (i / 60) };
    const { decision } = recenterApply(st, raw, bounds, DEFAULT_BOUNDARY);
    if (decision.level !== "none") flagged = true;
  }
  assert.equal(flagged, false, "slow lean drift should be absorbed → no flag");
});

test("adaptive re-centering: a FAST off-screen jump still flags", () => {
  const st = createRecenterState();
  const bounds = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
  for (let i = 0; i < 12; i++) recenterApply(st, { x: 0.5, y: 0.5 }, bounds, DEFAULT_BOUNDARY); // settle
  const { decision } = recenterApply(st, { x: 1.5, y: 0.5 }, bounds, DEFAULT_BOUNDARY);        // jump off right
  assert.ok(decision.level === "hard" && decision.direction === "right", JSON.stringify(decision));
});

test("adaptive re-centering: a GRADUAL look to the far right still flags (cap protects the margin)", () => {
  const st = createRecenterState();
  const bounds = { xMin: 0.2, xMax: 0.8, yMin: 0.2, yMax: 0.8 }; // compressed band
  let flagged = false, lastDir = null;
  for (let i = 0; i <= 50; i++) {
    const raw = { x: 0.5 + 0.6 * (i / 50), y: 0.5 }; // centre → 1.1, well past xMax
    const { decision } = recenterApply(st, raw, bounds, DEFAULT_BOUNDARY);
    if (decision.level === "hard") { flagged = true; lastDir = decision.direction; }
  }
  assert.ok(flagged && lastDir === "right", `flagged=${flagged} dir=${lastDir}`);
});


test("adaptive re-centering: disabled → behaves like a fixed boundary", () => {
  const st = createRecenterState();
  const bounds = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
  const cfg = Object.assign({}, DEFAULT_BOUNDARY, { RECENTER_ENABLED: false });
  const { decision } = recenterApply(st, { x: 1.3, y: 0.5 }, bounds, cfg);
  assert.ok(decision.level === "hard" && decision.direction === "right", JSON.stringify(decision));
});


test("decidePoR: eye-driven off-screen still flags via the PoR boundary", () => {
  const m = fitCalibration(calibration());
  const r = decidePoR(m, sample(...cm(1.5, 0.5), { ex: 0, ey: 10, ez: 60, headYaw: 0, headPitch: 0 }));
  assert.ok(r.level === "hard" && r.direction === "right", JSON.stringify(r));
});
