/* =================================================================
   gaze-math.mjs — pure functions for head-pose + iris gaze proctoring
   No DOM / no MediaPipe dependency, so these are unit-testable in Node.
   Imported by gaze-engine.js (browser, ESM).
   ================================================================= */

const RAD2DEG = 180 / Math.PI;

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/* -----------------------------------------------------------------
   matrixToEuler(data16)
   Decompose a MediaPipe facial transformation matrix into Euler angles.
   `data16` is the 16-element COLUMN-MAJOR matrix from
   FaceLandmarkerResult.facialTransformationMatrixes[i].data.
   Returns { yaw, pitch, roll } in DEGREES:
     - pitch = rotation about X (head nods up/down)
     - yaw   = rotation about Y (head turns left/right)
     - roll  = rotation about Z (head tilts toward a shoulder)
   The rotation basis columns are length-normalized first so any uniform
   scale baked into the matrix does not distort the angles.
   ----------------------------------------------------------------- */
export function matrixToEuler(data16) {
  if (!data16 || data16.length < 16) return { yaw: 0, pitch: 0, roll: 0, valid: false };

  // Column-major → basis columns X, Y, Z (the rotation part).
  let xx = data16[0], xy = data16[1], xz = data16[2];
  let yx = data16[4], yy = data16[5], yz = data16[6];
  let zx = data16[8], zy = data16[9], zz = data16[10];

  const nx = Math.hypot(xx, xy, xz) || 1;
  const ny = Math.hypot(yx, yy, yz) || 1;
  const nz = Math.hypot(zx, zy, zz) || 1;
  xx /= nx; xy /= nx; xz /= nx;
  yx /= ny; yy /= ny; yz /= ny;
  zx /= nz; zy /= nz; zz /= nz;

  // Row-major rotation entries R[row][col].
  const R00 = xx, R10 = xy, R20 = xz;
  const R01 = yx, R11 = yy, R21 = yz;
  const R02 = zx, R12 = zy, R22 = zz;

  const sy = Math.hypot(R00, R10);
  const singular = sy < 1e-6;

  let px, yw, rl;
  if (!singular) {
    px = Math.atan2(R21, R22); // about X
    yw = Math.atan2(-R20, sy); // about Y
    rl = Math.atan2(R10, R00); // about Z
  } else {
    px = Math.atan2(-R12, R11);
    yw = Math.atan2(-R20, sy);
    rl = 0;
  }

  return {
    pitch: px * RAD2DEG,
    yaw: yw * RAD2DEG,
    roll: rl * RAD2DEG,
    valid: true,
  };
}

/* -----------------------------------------------------------------
   Iris gaze
   ----------------------------------------------------------------- */
// Landmark indices (MediaPipe FaceMesh w/ iris refinement, 478 points).
//   iris centers: 468 (one eye), 473 (other eye)
//   horizontal corners: outer / inner eye corners
//   vertical lids: upper / lower eyelid points
export const EYE = {
  a: { iris: 468, hOuter: 33,  hInner: 133, vTop: 159, vBot: 145 },
  b: { iris: 473, hOuter: 263, hInner: 362, vTop: 386, vBot: 374 },
};

// Position of `p` between `a` and `b` → 0 at a, 1 at b, 0.5 midway.
export function ratio(p, a, b) {
  const d = b - a;
  if (Math.abs(d) < 1e-9) return 0.5;
  return (p - a) / d;
}

/* computeIrisGaze(landmarks) → { gazeX, gazeY, valid }
   gazeX, gazeY are centered in [-1, 1] (0 = looking straight), averaged
   over both eyes. gazeX > 0 = iris toward the inner corner side;
   gazeY > 0 = iris toward the lower lid. (Absolute direction sign is
   confirmed against the live overlay; detection uses baseline-relative
   magnitude so sign only affects the text label.) */
export function computeIrisGaze(landmarks) {
  if (!landmarks || landmarks.length < 478) return { gazeX: 0, gazeY: 0, valid: false };

  let sx = 0, sy = 0, n = 0;
  for (const key of ["a", "b"]) {
    const c = EYE[key];
    const iris = landmarks[c.iris];
    const hO = landmarks[c.hOuter], hI = landmarks[c.hInner];
    const vT = landmarks[c.vTop], vB = landmarks[c.vBot];
    if (!iris || !hO || !hI || !vT || !vB) continue;
    const hx = ratio(iris.x, hO.x, hI.x); // 0..1, 0.5 = centered horizontally
    const vy = ratio(iris.y, vT.y, vB.y); // 0..1, 0.5 = centered vertically
    sx += (hx - 0.5) * 2;
    sy += (vy - 0.5) * 2;
    n += 1;
  }
  if (n === 0) return { gazeX: 0, gazeY: 0, valid: false };
  return { gazeX: sx / n, gazeY: sy / n, valid: true };
}

/* computeBlendshapeGaze(categories) → { gazeX, gazeY, valid }
   Eye gaze from MediaPipe's purpose-built eye-look blendshapes (scores 0..1).
   `categories` is FaceLandmarkerResult.faceBlendshapes[0].categories
   (an array of { categoryName, score }).

   The 8 eyeLook* scores are anatomically labeled, so both eyes REINFORCE each
   other (unlike raw iris-corner ratios, where the two eyes' inner/outer corners
   sit on opposite image sides and cancel out). Convention:
     gazeX > 0  → looking to the subject's right
     gazeY > 0  → looking down
   Values are roughly in [-1, 1]; a strong off-screen glance reaches ~0.5–0.9. */
export function computeBlendshapeGaze(categories) {
  if (!categories || !categories.length) return { gazeX: 0, gazeY: 0, valid: false };
  const m = Object.create(null);
  for (const c of categories) m[c.categoryName] = c.score;
  const g = (name) => (typeof m[name] === "number" ? m[name] : 0);

  const right = (g("eyeLookOutRight") + g("eyeLookInLeft")) / 2;
  const left = (g("eyeLookOutLeft") + g("eyeLookInRight")) / 2;
  const up = (g("eyeLookUpLeft") + g("eyeLookUpRight")) / 2;
  const down = (g("eyeLookDownLeft") + g("eyeLookDownRight")) / 2;

  const required = ["eyeLookOutRight", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookInRight"];
  const valid = required.every((name) => name in m);

  return { gazeX: right - left, gazeY: down - up, valid };
}

/* -----------------------------------------------------------------
   Fusion decision
   ----------------------------------------------------------------- */
export const DEFAULT_THRESHOLDS = {
  YAW_DEG: 18,        // head turned left/right beyond this (relative to baseline)
  PITCH_DOWN_DEG: 15, // head pitched down beyond this (looking at desk/phone)
  PITCH_UP_DEG: 15,   // head pitched up beyond this (looking above the screen)
  GAZE_X: 0.35,       // eyes darted horizontally beyond this — blendshape scale [-1,1]
  GAZE_Y_DOWN: 0.45,  // eyes dropped down beyond this. Kept conservative: the screen is
  GAZE_Y_UP: 0.45,    // short vertically and laptop screens sit below eye level, so normal
                      // on-screen reading already swings gazeY a lot. Head pitch is the
                      // reliable vertical signal in Quick mode; 4-corner mode calibrates
                      // the true top/bottom edges for precise eye-vertical detection.
  INVERT_YAW: false,  // flip if "left"/"right" labels read backwards on your cam
  INVERT_GAZE_X: false,
};

/* decideLookingAway(signal, baseline, cfg) → { lookingAway, direction, reasons }
   `signal`   : { facePresent, yaw, pitch, roll, gazeX, gazeY }
   `baseline` : { yaw, pitch, gazeX, gazeY } captured during neutral calibration
   Decision is baseline-relative; head pose OR eye gaze can trigger it. */
export function decideLookingAway(signal, baseline, cfg = DEFAULT_THRESHOLDS) {
  const out = { lookingAway: false, direction: null, reasons: [] };
  if (!signal || !signal.facePresent) return out;
  const b = baseline || { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 };

  let dyaw = signal.yaw - b.yaw;
  let dpitch = signal.pitch - b.pitch;
  let dgx = signal.gazeX - b.gazeX;
  const dgy = signal.gazeY - b.gazeY;
  if (cfg.INVERT_YAW) dyaw = -dyaw;
  if (cfg.INVERT_GAZE_X) dgx = -dgx;

  // Head pose (strongest signal).
  if (dyaw > cfg.YAW_DEG) { out.lookingAway = true; out.direction = "right"; out.reasons.push("head-yaw-right"); }
  else if (dyaw < -cfg.YAW_DEG) { out.lookingAway = true; out.direction = "left"; out.reasons.push("head-yaw-left"); }

  if (!out.lookingAway && dpitch > cfg.PITCH_DOWN_DEG) { out.lookingAway = true; out.direction = "down"; out.reasons.push("head-pitch-down"); }
  else if (!out.lookingAway && dpitch < -cfg.PITCH_UP_DEG) { out.lookingAway = true; out.direction = "up"; out.reasons.push("head-pitch-up"); }

  // Eye gaze (head may be still).
  if (!out.lookingAway && dgx > cfg.GAZE_X) { out.lookingAway = true; out.direction = "right"; out.reasons.push("eyes-right"); }
  else if (!out.lookingAway && dgx < -cfg.GAZE_X) { out.lookingAway = true; out.direction = "left"; out.reasons.push("eyes-left"); }
  const gazeYUp = cfg.GAZE_Y_UP != null ? cfg.GAZE_Y_UP : cfg.GAZE_Y_DOWN;
  if (!out.lookingAway && dgy > cfg.GAZE_Y_DOWN) { out.lookingAway = true; out.direction = "down"; out.reasons.push("eyes-down"); }
  else if (!out.lookingAway && dgy < -gazeYUp) { out.lookingAway = true; out.direction = "up"; out.reasons.push("eyes-up"); }

  return out;
}

/* -----------------------------------------------------------------
   4-corner calibration → per-user thresholds
   ----------------------------------------------------------------- */
/* deriveThresholdsFromCorners(corners, floors, margin)
   `corners` : array of >=4 samples { yaw, pitch, gazeX, gazeY } captured while
               the candidate looks at each screen corner.
   The screen center ≈ the mean of the corners; the maximum per-axis deviation
   from that center is the on-screen boundary. Thresholds = that boundary × a
   small margin, clamped to a sensible floor so a barely-moving candidate can't
   produce a hair-trigger. Returns { baseline, thresholds } or null. */
export function deriveThresholdsFromCorners(corners, floors = {}, margin = 1.15) {
  if (!corners || corners.length < 4) return null;
  const mean = (sel) => corners.reduce((s, c) => s + sel(c), 0) / corners.length;
  const baseline = {
    yaw: mean((c) => c.yaw),
    pitch: mean((c) => c.pitch),
    gazeX: mean((c) => c.gazeX),
    gazeY: mean((c) => c.gazeY),
  };
  const maxDev = (sel, center) => Math.max(...corners.map((c) => Math.abs(sel(c) - center)));
  const yawDev = maxDev((c) => c.yaw, baseline.yaw);
  const pitchDev = maxDev((c) => c.pitch, baseline.pitch);
  const gxDev = maxDev((c) => c.gazeX, baseline.gazeX);
  const gyDev = maxDev((c) => c.gazeY, baseline.gazeY);

  const floorYaw = floors.YAW_DEG != null ? floors.YAW_DEG : 10;
  const floorPitch = floors.PITCH != null ? floors.PITCH : 10;
  const floorGX = floors.GAZE_X != null ? floors.GAZE_X : 0.18;
  const floorGY = floors.GAZE_Y_DOWN != null ? floors.GAZE_Y_DOWN : 0.22;

  const thresholds = {
    YAW_DEG: Math.max(yawDev * margin, floorYaw),
    PITCH_DOWN_DEG: Math.max(pitchDev * margin, floorPitch),
    PITCH_UP_DEG: Math.max(pitchDev * margin, floorPitch),
    GAZE_X: Math.max(gxDev * margin, floorGX),
    GAZE_Y_DOWN: Math.max(gyDev * margin, floorGY),
    GAZE_Y_UP: Math.max(gyDev * margin, floorGY),
  };
  return { baseline, thresholds };
}

/* -----------------------------------------------------------------
   Moving-average smoother (suppresses per-frame jitter)
   ----------------------------------------------------------------- */
export function makeSmoother(windowSize = 5) {
  const buf = [];
  return {
    push(v) {
      if (typeof v !== "number" || Number.isNaN(v)) return this.value();
      buf.push(v);
      if (buf.length > windowSize) buf.shift();
      return this.value();
    },
    value() {
      if (buf.length === 0) return 0;
      let s = 0;
      for (const x of buf) s += x;
      return s / buf.length;
    },
    reset() { buf.length = 0; },
  };
}
