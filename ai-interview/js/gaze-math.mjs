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
  a: { iris: 468, hOuter: 33, hInner: 133, vTop: 159, vBot: 145 },
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
  // --- Combined-gaze model -------------------------------------------------
  // Where you actually look on screen = head orientation + eye-in-head offset.
  // We fold the eye signal into the head angle so the decision is invariant to
  // how a movement splits between head and eyes (fixes head-tilt drift).
  EYE_GAIN_X: 50,     // degrees of effective gaze per unit of horizontal eye blendshape
  EYE_GAIN_Y: 30,     // degrees per unit of vertical eye blendshape
  // Head-gating: the eye signal is unreliable once the face is turned far, so we
  // down-weight it linearly to zero as |head deviation| approaches HEAD_GATE_DEG.
  HEAD_GATE_DEG: 25,
  EYE_MIN_WEIGHT: 0,
  // --- Soft / hard band on the combined gaze (degrees) ---------------------
  // soft  = "possibly looking away" (~50–80% confidence) → visual cue, NOT logged
  // hard  = soft + tolerance = "looking away" (>=80%)     → logged after sustain
  SOFT_YAW: 22, TOL_YAW: 14,            // hard yaw  = 36°
  SOFT_PITCH_DOWN: 12, TOL_PITCH_DOWN: 8,  // hard down = 20° (you tilt the head LESS looking
  // down than up, so down needs a lower bar)
  SOFT_PITCH_UP: 18, TOL_PITCH_UP: 12,     // hard up   = 30°
  INVERT_YAW: true,      // flip if "left"/"right" labels read backwards on your cam
  INVERT_GAZE_X: false,   // head-right and eye-right MUST agree in sign for the
  // combined model — flip one of these if they fight.
};

// Eye reliability weight: 1 when head is near neutral, ramps to EYE_MIN_WEIGHT
// once |head deviation| reaches HEAD_GATE_DEG.
function eyeWeight(headDevAbs, cfg) {
  const gate = cfg.HEAD_GATE_DEG != null ? cfg.HEAD_GATE_DEG : 25;
  const minW = cfg.EYE_MIN_WEIGHT != null ? cfg.EYE_MIN_WEIGHT : 0;
  if (gate <= 0) return 1;
  return Math.max(minW, Math.min(1, 1 - headDevAbs / gate));
}

/* computeCombined(signal, baseline, cfg) → { yaw, pitch, eyeWeightX, eyeWeightY }
   Combined gaze in degrees: head deviation + head-gated eye offset. Invariant to
   how a look splits between head and eyes, which removes head-tilt false positives. */
export function computeCombined(signal, baseline, cfg = DEFAULT_THRESHOLDS) {
  const b = baseline || { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 };
  let dHeadYaw = signal.yaw - b.yaw;
  let dHeadPitch = signal.pitch - b.pitch;
  let dGazeX = signal.gazeX - b.gazeX;
  const dGazeY = signal.gazeY - b.gazeY;
  if (cfg.INVERT_YAW) dHeadYaw = -dHeadYaw;
  if (cfg.INVERT_GAZE_X) dGazeX = -dGazeX;

  const wX = eyeWeight(Math.abs(dHeadYaw), cfg);
  const wY = eyeWeight(Math.abs(dHeadPitch), cfg);
  const gx = cfg.EYE_GAIN_X != null ? cfg.EYE_GAIN_X : 40;
  const gy = cfg.EYE_GAIN_Y != null ? cfg.EYE_GAIN_Y : 30;

  return {
    yaw: dHeadYaw + gx * wX * dGazeX,
    pitch: dHeadPitch + gy * wY * dGazeY,
    eyeWeightX: wX,
    eyeWeightY: wY,
  };
}

// Classify a magnitude against a soft/hard band → { level, confidence }.
function bandLevel(val, soft, hard) {
  if (val >= hard) {
    const extra = hard > 0 ? Math.min(1, (val - hard) / hard) : 0;
    return { level: "hard", confidence: Math.min(1, 0.8 + 0.2 * extra) };
  }
  if (val >= soft) {
    const t = hard > soft ? (val - soft) / (hard - soft) : 0;
    return { level: "soft", confidence: 0.5 + 0.3 * t };
  }
  return { level: "none", confidence: soft > 0 ? Math.min(0.49, (0.49 * val) / soft) : 0 };
}
const LEVEL_RANK = { none: 0, soft: 1, hard: 2 };

/* decideLookingAway(signal, baseline, cfg) →
     { level: "none"|"soft"|"hard", direction, confidence, combinedYaw, combinedPitch, reasons }
   Baseline-relative, combined head+eye, with a soft/hard confidence band. */
export function decideLookingAway(signal, baseline, cfg = DEFAULT_THRESHOLDS) {
  const out = { level: "none", direction: null, confidence: 0, combinedYaw: 0, combinedPitch: 0, reasons: [] };
  if (!signal || !signal.facePresent) return out;

  const c = computeCombined(signal, baseline, cfg);
  out.combinedYaw = c.yaw;
  out.combinedPitch = c.pitch;

  // Horizontal band.
  const h = bandLevel(Math.abs(c.yaw), cfg.SOFT_YAW, cfg.SOFT_YAW + cfg.TOL_YAW);
  const hDir = c.yaw >= 0 ? "right" : "left";

  // Vertical band (sign picks down vs up).
  const down = c.pitch >= 0;
  const vSoft = down ? cfg.SOFT_PITCH_DOWN : cfg.SOFT_PITCH_UP;
  const vTol = down ? cfg.TOL_PITCH_DOWN : cfg.TOL_PITCH_UP;
  const v = bandLevel(Math.abs(c.pitch), vSoft, vSoft + vTol);
  const vDir = down ? "down" : "up";

  // Pick the stronger axis (higher level; tie → higher confidence).
  let chosen, dir;
  if (LEVEL_RANK[h.level] > LEVEL_RANK[v.level]) { chosen = h; dir = hDir; }
  else if (LEVEL_RANK[v.level] > LEVEL_RANK[h.level]) { chosen = v; dir = vDir; }
  else { // equal rank
    if (h.confidence >= v.confidence) { chosen = h; dir = hDir; }
    else { chosen = v; dir = vDir; }
  }

  out.level = chosen.level;
  out.confidence = chosen.confidence;
  out.direction = out.level === "none" ? null : dir;
  if (out.level !== "none") out.reasons.push(`${dir}-${out.level}`);
  return out;
}

/* -----------------------------------------------------------------
   4-corner calibration → per-user thresholds
   ----------------------------------------------------------------- */
/* deriveThresholdsFromCorners(corners, cfg, floors, softMargin, hardMargin)
   `corners` : >=4 samples { yaw, pitch, gazeX, gazeY } captured while looking at
               each screen corner. Baseline = mean (screen center). We compute the
               COMBINED gaze (head + gated eye, using cfg's gains) at each corner;
               the max combined deviation is the on-screen edge. soft = edge ×
               softMargin, hard = edge × hardMargin (so tolerance = hard − soft),
               clamped to floors. Returns { baseline, thresholds } or null. */
export function deriveThresholdsFromCorners(corners, cfg = DEFAULT_THRESHOLDS, floors = {}, softMargin = 1.1, hardMargin = 1.4) {
  if (!corners || corners.length < 4) return null;
  const mean = (sel) => corners.reduce((s, c) => s + sel(c), 0) / corners.length;
  const baseline = {
    yaw: mean((c) => c.yaw),
    pitch: mean((c) => c.pitch),
    gazeX: mean((c) => c.gazeX),
    gazeY: mean((c) => c.gazeY),
  };

  // Combined-gaze extent across the corners (the on-screen boundary).
  let maxYaw = 0, maxPitch = 0;
  for (const c of corners) {
    const comb = computeCombined(c, baseline, cfg);
    maxYaw = Math.max(maxYaw, Math.abs(comb.yaw));
    maxPitch = Math.max(maxPitch, Math.abs(comb.pitch));
  }

  const floorSoftYaw = floors.SOFT_YAW != null ? floors.SOFT_YAW : 15;
  const floorTolYaw = floors.TOL_YAW != null ? floors.TOL_YAW : 8;
  const floorSoftPitch = floors.SOFT_PITCH != null ? floors.SOFT_PITCH : 14;
  const floorTolPitch = floors.TOL_PITCH != null ? floors.TOL_PITCH : 8;

  const softYaw = Math.max(maxYaw * softMargin, floorSoftYaw);
  const tolYaw = Math.max(maxYaw * (hardMargin - softMargin), floorTolYaw);
  const softPitch = Math.max(maxPitch * softMargin, floorSoftPitch);
  const tolPitch = Math.max(maxPitch * (hardMargin - softMargin), floorTolPitch);

  const thresholds = {
    SOFT_YAW: softYaw, TOL_YAW: tolYaw,
    SOFT_PITCH_DOWN: softPitch, TOL_PITCH_DOWN: tolPitch,
    SOFT_PITCH_UP: softPitch, TOL_PITCH_UP: tolPitch,
  };
  return { baseline, thresholds };
}

/* -----------------------------------------------------------------
   Adaptive baseline (slow drift compensation)
   ----------------------------------------------------------------- */
/* adaptBaseline(baseline, signal, alpha, gazeAlpha) → new baseline
   One EMA step nudging the baseline toward the current signal. `alpha` drives the
   HEAD axes (yaw/pitch); `gazeAlpha` (default = alpha) drives the EYE axes. The
   engine freezes the eye axes (gazeAlpha = 0) so a sustained eye glance is NOT
   absorbed — only head posture drift is tracked. Skipped during a hard look-away. */
export function adaptBaseline(baseline, signal, alpha, gazeAlpha) {
  const a = typeof alpha === "number" ? alpha : 0.04;
  const g = typeof gazeAlpha === "number" ? gazeAlpha : a;
  return {
    yaw: baseline.yaw + a * (signal.yaw - baseline.yaw),
    pitch: baseline.pitch + a * (signal.pitch - baseline.pitch),
    gazeX: baseline.gazeX + g * (signal.gazeX - baseline.gazeX),
    gazeY: baseline.gazeY + g * (signal.gazeY - baseline.gazeY),
  };
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
