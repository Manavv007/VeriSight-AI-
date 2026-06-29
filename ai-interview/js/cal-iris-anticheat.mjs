/* =================================================================
   cal-iris-anticheat.mjs — iris-based calibration cheat detection
   Pure functions (no DOM). Uses landmark iris gaze, not blendshapes.
   ================================================================= */

export const IRIS_ANTICHEAT = {
  CENTER_MAX: 0.15,           // |irisGazeX| on center dot (absolute)
  WRONG_SIGN: 0.12,           // iris opposite to dot direction
  OVERSHOOT_CAP: 1.35,        // max |iris offset| vs dot eccentricity
  OVERSHOOT_SCALE: 0.55,      // typical iris units per full screen half-span
  DOT_HALF_SPAN: 0.46,        // max |target.x - 0.5| in 9-point grid
  PINNED_YAW_STDDEV_MAX: 4,   // degrees — head "still"
  PINNED_IRIS_STDDEV_MIN: 0.08,
  PINNED_WINDOW: 30,          // frames
  PINNED_SUSTAIN_MS: 800,
};

export function stdDev(arr) {
  if (!arr || arr.length < 2) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length);
}

export function isCenterTarget(target) {
  if (!target) return false;
  return Math.abs(target.x - 0.5) < 0.05 && Math.abs(target.y - 0.5) < 0.05;
}

function maxIrisOffsetForDot(dotDir, cfg) {
  return cfg.OVERSHOOT_CAP * cfg.OVERSHOOT_SCALE * Math.abs(dotDir) / cfg.DOT_HALF_SPAN;
}

/* Defense 5 — per-frame live check during captureWindow */
export function checkIrisLiveFrame(irisGazeX, target, centerIrisX, cfg = IRIS_ANTICHEAT) {
  if (irisGazeX == null || !target) return null;

  if (isCenterTarget(target)) {
    if (Math.abs(irisGazeX) > cfg.CENTER_MAX) return "iris_cheat";
    return null;
  }

  const dotDir = target.x - 0.5;
  if (Math.abs(dotDir) < 0.08) return null;

  const ref = centerIrisX != null ? centerIrisX : 0;
  const eyeOff = irisGazeX - ref;

  if (dotDir < 0 && eyeOff > cfg.WRONG_SIGN) return "iris_cheat";
  if (dotDir > 0 && eyeOff < -cfg.WRONG_SIGN) return "iris_cheat";

  const slack = 0.08;
  if (Math.abs(eyeOff) > maxIrisOffsetForDot(dotDir, cfg) + slack) return "iris_cheat";
  return null;
}

/* normalize samples for iris validation — prefer blendshape gazeX (negated), fall back to landmark iris */
export function eyeSamplesForValidation(samples) {
  if (!samples || !samples.length) return [];
  return samples
    .filter((s) => s.target && (s.blendGazeX != null || s.irisGazeX != null || s.gazeX != null))
    .map((s) => {
      if (s.blendGazeX != null) {
        return Object.assign({}, s, {
          irisGazeX: -s.blendGazeX, // invert blendshape sign to align positive-right
          _eyeSource: "blend",
        });
      }
      return Object.assign({}, s, {
        irisGazeX: s.irisGazeX != null ? s.irisGazeX : s.gazeX,
        _eyeSource: "iris",
      });
    });
}

/* Defenses 2 + 3 — post-capture validation on averaged samples */
export function validateIrisSamples(samples, cfg = IRIS_ANTICHEAT) {
  const withIris = eyeSamplesForValidation(samples);
  if (withIris.length < 2) return { ok: false, reason: "iris_cheat", detail: "no_eye_signal" };

  const center = withIris.find((s) => isCenterTarget(s.target));
  const centerIrisX = center ? center.irisGazeX : 0;

  for (const s of withIris) {
    if (isCenterTarget(s.target) && Math.abs(s.irisGazeX) > cfg.CENTER_MAX) {
      return { ok: false, reason: "iris_cheat", detail: "center_not_neutral" };
    }
  }

  for (const s of withIris) {
    if (isCenterTarget(s.target)) continue;
    const dotDir = s.target.x - 0.5;
    if (Math.abs(dotDir) < 0.08) continue;

    const eyeOff = s.irisGazeX - centerIrisX;
    if (dotDir < 0 && eyeOff > 0.05) {
      return { ok: false, reason: "iris_cheat", detail: "wrong_sign" };
    }
    if (dotDir > 0 && eyeOff < -0.05) {
      return { ok: false, reason: "iris_cheat", detail: "wrong_sign" };
    }
    if (Math.abs(eyeOff) > maxIrisOffsetForDot(dotDir, cfg) + 0.1) {
      return { ok: false, reason: "iris_cheat", detail: "overshoot" };
    }
  }

  return { ok: true };
}

/* Map 4-corner name → normalized target for iris validation */
export function cornerToTarget(corner) {
  if (!corner) return null;
  const right = corner.includes("right");
  const bottom = corner.includes("bottom");
  return { x: right ? 0.92 : 0.08, y: bottom ? 0.88 : 0.12 };
}

export function validateIrisCorners(corners, cfg = IRIS_ANTICHEAT) {
  const samples = corners.map((c) => ({
    irisGazeX: c.irisGazeX,
    irisGazeY: c.irisGazeY,
    blendGazeX: c.blendGazeX,
    blendGazeY: c.blendGazeY,
    gazeX: c.gazeX,
    target: cornerToTarget(c.corner),
  }));
  return validateIrisSamples(samples, cfg);
}

/* Defense 6 — head stable but iris sweeping during one dot hold */
export function createIrisPinnedTracker(cfg = IRIS_ANTICHEAT) {
  const st = { yawBuf: [], irisBuf: [], pinnedStart: 0 };
  return {
    reset() {
      st.yawBuf = [];
      st.irisBuf = [];
      st.pinnedStart = 0;
    },
    tick(yaw, irisGazeX, tMs) {
      if (irisGazeX == null) return null;
      st.yawBuf.push(yaw);
      st.irisBuf.push(irisGazeX);
      if (st.yawBuf.length > cfg.PINNED_WINDOW) st.yawBuf.shift();
      if (st.irisBuf.length > cfg.PINNED_WINDOW) st.irisBuf.shift();
      if (st.yawBuf.length < cfg.PINNED_WINDOW) return null;

      const yawSd = stdDev(st.yawBuf);
      const irisSd = stdDev(st.irisBuf);
      if (yawSd < cfg.PINNED_YAW_STDDEV_MAX && irisSd > cfg.PINNED_IRIS_STDDEV_MIN) {
        if (!st.pinnedStart) st.pinnedStart = tMs;
        else if (tMs - st.pinnedStart >= cfg.PINNED_SUSTAIN_MS) return "iris_pinned";
      } else {
        st.pinnedStart = 0;
      }
      return null;
    },
  };
}
