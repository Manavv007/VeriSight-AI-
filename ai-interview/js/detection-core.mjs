/* =================================================================
   detection-core.mjs — headless detection state machine (no DOM/MediaPipe)
   Single source of truth for "signal → incidents", reusing the SAME gaze-math
   decision functions the live engine uses (decideLookingAway + adaptBaseline),
   plus the episode/sustain logic that mirrors proctor.js. Drives both the
   replay harness (tools/) and unit tests, so the agent can iterate without a camera.
   ================================================================= */
import {
  decideLookingAway, adaptBaseline, deriveThresholdsFromCorners, DEFAULT_THRESHOLDS,
} from "./gaze-math.mjs";

// Timing + adaptation defaults (mirror gaze-engine.js / proctor.js CFG).
export const DEFAULT_TIMING = {
  AWAY_FLAG_MS: 1000,   // sustained "hard" before a looking-away incident is logged
  FACE_LOST_MS: 1500,   // sustained no-face before a face-lost incident is logged
  SOFT_MIN_MS: 350,     // soft cue debounce
  ADAPT_ALPHA: 0.04,    // head-pose drift adaptation rate
  ADAPT_ALPHA_GAZE: 0,  // eye baseline frozen (sustained gaze not absorbed)
};

/* createDetector({ thresholds, timing, baseline, adaptEnabled }) → detector
   detector.process(frame, nowMs) where frame = { yaw, pitch, gazeX, gazeY, facePresent }
   returns { level, direction, confidence, combinedYaw, combinedPitch, facePresent, softActive, incident }
   `incident` is non-null only on the frame an incident is first logged. */
export function createDetector(opts = {}) {
  const cfg = Object.assign({}, DEFAULT_THRESHOLDS, DEFAULT_TIMING, opts.thresholds, opts.timing);
  let baseline = opts.baseline ? { ...opts.baseline } : { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 };
  const adaptEnabled = opts.adaptEnabled !== false;

  const incidents = [];
  let idc = 0;
  const off = { active: false, start: 0, logged: false, dir: null };
  const faceLost = { active: false, start: 0, logged: false };
  let softStart = 0;
  let softActive = false;

  function log(type, atMs, direction, confidence, extra) {
    const inc = Object.assign(
      { id: ++idc, type, direction: direction || null, confidence: confidence == null ? null : confidence, atMs },
      extra || {}
    );
    incidents.push(inc);
    return inc;
  }
  const closeOff = () => { off.active = false; off.start = 0; off.logged = false; off.dir = null; };
  const closeFace = () => { faceLost.active = false; faceLost.start = 0; faceLost.logged = false; };

  function process(frame, nowMs) {
    const facePresent = frame.facePresent !== false;
    let incident = null;

    if (!facePresent) {
      softStart = 0; softActive = false; closeOff();
      if (!faceLost.active) { faceLost.active = true; faceLost.start = nowMs; }
      else if (nowMs - faceLost.start >= cfg.FACE_LOST_MS && !faceLost.logged) {
        faceLost.logged = true;
        incident = log("facelost", nowMs, null, null);
      }
      return { level: "none", direction: null, confidence: 0, combinedYaw: 0, combinedPitch: 0, facePresent: false, softActive: false, incident };
    }
    if (faceLost.active) closeFace();

    const d = decideLookingAway(frame, baseline, cfg);

    if (d.level === "none" && adaptEnabled) {
      baseline = adaptBaseline(baseline, frame, cfg.ADAPT_ALPHA, cfg.ADAPT_ALPHA_GAZE);
    }

    if (d.level === "hard") {
      softStart = 0; softActive = false;
      if (!off.active) { off.active = true; off.start = nowMs; off.dir = d.direction; }
      if (nowMs - off.start >= cfg.AWAY_FLAG_MS && !off.logged) {
        off.logged = true;
        incident = log("offscreen", nowMs, off.dir, d.confidence, {
          combinedYaw: round1(d.combinedYaw), combinedPitch: round1(d.combinedPitch),
        });
      }
    } else if (d.level === "soft") {
      closeOff();
      if (!softStart) softStart = nowMs;
      if (nowMs - softStart >= cfg.SOFT_MIN_MS) softActive = true;
    } else {
      softStart = 0; softActive = false; closeOff();
    }

    return {
      level: d.level, direction: d.direction, confidence: d.confidence,
      combinedYaw: d.combinedYaw, combinedPitch: d.combinedPitch,
      facePresent: true, softActive, incident,
    };
  }

  return {
    process,
    getIncidents: () => incidents.slice(),
    getBaseline: () => ({ ...baseline }),
    setBaseline: (b) => { baseline = { ...b }; },
    getThresholds: () => ({ ...cfg }),
    calibrateFromCorners: (corners, floors, softMargin, hardMargin) => {
      const r = deriveThresholdsFromCorners(corners, cfg, floors, softMargin, hardMargin);
      if (r) { baseline = { ...r.baseline }; Object.assign(cfg, r.thresholds); }
      return r;
    },
  };
}

const round1 = (v) => (typeof v === "number" ? Math.round(v * 10) / 10 : v);

/* replayTrace(trace, { thresholds, timing }) → result
   trace = { label, expect, meta:{ baseline, thresholds }, frames:[{tMs,yaw,pitch,gazeX,gazeY,facePresent}] }
   Uses CURRENT thresholds (so edits to gaze-math take effect) unless overridden,
   and the trace's recorded baseline (the calibration context). */
export function replayTrace(trace, overrides = {}) {
  const det = createDetector({
    thresholds: Object.assign({}, trace.meta && trace.meta.thresholds ? {} : {}, overrides.thresholds),
    timing: overrides.timing,
    baseline: trace.meta ? trace.meta.baseline : undefined,
  });
  const perFrame = [];
  for (const f of trace.frames) {
    const r = det.process(f, f.tMs);
    perFrame.push({ tMs: f.tMs, level: r.level, direction: r.direction });
  }
  return { incidents: det.getIncidents(), perFrame, finalBaseline: det.getBaseline() };
}
