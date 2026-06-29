/* =================================================================
   gaze-engine.js  (ES module)
   MediaPipe Face Landmarker driver for browser-only gaze/head proctoring.
   - Owns the webcam stream and the per-frame detection loop
   - Emits a normalized signal: { facePresent, yaw, pitch, roll,
     gazeX, gazeY, lookingAway, direction, calibrating, ts }
   - Captures a neutral baseline (3 s) so detection is baseline-relative
   - Optionally draws the face mesh + iris overlay onto a canvas
   Exposes window.GazeEngine for the classic-script proctor.js to consume.
   ================================================================= */
import { FaceLandmarker, FilesetResolver, DrawingUtils } from "../vendor/tasks-vision/vision_bundle.mjs";
import {
  matrixToEuler, computeIrisGaze, computeBlendshapeGaze, decideLookingAway,
  deriveThresholdsFromCorners, adaptBaseline, makeSmoother, DEFAULT_THRESHOLDS,
} from "./gaze-math.mjs";
import { fitCalibration, predictPoR, decidePoR, createPoRTracker, decideBoundary, DEFAULT_BOUNDARY } from "./gaze-mapping.mjs";
import {
  checkIrisLiveFrame, validateIrisSamples, validateIrisCorners,
  createIrisPinnedTracker, IRIS_ANTICHEAT,
} from "./cal-iris-anticheat.mjs";

const WASM_PATH = "./vendor/tasks-vision/wasm";
const MODEL_PATH = "./models/face_landmarker.task";

const CFG = Object.assign({ SMOOTH_WINDOW: 5, ADAPT_ALPHA: 0.04, ADAPT_ALPHA_GAZE: 0 }, DEFAULT_THRESHOLDS, DEFAULT_BOUNDARY);

const state = {
  landmarker: null,
  video: null,
  overlay: null,
  octx: null,
  drawer: null,
  stream: null,
  running: false,
  lastVideoTime: -1,
  onSignal: null,
  baseline: { yaw: 0, pitch: 0, gazeX: 0, gazeY: 0 },
  calibrating: false,
  adaptEnabled: false,
  poRModel: null,
  poRTracker: null,
  calBuf: null,
  calTarget: null,       // { x, y } during dot capture — for live iris anti-cheat
  centerIrisX: null,
  collectingFrames: false,
  frameBuf: null,
  latest: null,
  smoothers: null,
};

function freshSmoothers() {
  return {
    yaw: makeSmoother(CFG.SMOOTH_WINDOW),
    pitch: makeSmoother(CFG.SMOOTH_WINDOW),
    roll: makeSmoother(CFG.SMOOTH_WINDOW),
    gazeX: makeSmoother(CFG.SMOOTH_WINDOW),
    gazeY: makeSmoother(CFG.SMOOTH_WINDOW),
    tx: makeSmoother(CFG.SMOOTH_WINDOW),
    ty: makeSmoother(CFG.SMOOTH_WINDOW),
    tz: makeSmoother(CFG.SMOOTH_WINDOW),
  };
}

// ------------------------------------------------------------------
// Initialization
// ------------------------------------------------------------------
async function init() {
  if (state.landmarker) return state.landmarker;
  const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_PATH, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    state.landmarker = await FaceLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch (e) {
    // Fall back to CPU on machines without a usable WebGL2/GPU delegate.
    state.landmarker = await FaceLandmarker.createFromOptions(fileset, opts("CPU"));
  }
  state.smoothers = freshSmoothers();
  return state.landmarker;
}

async function startCamera(videoEl) {
  state.video = videoEl;
  state.stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  videoEl.srcObject = state.stream;
  await videoEl.play();
  await new Promise((res) => {
    if (videoEl.readyState >= 2) return res();
    videoEl.addEventListener("loadeddata", () => res(), { once: true });
  });
}

function attachOverlay(canvasEl) {
  state.overlay = canvasEl;
  if (canvasEl) {
    state.octx = canvasEl.getContext("2d");
    state.drawer = new DrawingUtils(state.octx);
  }
}

// ------------------------------------------------------------------
// Detection loop
// ------------------------------------------------------------------
function start(onSignal) {
  state.onSignal = onSignal || null;
  if (state.running) return;
  state.running = true;
  state.lastVideoTime = -1;
  requestAnimationFrame(loop);
}

function stop() {
  state.running = false;
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
}

function pause() { state.running = false; }
function resume() {
  if (!state.running) {
    state.running = true;
    state.lastVideoTime = -1;
    requestAnimationFrame(loop);
  }
}

function loop() {
  if (!state.running) return;
  const video = state.video;
  if (video && state.landmarker && video.currentTime !== state.lastVideoTime && video.videoWidth > 0) {
    state.lastVideoTime = video.currentTime;
    let result = null;
    try {
      result = state.landmarker.detectForVideo(video, performance.now());
    } catch (_) { /* transient frame error */ }
    if (result) processResult(result);
  }
  requestAnimationFrame(loop);
}

function processResult(result) {
  const hasFace = result.faceLandmarks && result.faceLandmarks.length > 0;
  let signal;

  if (!hasFace) {
    if (state.smoothers) Object.values(state.smoothers).forEach((s) => s.reset());
    signal = {
      facePresent: false, yaw: 0, pitch: 0, roll: 0, gazeX: 0, gazeY: 0, tx: 0, ty: 0, tz: 0,
      lookingAway: false, direction: null, level: "none", confidence: 0, por: null,
      calibrating: state.calibrating, ts: Date.now(),
    };
    drawOverlay(null, "none");
  } else {
    const landmarks = result.faceLandmarks[0];
    const mtx = result.facialTransformationMatrixes && result.facialTransformationMatrixes[0];
    const euler = mtx ? matrixToEuler(mtx.data) : { yaw: 0, pitch: 0, roll: 0 };
    // Head 3D position (camera space) from the matrix translation column (cm-ish).
    const t = mtx ? mtx.data : null;

    const blend = result.faceBlendshapes && result.faceBlendshapes[0];
    const bGaze = blend ? computeBlendshapeGaze(blend.categories) : { valid: false };
    const gaze = bGaze.valid ? bGaze : computeIrisGaze(landmarks);
    const irisGaze = computeIrisGaze(landmarks);
    const blendGazeX = bGaze.valid ? bGaze.gazeX : null;
    const blendGazeY = bGaze.valid ? bGaze.gazeY : null;

    const sm = state.smoothers;
    const yaw = sm.yaw.push(euler.yaw);
    const pitch = sm.pitch.push(euler.pitch);
    const roll = sm.roll.push(euler.roll);
    const gazeX = sm.gazeX.push(gaze.gazeX);
    const gazeY = sm.gazeY.push(gaze.gazeY);
    const tx = sm.tx.push(t ? t[12] : 0);
    const ty = sm.ty.push(t ? t[13] : 0);
    const tz = sm.tz.push(t ? t[14] : 0);
    const irisGazeX = irisGaze.valid ? irisGaze.gazeX : null;
    const irisGazeY = irisGaze.valid ? irisGaze.gazeY : null;
    // Anti-cheat prefers blendshape gaze (with inverted sign to match target direction)
    // because raw iris-corner landmarks cancel out due to bilateral symmetry.
    const antiCheatEyeX = blendGazeX != null ? -blendGazeX : irisGazeX;
    const antiCheatEyeY = blendGazeY != null ? blendGazeY : irisGazeY;
    const antiCheatEyeSource = blendGazeX != null ? "blend" : (irisGazeX != null ? "iris" : null);

    const raw = {
      facePresent: true, yaw, pitch, roll, gazeX, gazeY, tx, ty, tz,
      irisGazeX, irisGazeY, blendGazeX, blendGazeY,
      antiCheatEyeX, antiCheatEyeY, antiCheatEyeSource,
      landmarkCount: landmarks.length,
    };

    if (state.calibrating && state.calBuf) {
      state.calBuf.yaw.push(yaw); state.calBuf.pitch.push(pitch);
      state.calBuf.gazeX.push(gazeX); state.calBuf.gazeY.push(gazeY);
      state.calBuf.tx.push(tx); state.calBuf.ty.push(ty); state.calBuf.tz.push(tz);
      if (irisGazeX != null) state.calBuf.irisGazeX.push(irisGazeX);
      if (irisGazeY != null) state.calBuf.irisGazeY.push(irisGazeY);
      if (blendGazeX != null) state.calBuf.blendGazeX.push(blendGazeX);
      if (blendGazeY != null) state.calBuf.blendGazeY.push(blendGazeY);
    }
    if (state.collectingFrames && state.frameBuf) {
      state.frameBuf.push({ headYaw: yaw, headPitch: pitch, gazeX, gazeY, tx, ty, tz });
    }

    // PRIMARY: point-of-regard → screen boundary (head/distance independent).
    // FALLBACK: angular head+eye model (used until a PoR model is calibrated).
    let por = null, decision;
    if (state.calibrating) {
      decision = { level: "none", direction: null, confidence: 0 };
    } else if (state.poRModel) {
      const r = (state.poRTracker || (state.poRTracker = createPoRTracker(state.poRModel, CFG)))
        .step({ headYaw: yaw, headPitch: pitch, gazeX, gazeY, tx, ty, tz });
      por = r.por;
      decision = r;
    } else {
      decision = decideLookingAway(raw, state.baseline, CFG);
    }

    signal = Object.assign(raw, {
      level: decision.level,
      lookingAway: decision.level === "hard",
      direction: decision.direction,
      confidence: decision.confidence,
      por,
      combinedYaw: decision.combinedYaw != null ? decision.combinedYaw : 0,
      combinedPitch: decision.combinedPitch != null ? decision.combinedPitch : 0,
      calibrating: state.calibrating,
      ts: Date.now(),
    });
    drawOverlay(landmarks, decision.level);

    // Drift compensation only applies to the angular FALLBACK (PoR is absolute).
    if (!state.poRModel && state.adaptEnabled && !state.calibrating && decision.level === "none") {
      state.baseline = adaptBaseline(state.baseline, raw, CFG.ADAPT_ALPHA, CFG.ADAPT_ALPHA_GAZE);
    }
  }

  state.latest = signal;
  if (state.onSignal) state.onSignal(signal);
}

// ------------------------------------------------------------------
// Overlay
// ------------------------------------------------------------------
function drawOverlay(landmarks, level) {
  const cv = state.overlay, ctx = state.octx, d = state.drawer;
  if (!cv || !ctx) return;
  if (state.video && (cv.width !== state.video.videoWidth || cv.height !== state.video.videoHeight)) {
    cv.width = state.video.videoWidth;
    cv.height = state.video.videoHeight;
  }
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!landmarks || !d) return;

  const hard = level === "hard", soft = level === "soft";
  const meshColor = hard ? "rgba(255,92,138,0.5)" : soft ? "rgba(255,180,84,0.45)" : "rgba(45,226,198,0.35)";
  const irisColor = hard ? "#ff5c8a" : soft ? "#ffb454" : "#2de2c6";
  d.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: meshColor, lineWidth: 0.5 });
  d.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: irisColor, lineWidth: 2 });
  d.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: irisColor, lineWidth: 2 });
  d.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: meshColor, lineWidth: 1 });
}

// ------------------------------------------------------------------
// Calibration capture
// ------------------------------------------------------------------
// Average the live (smoothed) signal over a window. Used both for the quick
// neutral baseline and for each target of the 4-corner calibration.
function captureWindow(durationMs = 3000) {
  return new Promise((resolve, reject) => {
    state.calBuf = {
      yaw: [], pitch: [], gazeX: [], gazeY: [], tx: [], ty: [], tz: [],
      irisGazeX: [], irisGazeY: [], blendGazeX: [], blendGazeY: [],
    };
    state.calibrating = true;
    setTimeout(() => {
      state.calibrating = false;
      const buf = state.calBuf;
      state.calBuf = null;
      if (!buf || buf.yaw.length < 3) {
        reject(new Error("No stable face detected — please face the camera and try again."));
        return;
      }
      const avg = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : null);
      const irisX = avg(buf.irisGazeX);
      const blendX = avg(buf.blendGazeX);
      resolve({
        yaw: avg(buf.yaw), pitch: avg(buf.pitch),
        gazeX: avg(buf.gazeX), gazeY: avg(buf.gazeY),
        tx: avg(buf.tx), ty: avg(buf.ty), tz: avg(buf.tz),
        irisGazeX: irisX, irisGazeY: avg(buf.irisGazeY),
        blendGazeX: blendX, blendGazeY: avg(buf.blendGazeY),
        antiCheatEyeX: irisX != null ? irisX : blendX,
        samples: buf.yaw.length,
      });
    }, durationMs);
  });
}

// Capture one calibration target: average features while the candidate looks at a
// known screen point. Returns a PoR calibration sample (raw head + eye + 3D pos).
function captureSample(durationMs, target) {
  return captureWindow(durationMs).then((w) => ({
    headYaw: w.yaw, headPitch: w.pitch, gazeX: w.gazeX, gazeY: w.gazeY,
    irisGazeX: w.irisGazeX, irisGazeY: w.irisGazeY,
    blendGazeX: w.blendGazeX, blendGazeY: w.blendGazeY,
    antiCheatEyeX: w.antiCheatEyeX,
    tx: w.tx, ty: w.ty, tz: w.tz, target, samples: w.samples,
  }));
}

function setCalTarget(target) {
  state.calTarget = target ? { x: target.x, y: target.y } : null;
}

function setCenterIrisX(v) {
  state.centerIrisX = typeof v === "number" ? v : null;
}

// Collect per-frame samples over a window (NOT averaged) — used for the head-movement
// calibration pass that identifies how head pose maps to the gaze point.
function captureFrames(durationMs) {
  return new Promise((resolve) => {
    state.frameBuf = [];
    state.collectingFrames = true;
    state.calibrating = true;
    setTimeout(() => {
      state.collectingFrames = false;
      state.calibrating = false;
      const f = state.frameBuf || [];
      state.frameBuf = null;
      resolve(f);
    }, durationMs);
  });
}

// Fit the point-of-regard model from 9 (or N) captured calibration samples.
function fitGaze(samples) {
  const model = fitCalibration(samples);
  if (model) { state.poRModel = model; state.poRTracker = createPoRTracker(model, CFG); state.adaptEnabled = false; }
  return model;
}

// Quick neutral baseline: capture "looking straight at the screen" and store it.
function captureBaseline(durationMs = 3000) {
  return captureWindow(durationMs).then((avg) => {
    state.baseline = { yaw: avg.yaw, pitch: avg.pitch, gazeX: avg.gazeX, gazeY: avg.gazeY };
    state.adaptEnabled = true;
    return avg;
  });
}

// 4-corner calibration: derive per-user baseline + soft/hard thresholds.
function calibrateFromCorners(corners, opts = {}) {
  const r = deriveThresholdsFromCorners(corners, CFG, opts.floors, opts.softMargin, opts.hardMargin);
  if (!r) return null;
  state.baseline = r.baseline;
  Object.assign(CFG, r.thresholds);
  state.adaptEnabled = true;
  return r;
}

// ------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------
window.GazeEngine = {
  async setup(videoEl, overlayEl) {
    await init();
    attachOverlay(overlayEl);
    await startCamera(videoEl);
    return true;
  },
  start,
  stop,
  pause,
  resume,
  captureBaseline,
  captureWindow,
  captureSample,
  captureFrames,
  setCalTarget,
  setCenterIrisX,
  fitGaze,
  calibrateFromCorners,
  checkIrisLiveFrame,
  validateIrisSamples,
  validateIrisCorners,
  createIrisPinnedTracker,
  IRIS_ANTICHEAT,
  hasGazeModel: () => !!state.poRModel,
  getPoRModel: () => state.poRModel,
  clearGazeModel: () => { state.poRModel = null; state.poRTracker = null; },
  getLatest: () => state.latest,
  getBaseline: () => Object.assign({}, state.baseline),
  setBaseline: (b) => { state.baseline = Object.assign({}, b); },
  getThresholds: () => Object.assign({}, CFG),
  setThresholds: (patch) => { Object.assign(CFG, patch || {}); },
  isReady: () => !!state.landmarker,
};

document.dispatchEvent(new CustomEvent("gazeengine:loaded"));
