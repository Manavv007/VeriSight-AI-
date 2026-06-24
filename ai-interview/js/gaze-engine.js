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
  deriveThresholdsFromCorners, makeSmoother, DEFAULT_THRESHOLDS,
} from "./gaze-math.mjs";

const WASM_PATH = "./vendor/tasks-vision/wasm";
const MODEL_PATH = "./models/face_landmarker.task";

const CFG = Object.assign({ SMOOTH_WINDOW: 5 }, DEFAULT_THRESHOLDS);

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
  calBuf: null,
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
      facePresent: false, yaw: 0, pitch: 0, roll: 0, gazeX: 0, gazeY: 0,
      lookingAway: false, direction: null, calibrating: state.calibrating, ts: Date.now(),
    };
    drawOverlay(null, false);
  } else {
    const landmarks = result.faceLandmarks[0];
    const mtx = result.facialTransformationMatrixes && result.facialTransformationMatrixes[0];
    const euler = mtx ? matrixToEuler(mtx.data) : { yaw: 0, pitch: 0, roll: 0 };

    // Eye gaze: prefer MediaPipe's calibrated eye-look blendshapes (both eyes
    // reinforce instead of cancelling); fall back to iris geometry if absent.
    const blend = result.faceBlendshapes && result.faceBlendshapes[0];
    const bGaze = blend ? computeBlendshapeGaze(blend.categories) : { valid: false };
    const gaze = bGaze.valid ? bGaze : computeIrisGaze(landmarks);

    const sm = state.smoothers;
    const yaw = sm.yaw.push(euler.yaw);
    const pitch = sm.pitch.push(euler.pitch);
    const roll = sm.roll.push(euler.roll);
    const gazeX = sm.gazeX.push(gaze.gazeX);
    const gazeY = sm.gazeY.push(gaze.gazeY);

    const raw = { facePresent: true, yaw, pitch, roll, gazeX, gazeY };

    if (state.calibrating && state.calBuf) {
      state.calBuf.yaw.push(yaw); state.calBuf.pitch.push(pitch);
      state.calBuf.gazeX.push(gazeX); state.calBuf.gazeY.push(gazeY);
    }

    const decision = state.calibrating
      ? { lookingAway: false, direction: null }
      : decideLookingAway(raw, state.baseline, CFG);

    signal = Object.assign(raw, {
      lookingAway: decision.lookingAway,
      direction: decision.direction,
      calibrating: state.calibrating,
      ts: Date.now(),
    });
    drawOverlay(landmarks, decision.lookingAway);
  }

  state.latest = signal;
  if (state.onSignal) state.onSignal(signal);
}

// ------------------------------------------------------------------
// Overlay
// ------------------------------------------------------------------
function drawOverlay(landmarks, away) {
  const cv = state.overlay, ctx = state.octx, d = state.drawer;
  if (!cv || !ctx) return;
  if (state.video && (cv.width !== state.video.videoWidth || cv.height !== state.video.videoHeight)) {
    cv.width = state.video.videoWidth;
    cv.height = state.video.videoHeight;
  }
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!landmarks || !d) return;

  const meshColor = away ? "rgba(255,92,138,0.5)" : "rgba(45,226,198,0.35)";
  const irisColor = away ? "#ff5c8a" : "#2de2c6";
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
    state.calBuf = { yaw: [], pitch: [], gazeX: [], gazeY: [] };
    state.calibrating = true;
    setTimeout(() => {
      state.calibrating = false;
      const buf = state.calBuf;
      state.calBuf = null;
      if (!buf || buf.yaw.length < 3) {
        reject(new Error("No stable face detected — please face the camera and try again."));
        return;
      }
      const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;
      resolve({
        yaw: avg(buf.yaw), pitch: avg(buf.pitch),
        gazeX: avg(buf.gazeX), gazeY: avg(buf.gazeY),
        samples: buf.yaw.length,
      });
    }, durationMs);
  });
}

// Quick neutral baseline: capture "looking straight at the screen" and store it.
function captureBaseline(durationMs = 3000) {
  return captureWindow(durationMs).then((avg) => {
    state.baseline = { yaw: avg.yaw, pitch: avg.pitch, gazeX: avg.gazeX, gazeY: avg.gazeY };
    return avg;
  });
}

// 4-corner calibration: derive per-user baseline + thresholds from corner samples.
function calibrateFromCorners(corners, floors, margin) {
  const r = deriveThresholdsFromCorners(corners, floors, margin);
  if (!r) return null;
  state.baseline = r.baseline;
  Object.assign(CFG, r.thresholds);
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
  calibrateFromCorners,
  getLatest: () => state.latest,
  getBaseline: () => Object.assign({}, state.baseline),
  setBaseline: (b) => { state.baseline = Object.assign({}, b); },
  getThresholds: () => Object.assign({}, CFG),
  setThresholds: (patch) => { Object.assign(CFG, patch || {}); },
  isReady: () => !!state.landmarker,
};

document.dispatchEvent(new CustomEvent("gazeengine:loaded"));
