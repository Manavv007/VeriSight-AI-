# GazeProctor — MediaPipe Gaze/Head Proctoring

Browser-only AI mock-interview + proctoring app. Detects "looking away" from
**head pose** (MediaPipe FaceLandmarker transformation matrix → yaw/pitch/roll)
**OR** **eye/iris gaze**, relative to a 3-second neutral baseline, sustained ≥1 s.
All video is processed locally in the browser; nothing is uploaded.

## Implementation status

- [x] Vendor `@mediapipe/tasks-vision` (`vision_bundle.mjs` + `wasm/`) and the
      `face_landmarker.task` model locally (self-contained, no CDN).
- [x] `js/gaze-engine.js` (ES module) — owns the webcam, runs `detectForVideo`,
      emits a normalized signal, draws the mesh/iris overlay, captures the baseline.
- [x] `js/gaze-math.mjs` — pure functions: `matrixToEuler`, `computeIrisGaze`,
      `decideLookingAway`, `makeSmoother` (unit-tested in Node).
- [x] `js/proctor.js` — neutral-baseline calibration, looking-away / face-lost /
      tab-switch episodes (0.5 s warn, 1.0 s flag, 1.5 s face-lost), audit log,
      integrity score, JSON/CSV export with head/gaze fields, summary, fullscreen.
- [x] `index.html` / `css/style.css` — mirrored video + landmark overlay in the
      camera card, live telemetry HUD, calibration overlay repurposed to baseline.
- [x] WebGazer removed (`js/webgazer.js` and old `mediapipe/face_mesh` deleted).

## Architecture

```
getUserMedia → FaceLandmarker.detectForVideo
   ├─ facialTransformationMatrixes → matrixToEuler → yaw/pitch/roll
   └─ 478 landmarks (iris) + blendshapes → computeIrisGaze → gazeX/gazeY
        → decideLookingAway(vs baseline) → { facePresent, lookingAway, direction }
        → proctor.js timers + audit + UI + JSON/CSV export
```

## How to run

```bash
cd ai-interview
python -m http.server 8000
# open http://localhost:8000 in Chrome or Edge (STT also needs Chromium/Safari)
```
Must be served over HTTP (ES modules + WASM do not load from file://).

## Detection model (combined gaze + soft/hard band)

"Looking away" is judged on a **combined gaze** angle = head deviation + head-gated eye
offset (`combinedYaw`, `combinedPitch`, degrees), all relative to the calibrated
baseline. This is invariant to how a look splits between head and eyes, which removes
head-tilt drift false positives. The eye term is **down-weighted as the head turns**
(unreliable at large angles). Each axis uses a two-tier band:
`soft` = possibly looking away (~50–80% confidence, visual cue only, **not logged**),
`hard` = soft + tolerance = looking away (>=80%, **logged** after the sustain timer).

## Tuning (gaze-math.mjs → DEFAULT_THRESHOLDS; corner mode overrides them)

- `EYE_GAIN_X` (40) / `EYE_GAIN_Y` (30): degrees of gaze per unit of eye blendshape.
- `HEAD_GATE_DEG` (25): head deviation at which the eye term is fully down-weighted.
- `SOFT_YAW` (22) / `TOL_YAW` (14): horizontal soft / (hard = soft+tol = 36°).
- `SOFT_PITCH_DOWN` (18) / `TOL_PITCH_DOWN` (12); `SOFT_PITCH_UP` (18) / `TOL_PITCH_UP` (12).
- `SMOOTH_WINDOW` (5, in gaze-engine.js): frames averaged to suppress jitter.
- `INVERT_YAW` / `INVERT_GAZE_X`: head-right and eye-right MUST agree in sign for the
  combined model — flip one if left/right reads backwards or detection feels weak.
- 4-corner mode derives `SOFT_*`/`TOL_*` from your real screen (`CORNER_*` in proctor.js).
Detection is baseline-relative, so sitting off-axis does not cause false flags.

## Automated verification performed

- `node --test` → 36/36 unit tests pass (Euler decomposition incl. scale, iris ratio,
  blendshape eye gaze, combined head+eye gaze, head-gated eye weighting, soft/hard band
  + confidence, baseline-relative, 4-corner threshold derivation, smoother).
- `node --check` passes for proctor.js, interview.js, gaze-math.mjs, and gaze-engine.js (ESM).
- All DOM ids referenced by JS exist in index.html; all runtime assets present;
  no stale `webgazer` references.
- Local server serves every asset 200 with correct MIME (`.mjs` → text/javascript,
  `.wasm` → application/wasm, model → application/octet-stream).

## Calibration modes (toggle on the welcome card before Start)

- **Quick** (optional): 3-second neutral baseline — look straight, hold still. Detection
  uses the default thresholds, baseline-relative.
- **4-corner (default, precise)**: look at each screen corner (~2 s each). The app measures the
  head/eye span across your real screen and seating, then derives per-user thresholds
  (`deriveThresholdsFromCorners`, clamped to safe floors × a 1.15 margin). Use this when
  you want "off screen" to match your exact monitor size rather than a fixed angle.

## Manual verification checklist (needs a webcam — run in browser)

1. Start → grant camera → mirrored video with face mesh + iris overlay appears.
2. Baseline: look straight, hold still 3 s → app reveals, top bar shows "Live ✓".
3. Turn head left/right or look down >1 s → "Looking Away (left/right/down)" incident
   logged with yaw/pitch/gaze in the timeline; camera card flashes red after ~0.5 s.
4. Cover the camera >1.5 s → "Face Lost" incident.
5. Switch tab / click another window → "Focus Lost / Tab Switch" incident.
6. Finish → summary modal; download JSON & CSV (now include head/gaze fields).

If left/right labels look reversed, set `INVERT_YAW: true` (and/or `INVERT_GAZE_X`).
