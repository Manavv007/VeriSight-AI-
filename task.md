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

## Detection model (point-of-regard → screen boundary; PRIMARY)

Detection asks **"where on the screen is the gaze pointing, and is that point inside
the screen rectangle?"** rather than "how far are the eyes from neutral." Each frame we
trace a gaze ray from the **live 3D head position** (the facial transformation matrix's
translation = head x/y/distance) along the total gaze angle (head + eye) to the screen
plane, giving a normalized point-of-regard `por:{x,y}` in `[0,1]²`. A 9-point calibration
fits the cm→screen map (`gaze-mapping.mjs`, with automatic yaw/pitch sign detection and
eye-gain search). Because the ray uses the live head position, the same on-screen point
maps to the same `por` regardless of head pose/distance — so moving closer/farther or
tilting the head does **not** create blind spots (proven in `tests/gaze-mapping.test.mjs`).
`decideBoundary` flags when `por` leaves the rectangle by a soft/hard margin; the angular
model (below) is the **fallback** used before calibration / when 3D data is missing.

Fallback (angular): combined head+eye gaze vs a baseline with a soft/hard band, head-gated
eye term, and slow drift adaptation.

## How to run

```bash
cd ai-interview
python -m http.server 8000
# open http://localhost:8000 in Chrome or Edge (STT also needs Chromium/Safari)
```
Must be served over HTTP (ES modules + WASM do not load from file://).

## Calibration modes (toggle on the welcome card)

- **9-point** (default): follow the dot to 9 screen positions; fits the point-of-regard
  gaze→screen map. Best accuracy; head/distance independent.
- **Quick**: 3-second neutral baseline → angular fallback model.

## Tuning

Point-of-regard (`gaze-mapping.mjs` → `DEFAULT_BOUNDARY`):
- `SOFT_MARGIN` (0.06) / `HARD_MARGIN` (0.14): how far `por` must be OUTSIDE [0,1] to
  cue (soft) / flag (hard). Lower = more sensitive at the screen edge.

Angular fallback (`gaze-math.mjs` → `DEFAULT_THRESHOLDS`):
- `EYE_GAIN_X` (50) / `EYE_GAIN_Y` (30), `HEAD_GATE_DEG` (25), `SOFT_YAW`/`TOL_YAW`,
  `SOFT_PITCH_DOWN`/`TOL_PITCH_DOWN`, `SOFT_PITCH_UP`/`TOL_PITCH_UP`, `INVERT_YAW` (true).

## Automated verification performed

- `node --test` → 57/57 unit tests pass (point-of-regard fit + head/distance independence,
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
