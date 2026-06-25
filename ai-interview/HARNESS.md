# Verification Harness (Layer 1) — autonomous tuning loop

The detection behavior depends on a live webcam the agent can't see. This harness
**decouples the decision logic from the camera** so it can be replayed and scored
headlessly, letting the agent iterate (tune → verify → repeat) without a human in
the loop every time.

## Pieces
- `js/detection-core.mjs` — DOM-free state machine. Reuses the **same** `gaze-math`
  decision functions the live engine uses (`decideLookingAway`, `adaptBaseline`) plus
  the episode/sustain logic that mirrors `proctor.js`. `createDetector()` +
  `replayTrace()`. This is the single source of truth the harness scores, so passing
  the harness reflects the real decision logic.
- `tools/gen-synthetic.mjs` — writes labeled synthetic traces to `corpus/` (Layer 2;
  approximations, good for breadth/regression).
- `tools/eval.mjs` — replays every trace in a corpus, computes metrics, prints a
  table, and exits non-zero on failure (so a loop can branch).
- `tools/replay.mjs` — replays one trace with a level timeline + incidents; supports
  threshold overrides for quick what-ifs.
- `tests/detection-core.test.mjs` — unit tests for the state machine.

## The loop the agent runs
```bash
cd ai-interview
node tools/gen-synthetic.mjs           # once: seed corpus/ (or record real traces, below)
node tools/eval.mjs corpus             # metrics + PASS/FAIL (exit 0/1)
# read metrics → edit thresholds in js/gaze-math.mjs (or timing in detection-core) →
node tools/eval.mjs corpus             # re-verify; repeat until PASS
```
Quick what-if without editing files:
```bash
node tools/replay.mjs corpus/off-left.json --SOFT_YAW=20 --EYE_GAIN_X=55
```
Metrics & targets (in `eval.mjs`): false positives (on-screen traces that flagged),
recall (off-screen detected), direction accuracy, detection latency.

## Recording REAL labeled traces (ground-truth fidelity)
Synthetic traces approximate the signal. For true fidelity, record your own once,
in the browser console **after calibration**, while performing each scenario:
```js
Proctor.startRecording("off-left")
// ...look off the left edge for ~3s...
Proctor.stopRecording({ flag: true, direction: "left", offFromMs: 300 })

Proctor.startRecording("on-edge-left")
// ...read along the LEFT border of the screen (on-screen) for ~3s...
Proctor.stopRecording({ flag: false })
```
Each call downloads a `trace-*.json`. Drop them into `corpus/` and run `eval.mjs`.
Record a spread: on-center, all four on-screen edges (false-positive traps), genuine
off-screen in each direction, face-lost, and a posture-drift segment. The recorded
trace stores your calibration baseline; `eval.mjs` applies the *current* `gaze-math`
thresholds, so editing thresholds + re-running tests your tuning against real data.

## Notes / limits
- The decision + adaptation math is shared with the app (faithful). The episode/timer
  logic in `detection-core.mjs` mirrors `proctor.js`; keep them in sync if you change
  timing (or refactor proctor to import the core — a clean follow-up).
- This is Layer 1 (signal replay). Layer 3 (Playwright + Chrome fake-camera running
  real MediaPipe on recorded `.y4m` clips) would additionally validate the
  MediaPipe-inference stage end-to-end.
