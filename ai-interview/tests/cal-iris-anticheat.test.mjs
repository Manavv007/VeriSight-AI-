import { test } from "node:test";
import assert from "node:assert/strict";
import {
  checkIrisLiveFrame, validateIrisSamples, validateIrisCorners,
  createIrisPinnedTracker, isCenterTarget, IRIS_ANTICHEAT,
} from "../js/cal-iris-anticheat.mjs";

test("isCenterTarget", () => {
  assert.equal(isCenterTarget({ x: 0.5, y: 0.5 }), true);
  assert.equal(isCenterTarget({ x: 0.04, y: 0.05 }), false);
});

test("checkIrisLiveFrame: center dot rejects extreme iris", () => {
  assert.equal(checkIrisLiveFrame(0.2, { x: 0.5, y: 0.5 }, 0), "iris_cheat");
  assert.equal(checkIrisLiveFrame(0.05, { x: 0.5, y: 0.5 }, 0), null);
});

test("checkIrisLiveFrame: left dot rejects iris looking right", () => {
  assert.equal(checkIrisLiveFrame(0.2, { x: 0.04, y: 0.5 }, 0), "iris_cheat");
  assert.equal(checkIrisLiveFrame(-0.1, { x: 0.04, y: 0.5 }, 0), null);
});

test("checkIrisLiveFrame: right dot rejects iris looking left", () => {
  assert.equal(checkIrisLiveFrame(-0.2, { x: 0.96, y: 0.5 }, 0), "iris_cheat");
});

test("validateIrisSamples: center neutrality", () => {
  const samples = [
    { target: { x: 0.5, y: 0.5 }, irisGazeX: 0.2, irisGazeY: 0 },
    { target: { x: 0.96, y: 0.5 }, irisGazeX: 0.15, irisGazeY: 0 },
  ];
  const r = validateIrisSamples(samples);
  assert.equal(r.ok, false);
  assert.equal(r.detail, "center_not_neutral");
});

test("validateIrisSamples: honest 9-point pattern passes", () => {
  const samples = [
    { target: { x: 0.5, y: 0.5 }, irisGazeX: 0.02, irisGazeY: 0 },
    { target: { x: 0.04, y: 0.5 }, irisGazeX: -0.12, irisGazeY: 0 },
    { target: { x: 0.96, y: 0.5 }, irisGazeX: 0.14, irisGazeY: 0 },
  ];
  assert.equal(validateIrisSamples(samples).ok, true);
});

test("validateIrisSamples: overshoot cheat fails", () => {
  const samples = [
    { target: { x: 0.5, y: 0.5 }, irisGazeX: 0, irisGazeY: 0 },
    { target: { x: 0.96, y: 0.5 }, irisGazeX: 0.85, irisGazeY: 0 },
  ];
  assert.equal(validateIrisSamples(samples).ok, false);
});

test("validateIrisCorners", () => {
  const ok = validateIrisCorners([
    { corner: "top-left", irisGazeX: -0.1 },
    { corner: "top-right", irisGazeX: 0.1 },
  ]);
  assert.equal(ok.ok, true);
});

test("validateIrisSamples: blend fallback when iris missing", () => {
  const samples = [
    { target: { x: 0.5, y: 0.5 }, gazeX: 0.02 },
    { target: { x: 0.04, y: 0.5 }, gazeX: -0.12 },
    { target: { x: 0.96, y: 0.5 }, gazeX: 0.14 },
  ];
  assert.equal(validateIrisSamples(samples).ok, true);
});

test("validateIrisSamples: no eye signal fails", () => {
  const samples = [
    { target: { x: 0.5, y: 0.5 } },
    { target: { x: 0.96, y: 0.5 } },
  ];
  const r = validateIrisSamples(samples);
  assert.equal(r.ok, false);
  assert.equal(r.detail, "no_eye_signal");
});
test("createIrisPinnedTracker: head still + iris sweep", () => {
  const tr = createIrisPinnedTracker({
    ...IRIS_ANTICHEAT,
    PINNED_WINDOW: 10,
    PINNED_SUSTAIN_MS: 100,
  });
  let hit = null;
  for (let i = 0; i < 15; i++) {
    const iris = -0.3 + (i % 3) * 0.3;
    hit = tr.tick(1 + (i % 2) * 0.5, iris, i * 50);
    if (hit) break;
  }
  assert.equal(hit, "iris_pinned");
});
