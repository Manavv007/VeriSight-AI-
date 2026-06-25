/* =================================================================
   proctor.js — Proctoring orchestrator (MediaPipe gaze engine)
   Consumes the per-frame signal from window.GazeEngine (gaze-engine.js)
   and provides:
     1. Neutral-baseline calibration (3s "look straight + hold still")
     2. Looking-away detection (head pose OR iris), time-window filtered
     3. Face-lost detection (no face sustained >= 1.5s)
     4. Tab-switch / window-blur detection (visibilitychange + blur)
     5. In-memory audit trail + JSON/CSV export (with head/gaze fields)
     6. Fullscreen alignment, integrity scoring, session clock, summary
   Depends on: gaze-engine.js (window.GazeEngine), interview.js (window.Interview)
   ================================================================= */
(function () {
  "use strict";

  // ---------------------------------------------------------------
  // Tunables (detection thresholds live in gaze-engine.js / gaze-math.mjs)
  // ---------------------------------------------------------------
  const CFG = {
    BASELINE_MS: 3000,       // neutral-baseline capture duration
    AWAY_WARN_MS: 500,       // show visual warning (no permanent flag)
    AWAY_FLAG_MS: 4000,      // write a permanent flag
    FACE_LOST_MS: 1500,      // write a face-lost flag
    ENGINE_TIMEOUT_MS: 12000,// max wait for the engine module to load
    SOFT_MIN_MS: 350,        // soft cue must persist this long before showing (debounce)
    PENALTY: { offscreen: 5, tabswitch: 8, facelost: 6 },
    // 4-corner calibration (optional precise mode)
    CORNER_READY_MS: 800,    // time to move eyes to the next corner before capturing
    CORNER_HOLD_MS: 1100,    // per-corner gaze capture window
    CORNER_SOFT_MARGIN: 1.2, // soft band = on-screen combined-gaze edge × this
    CORNER_HARD_MARGIN: 1.55, // hard band = edge × this (tolerance = hard − soft)
    CORNER_FLOORS: { SOFT_YAW: 15, TOL_YAW: 8, SOFT_PITCH: 14, TOL_PITCH: 8 },
  };

  // ---------------------------------------------------------------
  // State
  // ---------------------------------------------------------------
  const S = {
    calibrated: false,
    fullscreenPref: true,
    calMode: "corners",      // "corners" (4-corner, screen-aware — default) | "quick" (3s baseline)
    sessionStart: 0,
    baseline: null,
    lastSignal: null,
    incidents: [],
    offscreen: { active: false, start: 0, warned: false, incident: null, dir: null },
    faceLost: { active: false, start: 0, incident: null },
    focusLost: { active: false, start: 0, incident: null },
    softActive: false,
    softStart: 0,
    windowFocused: true,
    clockTimer: null,
    recording: false,
    recBuf: [],
    recStart: 0,
    recLabel: "trace",
  };

  // ---------------------------------------------------------------
  // DOM
  // ---------------------------------------------------------------
  const el = {};
  function cacheDom() {
    const ids = [
      "calibration-overlay", "cal-center", "cal-countdown", "cal-center-label",
      "cal-start-btn", "cal-mode-btn", "cal-mode-state", "cal-skip-fs", "cal-fs-state", "cal-status", "cal-card", "cal-accuracy-hint",
      "gaze-warning", "gaze-warning-text", "app",
      "focus-chip", "focus-state", "accuracy-value",
      "btn-fullscreen", "btn-recalibrate", "btn-finish",
      "camera-host", "camera-placeholder", "gaze-badge",
      "cam-video", "cam-overlay", "hud-telemetry", "hud-yaw", "hud-pitch", "hud-gaze",
      "btn-download-json", "btn-download-csv",
      "metric-flags", "metric-offscreen", "metric-tabswitch", "metric-facelost",
      "incident-list", "incident-empty", "integrity-score", "session-clock",
      "summary-overlay", "summary-grid", "summary-json", "summary-csv", "summary-close",
    ];
    ids.forEach((id) => { el[id] = document.getElementById(id); });
    el.cameraCard = document.querySelector(".camera-card");
  }

  // ---------------------------------------------------------------
  // Time helpers
  // ---------------------------------------------------------------
  const now = () => performance.now();
  function clockStr(d = new Date()) {
    const p = (n) => String(n).padStart(2, "0");
    return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }
  function durLabel(ms) {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  // ===============================================================
  // Engine lifecycle
  // ===============================================================
  function ensureEngine() {
    if (window.GazeEngine) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const to = setTimeout(
        () => reject(new Error("Gaze engine failed to load (check js/gaze-engine.js and vendored MediaPipe).")),
        CFG.ENGINE_TIMEOUT_MS
      );
      document.addEventListener("gazeengine:loaded", () => { clearTimeout(to); resolve(); }, { once: true });
    });
  }

  async function startEngine() {
    setStatus("Loading face model and camera…");
    await ensureEngine();
    await window.GazeEngine.setup(el["cam-video"], el["cam-overlay"]);
    if (el["camera-placeholder"]) el["camera-placeholder"].style.display = "none";
    if (el["hud-telemetry"]) el["hud-telemetry"].hidden = false;
    window.GazeEngine.start(handleSignal);
  }

  // ===============================================================
  // Calibration (quick neutral baseline OR 4-corner)
  // ===============================================================
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  function positionCenter(lx, ty) { const c = el["cal-center"]; if (c) { c.style.left = lx + "%"; c.style.top = ty + "%"; } }
  function resetCenter() { const c = el["cal-center"]; if (c) { c.style.left = ""; c.style.top = ""; } }
  function setCenterLabel(t) { if (el["cal-center-label"]) el["cal-center-label"].textContent = t; }

  // Quick mode: 3-second neutral baseline ("look straight, hold still").
  function runBaseline() {
    return new Promise((resolve, reject) => {
      resetCenter();
      setCenterLabel("Look here & hold still");
      if (el["cal-card"]) el["cal-card"].hidden = true;
      if (el["cal-center"]) el["cal-center"].classList.add("show");
      setStatus("Look straight at the screen and hold still…");

      let remaining = Math.ceil(CFG.BASELINE_MS / 1000);
      if (el["cal-countdown"]) el["cal-countdown"].textContent = String(remaining);
      const cd = setInterval(() => {
        remaining -= 1;
        if (el["cal-countdown"]) el["cal-countdown"].textContent = String(Math.max(remaining, 0));
        if (remaining <= 0) clearInterval(cd);
      }, 1000);

      window.GazeEngine.captureBaseline(CFG.BASELINE_MS)
        .then((avg) => {
          clearInterval(cd);
          if (el["cal-center"]) el["cal-center"].classList.remove("show");
          finishCalibration({
            mode: "quick",
            samples: avg.samples,
            baseline: { yaw: avg.yaw, pitch: avg.pitch, gazeX: avg.gazeX, gazeY: avg.gazeY },
          });
          resolve(avg);
        })
        .catch((err) => {
          clearInterval(cd);
          if (el["cal-center"]) el["cal-center"].classList.remove("show");
          if (el["cal-card"]) el["cal-card"].hidden = false;
          setStatus(err && err.message ? err.message : "Calibration failed. Try again.", "error");
          reject(err);
        });
    });
  }

  // Precise mode: look at each screen corner so thresholds match the real screen
  // size and seating distance.
  async function runCornerCalibration() {
    if (el["cal-card"]) el["cal-card"].hidden = true;
    if (el["cal-countdown"]) el["cal-countdown"].textContent = "";
    if (el["cal-center"]) el["cal-center"].classList.add("show");

    const seq = [
      ["top-left", 8, 12], ["top-right", 92, 12],
      ["bottom-left", 8, 88], ["bottom-right", 92, 88],
    ];
    const corners = [];
    try {
      for (let i = 0; i < seq.length; i++) {
        const [name, lx, ty] = seq[i];
        positionCenter(lx, ty);
        setCenterLabel(`Look at the dot (${i + 1}/4)`);
        setStatus(`Calibrating corner ${i + 1} of 4 — keep your gaze on the dot…`);
        await sleep(CFG.CORNER_READY_MS);
        const s = await window.GazeEngine.captureWindow(CFG.CORNER_HOLD_MS);
        corners.push(Object.assign({ corner: name }, s));
      }
    } catch (err) {
      resetCenter();
      if (el["cal-center"]) el["cal-center"].classList.remove("show");
      if (el["cal-card"]) el["cal-card"].hidden = false;
      setStatus(err && err.message ? err.message : "Calibration failed. Try again.", "error");
      throw err;
    }
    resetCenter();
    setCenterLabel("Look here & hold still");
    if (el["cal-center"]) el["cal-center"].classList.remove("show");

    const res = window.GazeEngine.calibrateFromCorners(corners, {
      floors: CFG.CORNER_FLOORS,
      softMargin: CFG.CORNER_SOFT_MARGIN,
      hardMargin: CFG.CORNER_HARD_MARGIN,
    });
    if (!res) {
      if (el["cal-card"]) el["cal-card"].hidden = false;
      setStatus("Could not derive calibration from corners. Try again.", "error");
      throw new Error("derive failed");
    }
    finishCalibration({ mode: "corners", samples: corners.length, baseline: res.baseline });
  }

  function finishCalibration(info) {
    S.calibrated = true;
    S.baseline = info.baseline;
    S.calMode = info.mode || S.calMode;
    S.sessionStart = Date.now();

    if (el["accuracy-value"]) el["accuracy-value"].textContent = "Live ✓";
    if (el["cal-accuracy-hint"]) {
      el["cal-accuracy-hint"].hidden = false;
      if (info.mode === "corners") {
        const t = window.GazeEngine ? window.GazeEngine.getThresholds() : {};
        const hardYaw = Math.round((t.SOFT_YAW || 0) + (t.TOL_YAW || 0));
        const hardPitch = Math.round((t.SOFT_PITCH_DOWN || 0) + (t.TOL_PITCH_DOWN || 0));
        el["cal-accuracy-hint"].textContent =
          `4-corner calibration set custom thresholds (looking-away at ~${hardYaw}° horizontal / ~${hardPitch}° vertical combined gaze). Recalibrate any time from the top bar.`;
      } else {
        el["cal-accuracy-hint"].textContent =
          `Baseline captured from ${info.samples} frames. Recalibrate any time from the top bar.`;
      }
    }

    el["calibration-overlay"].classList.add("hide");
    el["calibration-overlay"].setAttribute("aria-hidden", "true");
    el.app.classList.add("show");
    el.app.setAttribute("aria-hidden", "false");

    startSessionClock();
    attachProctorListeners();
    document.dispatchEvent(new CustomEvent("gazeproctor:ready", { detail: { baseline: info.baseline, mode: S.calMode } }));
  }

  // ===============================================================
  // Per-frame signal consumer
  // ===============================================================
  function handleSignal(sig) {
    // Engine draws the overlay every frame regardless of calibration state.
    updateTelemetry(sig);
    if (S.recording) {
      S.recBuf.push({
        tMs: Date.now() - S.recStart,
        yaw: sig.yaw, pitch: sig.pitch, gazeX: sig.gazeX, gazeY: sig.gazeY,
        facePresent: sig.facePresent,
      });
    }
    if (!S.calibrated) return;

    // While the window is unfocused / tab hidden, focus-loss tracking owns the
    // signal; suppress gaze-based episodes to avoid double counting.
    if (!S.windowFocused || document.hidden) {
      clearSoft();
      closeOffscreen(true);
      closeFaceLost(true);
      return;
    }

    S.lastSignal = sig;

    if (!sig.facePresent) {
      clearSoft();
      closeOffscreen(true);
      handleFaceLost();
      return;
    }
    closeFaceLost(false);

    if (sig.level === "hard") {
      S.softStart = 0;
      clearSoft();
      handleOffscreen(sig);
    } else if (sig.level === "soft") {
      closeOffscreen(false); // borderline → visual cue only, no permanent flag
      if (!S.softStart) S.softStart = now();
      if (now() - S.softStart >= CFG.SOFT_MIN_MS) showSoft(sig);
    } else {
      S.softStart = 0;
      clearSoft();
      closeOffscreen(false);
    }
  }

  // Soft zone = "possibly looking away": a non-penalizing visual cue, never logged.
  function showSoft(sig) {
    S.softActive = true;
    setFocusState("warn");
    setGazeBadge("warn", "Possibly away");
    showWarning(true, "Possibly looking away");
    if (el.cameraCard) el.cameraCard.classList.add("alert");
  }
  function clearSoft() {
    if (!S.softActive) return;
    S.softActive = false;
    if (!S.offscreen.active && !S.faceLost.active && !S.focusLost.active) {
      showWarning(false);
      if (el.cameraCard) el.cameraCard.classList.remove("alert");
      setFocusState("ok");
      setGazeBadge("ok", "On screen");
    }
  }

  function updateTelemetry(sig) {
    if (!el["hud-telemetry"] || el["hud-telemetry"].hidden) return;
    const cy = sig.combinedYaw != null ? Math.round(sig.combinedYaw) : 0;
    const cp = sig.combinedPitch != null ? Math.round(sig.combinedPitch) : 0;
    if (el["hud-yaw"]) el["hud-yaw"].textContent = sig.facePresent ? String(cy) : "—";
    if (el["hud-pitch"]) el["hud-pitch"].textContent = sig.facePresent ? String(cp) : "—";
    if (el["hud-gaze"]) el["hud-gaze"].textContent = sig.facePresent ? (sig.level || "none") : "—";
  }

  const DIR_TEXT = {
    left: "Looked to the left of the screen",
    right: "Looked to the right of the screen",
    up: "Looked up, away from the screen",
    down: "Looked down (toward desk or phone)",
  };

  function awayDetail(dir, confidence) {
    const base = DIR_TEXT[dir] || "Looked away from the screen";
    const pct = confidence != null ? ` (~${Math.round(confidence * 100)}% confidence)` : "";
    return base + pct;
  }

  function handleOffscreen(sig) {
    const t = now();
    if (!S.offscreen.active) {
      S.offscreen = { active: true, start: t, incident: null, dir: sig.direction };
      // Hard level is already high-confidence → cue immediately.
      setFocusState("warn");
      setGazeBadge("warn", "Looking away");
      showWarning(true, "Looking away");
      if (el.cameraCard) el.cameraCard.classList.add("alert");
    }
    const elapsed = t - S.offscreen.start;

    if (elapsed >= CFG.AWAY_FLAG_MS && !S.offscreen.incident) {
      setFocusState("bad");
      S.offscreen.incident = logIncident({
        type: "offscreen",
        details: awayDetail(S.offscreen.dir, sig.confidence),
        durationMs: elapsed,
        metrics: snapshot(sig),
      });
    } else if (S.offscreen.incident) {
      updateIncidentDuration(S.offscreen.incident, elapsed);
    }
  }

  function closeOffscreen(silent) {
    if (!S.offscreen.active) return;
    if (S.offscreen.incident) updateIncidentDuration(S.offscreen.incident, now() - S.offscreen.start);
    S.offscreen = { active: false, start: 0, incident: null, dir: null };
    if (!S.softActive) showWarning(false);
    if (el.cameraCard && !S.softActive) el.cameraCard.classList.remove("alert");
    if (!S.faceLost.active && !S.focusLost.active && !S.softActive) {
      setFocusState("ok");
      setGazeBadge("ok", "On screen");
    }
  }

  function handleFaceLost() {
    const t = now();
    if (!S.faceLost.active) S.faceLost = { active: true, start: t, incident: null };
    const elapsed = t - S.faceLost.start;
    if (elapsed >= CFG.FACE_LOST_MS) {
      if (!S.faceLost.incident) {
        setFocusState("bad");
        setGazeBadge("bad", "Face lost");
        showWarning(true, "Face not detected");
        S.faceLost.incident = logIncident({
          type: "facelost",
          details: "Face left the camera view or was obscured",
          durationMs: elapsed,
        });
      } else {
        updateIncidentDuration(S.faceLost.incident, elapsed);
      }
    }
  }

  function closeFaceLost(silent) {
    if (!S.faceLost.active) return;
    if (S.faceLost.incident) updateIncidentDuration(S.faceLost.incident, now() - S.faceLost.start);
    S.faceLost = { active: false, start: 0, incident: null };
    if (!S.offscreen.active && !S.focusLost.active) {
      showWarning(false);
      setFocusState("ok");
      setGazeBadge("ok", "On screen");
    }
  }

  function snapshot(sig) {
    return {
      yaw: Math.round(sig.yaw * 10) / 10,
      pitch: Math.round(sig.pitch * 10) / 10,
      roll: Math.round(sig.roll * 10) / 10,
      gazeX: Math.round(sig.gazeX * 100) / 100,
      gazeY: Math.round(sig.gazeY * 100) / 100,
      combinedYaw: sig.combinedYaw != null ? Math.round(sig.combinedYaw * 10) / 10 : null,
      combinedPitch: sig.combinedPitch != null ? Math.round(sig.combinedPitch * 10) / 10 : null,
      confidence: sig.confidence != null ? Math.round(sig.confidence * 100) / 100 : null,
      level: sig.level || null,
      direction: sig.direction || null,
    };
  }

  // ===============================================================
  // Tab switch / window blur
  // ===============================================================
  function onFocusLostStart(reason) {
    if (S.focusLost.active) return;
    S.focusLost = { active: true, start: now(), incident: null };
    setFocusState("bad");
    setGazeBadge("bad", "Window left");
    S.focusLost.incident = logIncident({ type: "tabswitch", details: reason, durationMs: 0 });
  }

  function onFocusLostEnd() {
    if (!S.focusLost.active) return;
    if (S.focusLost.incident) updateIncidentDuration(S.focusLost.incident, now() - S.focusLost.start);
    S.focusLost = { active: false, start: 0, incident: null };
    if (!S.offscreen.active && !S.faceLost.active) {
      setFocusState("ok");
      setGazeBadge("ok", "On screen");
    }
  }

  function attachProctorListeners() {
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) onFocusLostStart("Switched tab or minimized the window");
      else onFocusLostEnd();
    });
    window.addEventListener("blur", () => {
      S.windowFocused = false;
      onFocusLostStart("Browser window lost focus");
    });
    window.addEventListener("focus", () => {
      S.windowFocused = true;
      onFocusLostEnd();
    });
  }

  // ===============================================================
  // Incident log + metrics
  // ===============================================================
  const LABELS = {
    offscreen: { label: "Looking Away", icon: "👀" },
    tabswitch: { label: "Focus Lost / Tab Switch", icon: "🪟" },
    facelost: { label: "Face Lost", icon: "🚫" },
  };

  function logIncident({ type, details, durationMs, metrics }) {
    const meta = LABELS[type];
    const d = new Date();
    const incident = {
      id: S.incidents.length + 1,
      type,
      label: meta.label,
      icon: meta.icon,
      timestamp: clockStr(d),
      iso: d.toISOString(),
      sessionMs: S.sessionStart ? Date.now() - S.sessionStart : 0,
      durationMs: Math.round(durationMs || 0),
      details: details || "",
      metrics: metrics || null,
    };
    S.incidents.push(incident);
    renderIncident(incident);
    updateMetrics();
    return incident;
  }

  function updateIncidentDuration(incident, ms) {
    incident.durationMs = Math.round(ms);
    const node = document.querySelector(`[data-incident="${incident.id}"] .incident-dur`);
    if (node) node.textContent = durLabel(incident.durationMs);
  }

  function renderIncident(incident) {
    if (el["incident-empty"]) el["incident-empty"].style.display = "none";
    const li = document.createElement("li");
    li.className = `incident type-${incident.type}`;
    li.setAttribute("data-incident", String(incident.id));
    li.innerHTML = `
      <span class="incident-icon">${incident.icon}</span>
      <span class="incident-body">
        <span class="incident-type">${incident.label}</span>
        <span class="incident-detail">${escapeHtml(incident.details)}</span>
      </span>
      <span class="incident-meta">
        <span class="incident-time">${incident.timestamp}</span><br />
        <span class="incident-dur">${incident.durationMs ? durLabel(incident.durationMs) : "—"}</span>
      </span>`;
    el["incident-list"].prepend(li);
  }

  function counts() {
    const c = { offscreen: 0, tabswitch: 0, facelost: 0 };
    S.incidents.forEach((i) => { c[i.type] = (c[i.type] || 0) + 1; });
    return c;
  }

  function integrityScore() {
    const c = counts();
    const penalty =
      c.offscreen * CFG.PENALTY.offscreen +
      c.tabswitch * CFG.PENALTY.tabswitch +
      c.facelost * CFG.PENALTY.facelost;
    return Math.max(0, 100 - penalty);
  }

  function updateMetrics() {
    const c = counts();
    if (el["metric-flags"]) el["metric-flags"].textContent = String(S.incidents.length);
    if (el["metric-offscreen"]) el["metric-offscreen"].textContent = String(c.offscreen);
    if (el["metric-tabswitch"]) el["metric-tabswitch"].textContent = String(c.tabswitch);
    if (el["metric-facelost"]) el["metric-facelost"].textContent = String(c.facelost);

    const score = integrityScore();
    const node = el["integrity-score"];
    if (node) {
      node.textContent = `Integrity: ${score}%`;
      node.classList.remove("warn", "bad");
      if (score < 60) node.classList.add("bad");
      else if (score < 85) node.classList.add("warn");
    }
  }

  // ===============================================================
  // UI helpers
  // ===============================================================
  function setStatus(msg, kind) {
    const node = el["cal-status"];
    if (!node) return;
    node.textContent = msg;
    node.classList.remove("is-error", "is-ok");
    if (kind === "error") node.classList.add("is-error");
    if (kind === "ok") node.classList.add("is-ok");
  }

  function setFocusState(stateName) {
    if (!el["focus-chip"]) return;
    el["focus-chip"].setAttribute("data-state", stateName);
    const map = { ok: "Focused", warn: "Looking away", bad: "Off screen" };
    if (el["focus-state"]) el["focus-state"].textContent = map[stateName] || "Focused";
  }

  function setGazeBadge(stateName, text) {
    if (!el["gaze-badge"]) return;
    el["gaze-badge"].setAttribute("data-state", stateName);
    el["gaze-badge"].textContent = text;
  }

  function showWarning(show, text) {
    if (!el["gaze-warning"]) return;
    if (text && el["gaze-warning-text"]) el["gaze-warning-text"].textContent = text;
    el["gaze-warning"].classList.toggle("show", !!show);
    el["gaze-warning"].setAttribute("aria-hidden", show ? "false" : "true");
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  // ===============================================================
  // Session clock
  // ===============================================================
  function startSessionClock() {
    if (S.clockTimer) clearInterval(S.clockTimer);
    S.clockTimer = setInterval(() => {
      const secs = Math.floor((Date.now() - S.sessionStart) / 1000);
      const m = String(Math.floor(secs / 60)).padStart(2, "0");
      const s = String(secs % 60).padStart(2, "0");
      if (el["session-clock"]) el["session-clock"].textContent = `${m}:${s}`;
    }, 1000);
  }

  // ===============================================================
  // Fullscreen
  // ===============================================================
  function requestFullscreen() {
    const root = document.documentElement;
    const fn = root.requestFullscreen || root.webkitRequestFullscreen || root.msRequestFullscreen;
    if (fn) return fn.call(root).catch(() => { });
    return Promise.resolve();
  }
  function exitFullscreen() {
    const fn = document.exitFullscreen || document.webkitExitFullscreen || document.msExitFullscreen;
    if (document.fullscreenElement && fn) fn.call(document);
  }
  function toggleFullscreen() {
    if (document.fullscreenElement) exitFullscreen();
    else requestFullscreen();
  }

  // ===============================================================
  // Report building + export
  // ===============================================================
  function buildReport() {
    const c = counts();
    const progress = window.Interview ? window.Interview.getProgress() : { answered: 0, total: 0 };
    const responses = window.Interview ? window.Interview.getResponses() : [];
    const thresholds = window.GazeEngine ? window.GazeEngine.getThresholds() : null;
    const endedAt = new Date();
    const durationSec = S.sessionStart ? Math.floor((Date.now() - S.sessionStart) / 1000) : 0;
    return {
      meta: {
        app: "GazeProctor",
        engine: "MediaPipe FaceLandmarker",
        generatedAt: endedAt.toISOString(),
        sessionDurationSec: durationSec,
        baseline: S.baseline
          ? {
            yaw: round1(S.baseline.yaw), pitch: round1(S.baseline.pitch),
            gazeX: round2(S.baseline.gazeX), gazeY: round2(S.baseline.gazeY)
          }
          : null,
        thresholds,
      },
      summary: {
        totalFlags: S.incidents.length,
        lookingAway: c.offscreen,
        tabSwitches: c.tabswitch,
        faceLost: c.facelost,
        integrityScore: integrityScore(),
        questionsAnswered: progress.answered,
        questionsTotal: progress.total,
      },
      incidents: S.incidents.map((i) => ({
        id: i.id,
        type: i.type,
        label: i.label,
        timestamp: i.timestamp,
        iso: i.iso,
        sessionMs: i.sessionMs,
        durationMs: i.durationMs,
        details: i.details,
        direction: i.metrics ? i.metrics.direction : null,
        level: i.metrics ? i.metrics.level : null,
        confidence: i.metrics ? i.metrics.confidence : null,
        yaw: i.metrics ? i.metrics.yaw : null,
        pitch: i.metrics ? i.metrics.pitch : null,
        roll: i.metrics ? i.metrics.roll : null,
        gazeX: i.metrics ? i.metrics.gazeX : null,
        gazeY: i.metrics ? i.metrics.gazeY : null,
        combinedYaw: i.metrics ? i.metrics.combinedYaw : null,
        combinedPitch: i.metrics ? i.metrics.combinedPitch : null,
      })),
      interview: responses,
    };
  }

  const round1 = (v) => Math.round(v * 10) / 10;
  const round2 = (v) => Math.round(v * 100) / 100;

  function download(filename, text, mime) {
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportJSON() {
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    download(`proctoring-report-${stamp}.json`, JSON.stringify(buildReport(), null, 2), "application/json");
  }

  function csvCell(v) {
    const s = String(v == null ? "" : v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }

  function exportCSV() {
    const report = buildReport();
    const lines = [];
    lines.push("GazeProctor Proctoring Report");
    lines.push(`Generated,${csvCell(report.meta.generatedAt)}`);
    lines.push(`Engine,${csvCell(report.meta.engine)}`);
    lines.push(`Session Duration (s),${report.meta.sessionDurationSec}`);
    lines.push(`Integrity Score,${report.summary.integrityScore}%`);
    lines.push(`Total Flags,${report.summary.totalFlags}`);
    lines.push(`Looking Away,${report.summary.lookingAway}`);
    lines.push(`Tab Switches,${report.summary.tabSwitches}`);
    lines.push(`Face Lost,${report.summary.faceLost}`);
    lines.push("");
    lines.push("Incident #,Type,Timestamp,Duration,Direction,Level,Confidence,Yaw,Pitch,Roll,GazeX,GazeY,CombinedYaw,CombinedPitch,Details");
    report.incidents.forEach((i) => {
      lines.push([
        i.id, csvCell(i.label), csvCell(i.timestamp), csvCell(durLabel(i.durationMs)),
        csvCell(i.direction), csvCell(i.level),
        i.confidence != null ? Math.round(i.confidence * 100) + "%" : "",
        i.yaw, i.pitch, i.roll, i.gazeX, i.gazeY, i.combinedYaw, i.combinedPitch, csvCell(i.details),
      ].join(","));
    });
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    download(`proctoring-report-${stamp}.csv`, lines.join("\n"), "text/csv");
  }

  // ===============================================================
  // Finish / summary
  // ===============================================================
  function finishSession() {
    try { window.GazeEngine && window.GazeEngine.pause(); } catch (_) { }
    if (S.clockTimer) clearInterval(S.clockTimer);
    showWarning(false);
    if (el.cameraCard) el.cameraCard.classList.remove("alert");

    const report = buildReport();
    if (el["summary-grid"]) {
      const tiles = [
        { v: report.summary.integrityScore + "%", l: "Integrity Score" },
        { v: report.summary.totalFlags, l: "Total Flags" },
        { v: report.summary.lookingAway, l: "Looking Away" },
        { v: report.summary.tabSwitches, l: "Tab Switches" },
        { v: report.summary.faceLost, l: "Face Lost" },
        { v: `${report.summary.questionsAnswered}/${report.summary.questionsTotal}`, l: "Questions Answered" },
      ];
      el["summary-grid"].innerHTML = tiles
        .map((t) => `<div class="summary-tile"><div class="st-value">${t.v}</div><div class="st-label">${t.l}</div></div>`)
        .join("");
    }
    if (el["summary-overlay"]) {
      el["summary-overlay"].hidden = false;
      el["summary-overlay"].classList.remove("hide");
      el["summary-overlay"].setAttribute("aria-hidden", "false");
    }
  }

  function closeSummary() {
    if (el["summary-overlay"]) {
      el["summary-overlay"].classList.add("hide");
      el["summary-overlay"].setAttribute("aria-hidden", "true");
      setTimeout(() => { el["summary-overlay"].hidden = true; }, 450);
    }
  }

  // ===============================================================
  // Recalibration
  // ===============================================================
  function recalibrate() {
    S.calibrated = false;
    closeOffscreen(true);
    closeFaceLost(true);
    onFocusLostEnd();
    if (S.clockTimer) clearInterval(S.clockTimer);

    el.app.classList.remove("show");
    el.app.setAttribute("aria-hidden", "true");
    el["calibration-overlay"].classList.remove("hide");
    el["calibration-overlay"].setAttribute("aria-hidden", "false");

    // Camera + engine already running → recapture directly in the chosen mode.
    if (window.GazeEngine && window.GazeEngine.isReady()) {
      (S.calMode === "corners" ? runCornerCalibration() : runBaseline()).catch(() => { });
    } else if (el["cal-card"]) {
      el["cal-card"].hidden = false;
    }
  }

  // ===============================================================
  // Init / wiring
  // ===============================================================
  function bind() {
    el["cal-start-btn"].addEventListener("click", async () => {
      el["cal-start-btn"].disabled = true;
      if (S.fullscreenPref) await requestFullscreen();
      try {
        await startEngine();
        if (S.calMode === "corners") await runCornerCalibration();
        else await runBaseline();
      } catch (err) {
        el["cal-start-btn"].disabled = false;
        if (el["cal-card"]) el["cal-card"].hidden = false;
        setStatus("Could not start: " + (err && err.message ? err.message : "camera permission denied"), "error");
      }
    });

    if (el["cal-mode-btn"]) {
      el["cal-mode-btn"].addEventListener("click", () => {
        S.calMode = S.calMode === "quick" ? "corners" : "quick";
        if (el["cal-mode-state"]) el["cal-mode-state"].textContent = S.calMode === "corners" ? "4-corner" : "Quick";
      });
    }

    el["cal-skip-fs"].addEventListener("click", () => {
      S.fullscreenPref = !S.fullscreenPref;
      el["cal-fs-state"].textContent = S.fullscreenPref ? "On" : "Off";
    });

    el["btn-fullscreen"].addEventListener("click", toggleFullscreen);
    el["btn-recalibrate"].addEventListener("click", recalibrate);
    el["btn-finish"].addEventListener("click", finishSession);

    el["btn-download-json"].addEventListener("click", exportJSON);
    el["btn-download-csv"].addEventListener("click", exportCSV);
    if (el["summary-json"]) el["summary-json"].addEventListener("click", exportJSON);
    if (el["summary-csv"]) el["summary-csv"].addEventListener("click", exportCSV);
    if (el["summary-close"]) el["summary-close"].addEventListener("click", closeSummary);

    document.addEventListener("interview:complete", finishSession);
    window.addEventListener("beforeunload", () => { try { window.GazeEngine && window.GazeEngine.stop(); } catch (_) { } });
  }

  function init() {
    cacheDom();
    if (!el["cal-start-btn"]) return;
    bind();
  }

  // Public API for debugging / external integration.
  window.Proctor = {
    getIncidents: () => S.incidents.slice(),
    getReport: buildReport,
    getBaseline: () => S.baseline,
    exportJSON,
    exportCSV,
    finish: finishSession,
    recalibrate,
    // ---- Trace recording for the headless eval harness (tools/) ----
    // Usage in the browser console, AFTER calibration, while performing a scenario:
    //   Proctor.startRecording("off-left")
    //   ...look off the left edge for a few seconds...
    //   Proctor.stopRecording({ flag: true, direction: "left", offFromMs: 300 })
    // For an on-screen scenario:  Proctor.stopRecording({ flag: false })
    // Drop the downloaded JSON into ai-interview/corpus/ and run: node tools/eval.mjs corpus
    startRecording: (label) => {
      S.recording = true; S.recStart = Date.now(); S.recBuf = []; S.recLabel = label || "trace";
      return "recording: " + S.recLabel;
    },
    stopRecording: (expect) => {
      S.recording = false;
      const trace = {
        label: S.recLabel,
        expect: expect || {},
        meta: {
          baseline: S.baseline,
          thresholds: window.GazeEngine ? window.GazeEngine.getThresholds() : null,
          calMode: S.calMode,
          recordedAt: new Date().toISOString(),
        },
        frames: S.recBuf.slice(),
      };
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      download(`trace-${S.recLabel}-${stamp}.json`, JSON.stringify(trace), "application/json");
      return `saved ${trace.frames.length} frames as trace-${S.recLabel}-…json`;
    },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
