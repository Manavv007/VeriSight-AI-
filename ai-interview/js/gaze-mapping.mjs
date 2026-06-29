/* =================================================================
   gaze-mapping.mjs — point-of-regard (PoR) via normalized ridge regression
   No DOM / no MediaPipe → unit-testable in Node.

   Maps head pose + eye gaze + head position → a normalized on-screen point
   (x,y in [0,1]). We deliberately use a LINEAR, feature-normalized, ridge-
   regularized fit instead of an explicit tan()/distance ray-plane, because the
   geometric form amplifies noisy MediaPipe inputs (a few degrees of head jitter
   blew the point past ±400%). The regression:
     - learns each feature's weight AND SIGN from data (no sign guessing),
     - cancels head movement against the compensating eye movement when those
       are seen together (the head-movement calibration pass provides this),
     - stays bounded/stable (normalized features + ridge).
   Head independence is LEARNED (from the head-movement pass), not assumed.

   Screen-normalized: x,y in [0,1], (0,0) = top-left.
   ================================================================= */

// Feature vectors (raw). X (horizontal): eye gaze + head yaw + head lateral pos + distance.
//                        Y (vertical):   eye gaze + head pitch + head vertical pos + distance.
// Distance (tz) lets the model compensate the candidate moving closer/farther.
function featuresX(s) { return [s.gazeX, s.headYaw, s.tx, s.tz]; }
function featuresY(s) { return [s.gazeY, s.headPitch, s.ty, s.tz]; }

function normStats(rows) {
  const m = rows[0].length;
  const mean = new Array(m).fill(0), std = new Array(m).fill(0);
  for (const r of rows) for (let j = 0; j < m; j++) mean[j] += r[j];
  for (let j = 0; j < m; j++) mean[j] /= rows.length;
  for (const r of rows) for (let j = 0; j < m; j++) std[j] += (r[j] - mean[j]) ** 2;
  for (let j = 0; j < m; j++) std[j] = Math.sqrt(std[j] / rows.length) || 1;
  return { mean, std };
}
const applyNorm = (v, ns) => v.map((x, j) => (x - ns.mean[j]) / ns.std[j]);
const dot = (a, b) => a.reduce((s, ai, i) => s + ai * b[i], 0);

// Ridge least squares; A rows include the bias column at index 0 (NOT regularized).
function ridge(A, b, lambdas) {
  const m = A[0].length;
  const lam = (i) => (Array.isArray(lambdas) ? (lambdas[i] || 0) : (i === 0 ? 0 : (lambdas || 0)));
  const M = Array.from({ length: m }, () => new Array(m).fill(0));
  const v = new Array(m).fill(0);
  for (let r = 0; r < A.length; r++) {
    for (let i = 0; i < m; i++) {
      v[i] += A[r][i] * b[r];
      for (let j = 0; j < m; j++) M[i][j] += A[r][i] * A[r][j];
    }
  }
  for (let i = 0; i < m; i++) M[i][i] += lam(i); // bias (i=0) gets 0
  // Gaussian elimination
  for (let col = 0; col < m; col++) {
    let piv = col;
    for (let r = col + 1; r < m; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    [v[col], v[piv]] = [v[piv], v[col]];
    const d = M[col][col] || 1e-9;
    for (let j = col; j < m; j++) M[col][j] /= d;
    v[col] /= d;
    for (let r = 0; r < m; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let j = col; j < m; j++) M[r][j] -= f * M[col][j];
      v[r] -= f * v[col];
    }
  }
  return v;
}

/* fitCalibration(samples, opts) → { nsX, nsY, cX, cY, residual }
   samples = [{ headYaw, headPitch, gazeX, gazeY, tx, ty, tz, target:{x,y} }] */
export function fitCalibration(samples, opts = {}) {
  // Per-feature ridge. Eye-gaze & head-ANGLE are distance-invariant and stable → light
  // regularization (keep them strong). Head POSITION/DISTANCE (tx,ty,tz) jump far outside
  // the calibrated range when the candidate leans in/out, so the linear model would
  // extrapolate and blow the prediction up (por → 140% / -100%). Shrink them hard so they
  // act only as a gentle correction, never a runaway term.
  const LG = opts.lambdaGaze == null ? 0.3 : opts.lambdaGaze; // gaze + head angle
  const LP = opts.lambdaPos == null ? 6 : opts.lambdaPos;     // tx / ty / tz
  const rawX = samples.map(featuresX), rawY = samples.map(featuresY);
  const nsX = normStats(rawX), nsY = normStats(rawY);
  const AX = rawX.map((r) => [1, ...applyNorm(r, nsX)]);
  const AY = rawY.map((r) => [1, ...applyNorm(r, nsY)]);
  // Column order: [bias, gaze, headAngle, t(x|y), tz]
  const lamX = [0, LG, LG, LP, LP];
  const lamY = [0, LG, LG, LP, LP];
  const cX = ridge(AX, samples.map((s) => s.target.x), lamX);
  const cY = ridge(AY, samples.map((s) => s.target.y), lamY);
  let res = 0;
  for (let i = 0; i < samples.length; i++) {
    res += (dot(cX, AX[i]) - samples[i].target.x) ** 2 + (dot(cY, AY[i]) - samples[i].target.y) ** 2;
  }
  const neutralHead = {
    yaw: samples.reduce((s, x) => s + x.headYaw, 0) / samples.length,
    pitch: samples.reduce((s, x) => s + x.headPitch, 0) / samples.length,
  };
  // Calibrated on-screen extent: the model often compresses gaze into a narrow band
  // (e.g. screen → por.y 0.32..0.56), so a fixed [0,1] boundary never triggers. Fit
  // por ≈ A·target + B per axis over the samples and take the model's output at the
  // true screen edges (target 0 and 1) as the boundary, so "off screen" is relative
  // to where the model ACTUALLY maps the edges.
  const model0 = { nsX, nsY, cX, cY };
  const edge = (axis) => {
    let st = 0, sp = 0, stt = 0, stp = 0;
    const n = samples.length;
    for (const s of samples) {
      const t = s.target[axis];
      const p = predictPoR(model0, s)[axis];
      st += t; sp += p; stt += t * t; stp += t * p;
    }
    const den = n * stt - st * st;
    const A = den !== 0 ? (n * stp - st * sp) / den : 1;
    const B = (sp - A * st) / n;
    const e0 = B, e1 = A + B;
    return e0 <= e1 ? { min: e0, max: e1 } : { min: e1, max: e0 };
  };
  const bx = edge("x"), by = edge("y");
  // Pad the on-screen extent outward: the screen corners need more extreme eye angles
  // than any (inset) calibration dot, and the linear model under-reaches them, so the
  // true corners land a little beyond [min,max]. Padding keeps on-corner looks on-screen.
  const padFrac = opts.boundsPad == null ? 0.12 : opts.boundsPad;
  const px = padFrac * (bx.max - bx.min), py = padFrac * (by.max - by.min);
  return {
    nsX, nsY, cX, cY, residual: res / samples.length, neutralHead,
    bounds: { xMin: bx.min - px, xMax: bx.max + px, yMin: by.min - py, yMax: by.max + py },
  };
}

/* predictPoR(model, s) → { x, y } normalized screen point (clamped to a sane range). */
export function predictPoR(model, s) {
  const fx = [1, ...applyNorm(featuresX(s), model.nsX)];
  const fy = [1, ...applyNorm(featuresY(s), model.nsY)];
  const clamp = (v) => Math.max(-1, Math.min(2, v));
  return { x: clamp(dot(model.cX, fx)), y: clamp(dot(model.cY, fy)) };
}

export const DEFAULT_BOUNDARY = {
  SOFT_MARGIN: 0.06, // PoR this far OUTSIDE [0,1] → "possibly off" (soft)
  HARD_MARGIN: 0.11, // PoR this far outside → "off screen" (hard)
  // Head-deviation trigger (degrees beyond the calibrated neutral head pose).
  // Catches off-screen looks done by MOVING THE HEAD (esp. up — eyes can't roll up —
  // and left/right via head turn), which the PoR's head-compensation suppresses.
  HEAD_YAW_SOFT: 14, HEAD_YAW_HARD: 20,
  HEAD_PITCH_UP_SOFT: 12, HEAD_PITCH_UP_HARD: 16,   // chin-up is easier (up is head-driven)
  HEAD_PITCH_DOWN_SOFT: 14, HEAD_PITCH_DOWN_HARD: 20,
  // Adaptive re-centering: while the gaze is ON screen (level "none"), slowly pull the
  // working centre toward the current PoR. This absorbs SLOW drift from leaning in/out
  // or repositioning (which moves PoR gradually) without absorbing a real off-screen
  // glance (a fast, large jump that crosses the boundary before the centre can follow).
  RECENTER_ALPHA: 0.05,     // per-frame EMA toward centre while on-screen (~1s time const @20fps)
  RECENTER_ENABLED: true,
  // Bound the re-centering so it can absorb DRIFT but never swallow a whole side's margin
  // (which would make looking toward that edge stop flagging):
  RECENTER_MAX_FRAC: 0.35,  // cap |bias| to this fraction of the bounds half-width (per axis)
  RECENTER_BAND_FRAC: 0.70, // only re-center while looking within this fraction of half-width of centre
};

/* decideBoundary(por, cfg, bounds) → { level, direction, confidence, outside, por }
   `bounds` = calibrated on-screen por extent (defaults to the full [0,1] square). */
export function decideBoundary(por, cfg = DEFAULT_BOUNDARY, bounds) {
  const b = bounds || { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
  const outX = por.x < b.xMin ? b.xMin - por.x : por.x > b.xMax ? por.x - b.xMax : 0;
  const outY = por.y < b.yMin ? b.yMin - por.y : por.y > b.yMax ? por.y - b.yMax : 0;
  const outside = Math.max(outX, outY);

  let direction = null;
  if (outX >= outY && outX > 0) direction = por.x < b.xMin ? "left" : "right";
  else if (outY > 0) direction = por.y < b.yMin ? "up" : "down";

  let level = "none";
  if (outside >= cfg.HARD_MARGIN) level = "hard";
  else if (outside >= cfg.SOFT_MARGIN) level = "soft";

  let confidence = 0;
  if (level === "soft") confidence = 0.5 + 0.3 * ((outside - cfg.SOFT_MARGIN) / Math.max(1e-6, cfg.HARD_MARGIN - cfg.SOFT_MARGIN));
  else if (level === "hard") confidence = Math.min(1, 0.8 + 0.2 * ((outside - cfg.HARD_MARGIN) / cfg.HARD_MARGIN));

  return { level, direction: level === "none" ? null : direction, confidence, outside, por };
}


const _rank = { none: 0, soft: 1, hard: 2 };
function _band(mag, soft, hard) {
  if (mag >= hard) return { level: "hard", confidence: Math.min(1, 0.8 + 0.2 * ((mag - hard) / Math.max(1e-6, hard))) };
  if (mag >= soft) return { level: "soft", confidence: 0.5 + 0.3 * ((mag - soft) / Math.max(1e-6, hard - soft)) };
  return { level: "none", confidence: 0 };
}

/* decideHead(s, model, cfg) → { level, direction, confidence }
   Head-YAW deviation trigger only (catches left/right done by turning the head, which
   PoR head-compensation suppresses). Pitch is intentionally NOT used here — up/down is
   handled by the iris-based boundary PoR, so a face tilted up while the eyes stay on
   screen does NOT flag. Direction sign comes from the model's learned cX coefficient. */
export function decideHead(s, model, cfg = DEFAULT_BOUNDARY) {
  const n = (model && model.neutralHead) || { yaw: 0, pitch: 0 };
  const dY = s.headYaw - n.yaw;
  const sX = model && model.cX && model.cX[2] < 0 ? -1 : 1; // headYaw → por.x direction
  const towardRight = dY * sX;
  const h = _band(Math.abs(dY), cfg.HEAD_YAW_SOFT, cfg.HEAD_YAW_HARD);
  return { level: h.level, direction: h.level === "none" ? null : (towardRight >= 0 ? "right" : "left"), confidence: h.confidence };
}

/* decidePoR(model, s, cfg) → { por, level, direction, confidence }
   Primary detection: PoR boundary (eye-driven) OR head-deviation (head-driven),
   whichever is stronger. This is what the engine and harness call. */
export function decidePoR(model, s, cfg = DEFAULT_BOUNDARY) {
  const por = predictPoR(model, s);
  const a = decideBoundary(por, cfg, model && model.bounds);
  const b = decideHead(s, model, cfg);
  let chosen;
  if (_rank[a.level] > _rank[b.level]) chosen = a;
  else if (_rank[b.level] > _rank[a.level]) chosen = b;
  else chosen = a.confidence >= b.confidence ? a : b;
  return { por, level: chosen.level, direction: chosen.level === "none" ? null : chosen.direction, confidence: chosen.confidence };
}

/* ---- Adaptive re-centering (drift absorber) ----------------------------------------
   The PoR mapping amplifies tiny eye/head changes, so leaning in/out or shifting in the
   seat makes PoR drift gradually even though the eyes stay on screen. We track the
   candidate's SETTLED position as a moving centre: while on screen, the working centre
   eases toward the current PoR, cancelling slow drift. A real off-screen glance is a
   fast, large jump that crosses the boundary before the centre can catch up, so it still
   flags (and once flagging, adaptation freezes so the incident persists for the sustain
   timer). */
export function createRecenterState() {
  return { init: false, biasX: 0, biasY: 0 };
}

/* recenterApply(state, rawPor, bounds, cfg) → { adj, decision } (mutates state) */
export function recenterApply(state, raw, bounds, cfg = DEFAULT_BOUNDARY) {
  const b = bounds || { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
  const cx = (b.xMin + b.xMax) / 2, cy = (b.yMin + b.yMax) / 2;
  if (cfg.RECENTER_ENABLED === false) {
    return { adj: raw, decision: decideBoundary(raw, cfg, b) };
  }
  const adj = { x: raw.x + state.biasX, y: raw.y + state.biasY };
  const decision = decideBoundary(adj, cfg, b);
  const halfX = Math.max(1e-3, (b.xMax - b.xMin) / 2), halfY = Math.max(1e-3, (b.yMax - b.yMin) / 2);
  // Only re-center while looking near the middle — an edge look must NOT drag the centre.
  const nearCentre = Math.abs(adj.x - cx) < cfg.RECENTER_BAND_FRAC * halfX
                  && Math.abs(adj.y - cy) < cfg.RECENTER_BAND_FRAC * halfY;
  if (decision.level === "none" && nearCentre) {
    const a = cfg.RECENTER_ALPHA;
    state.biasX += a * (cx - adj.x);
    state.biasY += a * (cy - adj.y);
    const mx = cfg.RECENTER_MAX_FRAC * halfX, my = cfg.RECENTER_MAX_FRAC * halfY;
    state.biasX = Math.max(-mx, Math.min(mx, state.biasX)); // cap so a side's margin survives
    state.biasY = Math.max(-my, Math.min(my, state.biasY));
  }
  return { adj, decision };
}

/* createPoRTracker(model, cfg) → { step(s), reset() }
   Stateful per-session wrapper around decidePoR with adaptive re-centering applied to the
   PoR-boundary branch. Head-yaw trigger is unaffected. Returns the same shape as decidePoR
   (plus porRaw). Engine and detection-core use this so live + replay behave identically. */
export function createPoRTracker(model, cfg = DEFAULT_BOUNDARY) {
  const state = createRecenterState();
  return {
    reset() { state.init = false; state.biasX = 0; state.biasY = 0; },
    step(s) {
      const raw = predictPoR(model, s);
      const { adj, decision: a } = recenterApply(state, raw, model && model.bounds, cfg);
      const b = decideHead(s, model, cfg);
      let chosen;
      if (_rank[a.level] > _rank[b.level]) chosen = a;
      else if (_rank[b.level] > _rank[a.level]) chosen = b;
      else chosen = a.confidence >= b.confidence ? a : b;
      return { por: adj, porRaw: raw, level: chosen.level, direction: chosen.level === "none" ? null : chosen.direction, confidence: chosen.confidence };
    },
  };
}
