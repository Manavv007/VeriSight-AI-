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

// Feature vectors (raw). X (horizontal): eye gaze + head yaw + head lateral pos.
//                        Y (vertical):   eye gaze + head pitch + head vertical pos.
function featuresX(s) { return [s.gazeX, s.headYaw, s.tx]; }
function featuresY(s) { return [s.gazeY, s.headPitch, s.ty]; }

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
function ridge(A, b, lambda) {
  const m = A[0].length;
  const M = Array.from({ length: m }, () => new Array(m).fill(0));
  const v = new Array(m).fill(0);
  for (let r = 0; r < A.length; r++) {
    for (let i = 0; i < m; i++) {
      v[i] += A[r][i] * b[r];
      for (let j = 0; j < m; j++) M[i][j] += A[r][i] * A[r][j];
    }
  }
  for (let i = 1; i < m; i++) M[i][i] += lambda; // skip bias (i=0)
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
  const lambda = opts.lambda == null ? 0.3 : opts.lambda;
  const rawX = samples.map(featuresX), rawY = samples.map(featuresY);
  const nsX = normStats(rawX), nsY = normStats(rawY);
  const AX = rawX.map((r) => [1, ...applyNorm(r, nsX)]);
  const AY = rawY.map((r) => [1, ...applyNorm(r, nsY)]);
  const cX = ridge(AX, samples.map((s) => s.target.x), lambda);
  const cY = ridge(AY, samples.map((s) => s.target.y), lambda);
  let res = 0;
  for (let i = 0; i < samples.length; i++) {
    res += (dot(cX, AX[i]) - samples[i].target.x) ** 2 + (dot(cY, AY[i]) - samples[i].target.y) ** 2;
  }
  return { nsX, nsY, cX, cY, residual: res / samples.length };
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
  HARD_MARGIN: 0.14, // PoR this far outside → "off screen" (hard)
};

/* decideBoundary(por, cfg) → { level, direction, confidence, outside, por } */
export function decideBoundary(por, cfg = DEFAULT_BOUNDARY) {
  const outX = por.x < 0 ? -por.x : por.x > 1 ? por.x - 1 : 0;
  const outY = por.y < 0 ? -por.y : por.y > 1 ? por.y - 1 : 0;
  const outside = Math.max(outX, outY);

  let direction = null;
  if (outX >= outY && outX > 0) direction = por.x < 0 ? "left" : "right";
  else if (outY > 0) direction = por.y < 0 ? "up" : "down";

  let level = "none";
  if (outside >= cfg.HARD_MARGIN) level = "hard";
  else if (outside >= cfg.SOFT_MARGIN) level = "soft";

  let confidence = 0;
  if (level === "soft") confidence = 0.5 + 0.3 * ((outside - cfg.SOFT_MARGIN) / Math.max(1e-6, cfg.HARD_MARGIN - cfg.SOFT_MARGIN));
  else if (level === "hard") confidence = Math.min(1, 0.8 + 0.2 * ((outside - cfg.HARD_MARGIN) / cfg.HARD_MARGIN));

  return { level, direction: level === "none" ? null : direction, confidence, outside, por };
}
