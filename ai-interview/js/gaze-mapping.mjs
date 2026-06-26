/* =================================================================
   gaze-mapping.mjs — point-of-regard (PoR) via geometric gaze-ray → screen plane
   No DOM / no MediaPipe → unit-testable in Node.

   Idea (head/distance INDEPENDENT by construction):
   Each frame we know the head's 3D position (tx, ty, tz from MediaPipe's facial
   transformation matrix; tz ≈ distance) and the total gaze angle (head + eye).
   We trace the gaze ray to the screen plane (≈ the camera plane) to get the hit
   point in cm:  X = tx + |tz|·tan(totalYaw),  Y = ty + |tz|·tan(totalPitch).
   Because tx/ty/tz are LIVE, the same on-screen point yields the same X/Y no
   matter where the head is — so moving closer/farther or turning the head does
   NOT shift the boundary. Calibration only fits the cm→normalized-screen map.

   Convention: headYaw/​gazeX are already in a frame where "looking right" is
   positive (the engine applies INVERT_YAW). Screen-normalized: x,y in [0,1],
   (0,0) = top-left, (1,1) = bottom-right.
   ================================================================= */

const DEG = Math.PI / 180;

/* Ray → screen-plane hit point in cm. fx/fy flip head yaw/pitch sign so head and
   eye reinforce (sign convention auto-detected during calibration). */
export function intersect(s, eyeGainDeg, fx = 1, fy = 1) {
  const G = eyeGainDeg;
  const dist = Math.abs(s.tz);
  const totalYaw = fx * s.headYaw + G * s.gazeX;     // degrees
  const totalPitch = fy * s.headPitch + G * s.gazeY; // degrees
  const X = s.tx + dist * Math.tan(totalYaw * DEG);
  const Y = s.ty + dist * Math.tan(totalPitch * DEG);
  return { X, Y };
}

// Solve least-squares for coeffs c in (b ≈ A c), A: n×m rows, via normal equations
// (AᵀA + λI) c = Aᵀb with tiny ridge for stability. Gaussian elimination.
function lstsq(A, b, lambda = 1e-6) {
  const m = A[0].length;
  const M = Array.from({ length: m }, () => new Array(m).fill(0));
  const v = new Array(m).fill(0);
  for (let r = 0; r < A.length; r++) {
    for (let i = 0; i < m; i++) {
      v[i] += A[r][i] * b[r];
      for (let j = 0; j < m; j++) M[i][j] += A[r][i] * A[r][j];
    }
  }
  for (let i = 0; i < m; i++) M[i][i] += lambda;
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

/* fitCalibration(samples) → { eyeGain, cX, cY, residual }
   samples = [{ headYaw, headPitch, gazeX, gazeY, tx, ty, tz, target:{x,y} }]
   Searches eye-gain (deg per gaze unit) and fits a 2D affine cm→screen for each. */
export function fitCalibration(samples, opts = {}) {
  const gLo = opts.gMin == null ? 10 : opts.gMin;
  const gHi = opts.gMax == null ? 45 : opts.gMax;
  const gStep = opts.gStep == null ? 1 : opts.gStep;
  let best = null;
  for (const fx of [1, -1]) {
    for (const fy of [1, -1]) {
      for (let G = gLo; G <= gHi; G += gStep) {
        const A = [], bx = [], by = [];
        for (const s of samples) {
          const { X, Y } = intersect(s, G, fx, fy);
          A.push([1, X, Y]);
          bx.push(s.target.x);
          by.push(s.target.y);
        }
        const cX = lstsq(A, bx);
        const cY = lstsq(A, by);
        let res = 0;
        for (let i = 0; i < A.length; i++) {
          const px = cX[0] + cX[1] * A[i][1] + cX[2] * A[i][2];
          const py = cY[0] + cY[1] * A[i][1] + cY[2] * A[i][2];
          res += (px - bx[i]) ** 2 + (py - by[i]) ** 2;
        }
        res /= samples.length;
        if (!best || res < best.residual) best = { eyeGain: G, fx, fy, cX, cY, residual: res };
      }
    }
  }
  return best;
}

/* predictPoR(model, s) → { x, y } normalized screen point. */
export function predictPoR(model, s) {
  const { X, Y } = intersect(s, model.eyeGain, model.fx, model.fy);
  return {
    x: model.cX[0] + model.cX[1] * X + model.cX[2] * Y,
    y: model.cY[0] + model.cY[1] * X + model.cY[2] * Y,
  };
}

export const DEFAULT_BOUNDARY = {
  SOFT_MARGIN: 0.06, // PoR this far OUTSIDE [0,1] → "possibly off" (soft)
  HARD_MARGIN: 0.14, // PoR this far outside → "off screen" (hard)
};

/* decideBoundary(por, cfg) → { level, direction, confidence, outside }
   How far the point-of-regard is outside the screen rectangle decides soft/hard. */
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
