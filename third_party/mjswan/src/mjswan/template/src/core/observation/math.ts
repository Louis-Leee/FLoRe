export function normalizeQuat(quat: ArrayLike<number>): number[] {
  const w = quat[0] ?? 1;
  const x = quat[1] ?? 0;
  const y = quat[2] ?? 0;
  const z = quat[3] ?? 0;
  const n = Math.hypot(w, x, y, z);
  if (n < 1e-9) {
    return [1, 0, 0, 0];
  }
  const inv = 1.0 / n;
  return [w * inv, x * inv, y * inv, z * inv];
}

export function quatMultiply(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
  const aw = a[0] ?? 1;
  const ax = a[1] ?? 0;
  const ay = a[2] ?? 0;
  const az = a[3] ?? 0;
  const bw = b[0] ?? 1;
  const bx = b[1] ?? 0;
  const by = b[2] ?? 0;
  const bz = b[3] ?? 0;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

export function quatInverse(quat: ArrayLike<number>): number[] {
  const w = quat[0] ?? 1;
  const x = quat[1] ?? 0;
  const y = quat[2] ?? 0;
  const z = quat[3] ?? 0;
  const normSq = w * w + x * x + y * y + z * z;
  if (normSq < 1e-9) {
    return [1, 0, 0, 0];
  }
  const inv = 1.0 / normSq;
  return [w * inv, -x * inv, -y * inv, -z * inv];
}

export function quatApplyInv(quat: ArrayLike<number>, vec: ArrayLike<number>): number[] {
  const w = quat[0] ?? 1;
  const x = quat[1] ?? 0;
  const y = quat[2] ?? 0;
  const z = quat[3] ?? 0;
  const vx = vec[0] ?? 0;
  const vy = vec[1] ?? 0;
  const vz = vec[2] ?? 0;
  const tx = 2.0 * (y * vz - z * vy);
  const ty = 2.0 * (z * vx - x * vz);
  const tz = 2.0 * (x * vy - y * vx);
  const cx = y * tz - z * ty;
  const cy = z * tx - x * tz;
  const cz = x * ty - y * tx;
  return [vx - w * tx + cx, vy - w * ty + cy, vz - w * tz + cz];
}

export function quatToRot6d(quat: ArrayLike<number>): number[] {
  const [w, x, y, z] = normalizeQuat(quat);
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;

  const r00 = 1.0 - 2.0 * (yy + zz);
  const r01 = 2.0 * (xy - wz);
  const r10 = 2.0 * (xy + wz);
  const r11 = 1.0 - 2.0 * (xx + zz);
  const r20 = 2.0 * (xz - wy);
  const r21 = 2.0 * (yz + wx);

  // mjlab uses `matrix_from_quat(q)[..., :2].reshape(...)`, which flattens the
  // first two matrix columns in row-major order.
  return [r00, r01, r10, r11, r20, r21];
}

export function clampFutureIndices(
  base: number,
  steps: number[],
  length: number
): number[] {
  return steps.map((step) => {
    const idx = base + step;
    if (idx < 0) return 0;
    if (idx >= length) return Math.max(0, length - 1);
    return idx;
  });
}
