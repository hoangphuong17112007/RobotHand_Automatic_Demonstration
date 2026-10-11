"""ICP (Iterative Closest Point) khu vuc cho hinh hop chu nhat — xac minh/loc pose 6D.

Khieu hop voi yeu cau: sau khi Mask R-CNN / Transporter chon ung vien, khoi
"ICP check" nay  khop bo (corners) cua linh kien da biet (tu catalog CAD) voi
diem cloud quan sat duoc de (1) lam min (x, y, yaw) quanh truc dung, (2) do
phu hop (residual / inlier) => trung tin vao pose de Deep Grasping dung tinh
thong so cho tung bac tu do.
"""
from __future__ import annotations

import numpy as np


def yaw_from_cloud(pts: np.ndarray) -> float:
    """Uoc luong yaw (rad, quanh truc z) tu PCA tren toa do ngang cua cloud."""
    if pts is None or len(pts) < 3:
        return 0.0
    xy = np.asarray(pts)[:, :2]
    mu = xy.mean(axis=0)
    cov = (xy - mu).T @ (xy - mu) / max(len(xy), 1)
    if np.allclose(cov, 0.0):
        return 0.0
    w, v = np.linalg.eigh(cov)
    return float(np.arctan2(v[1, -1], v[0, -1]))


class BoxIcp:
    """Khop hop chu nhat 2D (mat cat ngang) voi cloud de tinh chinh (x, y, yaw)."""

    def __init__(self, iters: int = 8, max_dyaw: float = 0.06):
        self.iters = int(iters)
        self.max_dyaw = float(max_dyaw)

    @staticmethod
    def _half_direct(pts: np.ndarray, z_top: float, half_h: float) -> np.ndarray:
        """Lay lop diem gan mat tren cua linh kien (bo diem duoi/nen)."""
        if pts is None or len(pts) == 0:
            return pts
        zmed = float(np.median(pts[:, 2]))
        lo = zmed - 0.5 * half_h
        return pts[pts[:, 2] >= lo]

    def _clamp_dists(self, u: np.ndarray, v: np.ndarray,
                     a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        cu = np.clip(u, -a, a)
        cv = np.clip(v, -b, b)
        return cu, cv

    def fit(self, pts: np.ndarray, half, init=(0.0, 0.0, 0.0),
            z_top: float | None = None):
        """Tra ve (x, y, yaw, residual, inlier_frac).

        pts   : (n,3) cloud the gioi cua linh kien.
        half  : (a, b, h) kich thuoc box theo mo hinh.
        init  : (x0, y0, yaw0) khoi tao vi tri + huong.
        z_top : neu cho, loc cloud gan mat tren truoc khi khop.
        """
        if pts is None or len(pts) < 3:
            x, y, yaw = init
            return float(x), float(y), float(yaw), 9.9, 0.0
        pts = np.asarray(pts, dtype=float)
        a, b, h = float(half[0]), float(half[1]), float(half[2])
        if z_top is not None:
            pts = self._half_direct(pts, z_top, h)
        if len(pts) < 3:
            return float(init[0]), float(init[1]), float(init[2]), 9.9, 0.0
        x, y = float(init[0]), float(init[1])
        yaw = float(init[2]) if np.isfinite(init[2]) else 0.0
        for _ in range(self.iters):
            cy, sy = np.cos(yaw), np.sin(yaw)
            R = np.array([[cy, sy], [-sy, cy]])     # world -> local
            loc = (pts[:, :2] - np.array([x, y])) @ R.T
            cu, cv = self._clamp_dists(loc[:, 0], loc[:, 1], a, b)
            err = np.stack([cu - loc[:, 0], cv - loc[:, 1]], axis=1)
            # dich tam: mang mo hinh ve phia cac diem quan sat
            mov = -err.mean(axis=0)
            dx, dy = mov @ R                      # local deltas -> world
            x += dx
            y += dy
            # quay nho: goc de giam thanh phan tiep tuyen cua loi
            e2 = loc * err
            num = float(np.mean(e2[:, 0] * loc[:, 1] - e2[:, 1] * loc[:, 0]))
            den = float(np.mean(loc[:, 0] ** 2 + loc[:, 1] ** 2)) + 1e-9
            dy_ = float(np.clip(num / den, -self.max_dyaw, self.max_dyaw))
            yaw += dy_
        # do phu hop
        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([[cy, sy], [-sy, cy]])
        loc = (pts[:, :2] - np.array([x, y])) @ R.T
        cu, cv = self._clamp_dists(loc[:, 0], loc[:, 1], a, b)
        dist = np.hypot(cu - loc[:, 0], cv - loc[:, 1])
        residual = float(np.sqrt(np.mean(dist ** 2)))
        inside = (np.abs(loc[:, 0]) <= a) & (np.abs(loc[:, 1]) <= b)
        inlier = float(inside.mean())
        return float(x), float(y), float(yaw), residual, inlier