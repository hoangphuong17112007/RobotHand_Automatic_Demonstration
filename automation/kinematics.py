"""Dong hoc xuong chieu (FK/IK) cho canh tay 6 bac tu do."""
from __future__ import annotations

import mujoco
import numpy as np

from . import config as C


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / s


def frame_quat(approach: np.ndarray, close: np.ndarray) -> np.ndarray:
    a = approach / np.linalg.norm(approach)
    c = close - np.dot(close, a) * a
    c /= np.linalg.norm(c)
    R = np.column_stack((a, c, np.cross(a, c)))
    return rot_to_quat(R)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R @ v


class Kinematics:
    """FK/IK tren mot ban sao mjModel/mjData rieng (khong chia se voi vong lap mo phong)."""

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.d = mujoco.MjData(m)
        self.site_id = m.site(C.EE_SITE).id
        self.tcp_id = m.site(C.TCP_SITE).id
        self.grip_body = m.body("gripper_Gripper").id
        self.joint_ids = [m.joint(n).id for n in C.ARM_JOINTS]
        self.dofs = np.array([int(m.jnt_dofadr[j]) for j in self.joint_ids])
        self.qpos_adr = np.array([int(m.jnt_qposadr[j]) for j in self.joint_ids])
        self.n = len(self.dofs)
        self._home_frame()
        self._jvp = np.zeros((3, m.nv))
        self._jvr = np.zeros((3, m.nv))

    def _home_frame(self) -> None:
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        R = self.d.xmat[self.grip_body].reshape(3, 3)
        tip = self.d.site_xpos[self.site_id].copy()
        tcp = self.d.site_xpos[self.tcp_id].copy()
        self.tip_local = R.T @ tip
        self.tcp_local = R.T @ tcp
        self.appr_local = self.tip_local - self.tcp_local
        self.appr_local /= np.linalg.norm(self.appr_local)
        try:
            cl = self.d.site_xpos[self.m.site("gripper_gripper_left_site").id]
            cr = self.d.site_xpos[self.m.site("gripper_gripper_right_site").id]
            self.close_local = R.T @ (cl - cr)
        except (ValueError, KeyError):
            self.close_local = np.cross(self.appr_local, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(self.close_local) < 1e-6:
            self.close_local = np.cross(self.appr_local, np.array([1.0, 0.0, 0.0]))
        self.close_local /= np.linalg.norm(self.close_local)
        self.tip_from_tcp = float(np.linalg.norm(self.tip_local - self.tcp_local))

    def set_q(self, q: np.ndarray) -> None:
        self.d.qpos[self.qpos_adr] = q
        mujoco.mj_forward(self.m, self.d)

    def read_frame(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R = d.xmat[self.grip_body].reshape(3, 3)
        tip = d.site_xpos[self.site_id].copy()
        return tip, R @ self.appr_local, R @ self.close_local

    def frame(self, q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if q is not None:
            self.set_q(q)
        R = self.d.xmat[self.grip_body].reshape(3, 3)
        tip = self.d.site_xpos[self.site_id].copy()
        return tip, R @ self.appr_local, R @ self.close_local

    def frame_from(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        saved = self.d.qpos[self.qpos_adr].copy()
        self.set_q(q)
        out = (self.d.site_xpos[self.site_id].copy(),
               self.d.xmat[self.grip_body].reshape(3, 3) @ self.appr_local,
               self.d.xmat[self.grip_body].reshape(3, 3) @ self.close_local)
        self.d.qpos[self.qpos_adr] = saved
        mujoco.mj_forward(self.m, self.d)
        return out

    def tip_target_from_tcp(self, tcp: np.ndarray, approach: np.ndarray) -> np.ndarray:
        return tcp + self.tip_from_tcp * (approach / np.linalg.norm(approach))

    def ik(self, tip: np.ndarray, approach: np.ndarray, close: np.ndarray,
           q_init: np.ndarray, iters: int = C.IK_MAX_ITER,
           tol_p: float = C.IK_TOL_POS, tol_r: float = C.IK_TOL_ROT,
           damp: float = C.IK_DAMPING) -> tuple[np.ndarray, float, float]:
        q = np.array(q_init, dtype=float)
        a_des = approach / np.linalg.norm(approach)
        c_des = close - np.dot(close, a_des) * a_des
        c_des /= np.linalg.norm(c_des)
        best = (np.inf, np.inf, q.copy())
        for _ in range(iters):
            self.set_q(q)
            R = self.d.xmat[self.grip_body].reshape(3, 3)
            p = self.d.site_xpos[self.site_id].copy()
            a = R @ self.appr_local
            c = R @ self.close_local
            ep = tip - p
            ea = np.cross(a, a_des)
            ec = np.cross(c, c_des)
            err_p = float(np.linalg.norm(ep))
            err_r = float(np.linalg.norm(ea) + 0.5 * np.linalg.norm(ec))
            if err_p < best[0]:
                best = (err_p, err_r, q.copy())
            if err_p < tol_p and err_r < tol_r:
                return q, err_p, err_r
            mujoco.mj_jacSite(self.m, self.d, self._jvp, self._jvr, self.site_id)
            jp = self._jvp[:, self.dofs]
            jr = self._jvr[:, self.dofs]
            J = np.vstack([jp, jr, jr])
            e = np.concatenate([ep, ea, ec])
            W = np.concatenate([np.full(3, 1.0), np.full(3, 0.7), np.full(3, 0.35)])
            Jw = J * W[:, None]
            ew = e * W
            A = Jw @ Jw.T + damp ** 2 * np.eye(J.shape[0])
            dq = Jw.T @ np.linalg.solve(A, ew)
            q = q + np.clip(dq, -0.12, 0.12)
        return best[2], best[0], best[1]

    def ik_frame(self, tip: np.ndarray, approach: np.ndarray, close: np.ndarray,
                 q_init: np.ndarray | None = None, **kw) -> tuple[np.ndarray, float, float]:
        if q_init is None:
            q_init = self.d.qpos[self.qpos_adr].copy()
        return self.ik(tip, approach, close, q_init, **kw)
