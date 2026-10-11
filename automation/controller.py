"""Dieu khien van cong trog (Impedance) va gap do."""
from __future__ import annotations

import mujoco
import numpy as np

from . import config as C
from .kinematics import Kinematics


class PoseTarget:
    __slots__ = ("tip", "approach", "close", "v", "w")

    def __init__(self, tip, approach, close, v=None, w=None):
        self.tip = np.asarray(tip, dtype=float)
        self.approach = np.asarray(approach, dtype=float)
        self.approach = self.approach / np.linalg.norm(self.approach)
        self.close = np.asarray(close, dtype=float)
        self.close = self.close - np.dot(self.close, self.approach) * self.approach
        self.close = self.close / np.linalg.norm(self.close)
        self.v = np.zeros(3) if v is None else np.asarray(v, dtype=float)
        self.w = np.zeros(3) if w is None else np.asarray(w, dtype=float)


class ImpedanceController:
    """tau = qfrc_bias + J^T (Kp e + Kd e_dot) tren vi tri va huong cua ee_site."""

    def __init__(self, m: mujoco.MjModel, kin: Kinematics):
        self.m = m
        self.kin = kin
        self.site_id = m.site(C.EE_SITE).id
        self.dofs = kin.dofs
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))
        self._vel = np.zeros(6)
        self.Kp = C.KP_POS
        self.Kd = C.KD_POS
        self.Kr = C.KP_ROT
        self.Kdr = C.KD_ROT
        self.tau = np.zeros(len(self.dofs))

    def compute(self, d: mujoco.MjData, target: PoseTarget) -> np.ndarray:
        m, kin = self.m, self.kin
        mujoco.mj_jacSite(m, d, self._jp, self._jr, self.site_id)
        R = d.xmat[kin.grip_body].reshape(3, 3)
        p = d.site_xpos[self.site_id].copy()
        a = R @ kin.appr_local
        c = R @ kin.close_local
        ep = target.tip - p
        ea = np.cross(a, target.approach)
        ec = np.cross(c, target.close)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.site_id, self._vel, 0)
        v = self._vel[3:6]
        w = self._vel[0:3]
        F = self.Kp * ep + self.Kd * (target.v - v)
        T = self.Kr * (ea + 0.5 * ec) + self.Kdr * (target.w - w)
        jp = self._jp[:, self.dofs]
        jr = self._jr[:, self.dofs]
        tau = d.qfrc_bias[self.dofs] + jp.T @ F + jr.T @ T
        self.tau = tau
        return tau


class Gripper:
    """Dieu khien dong gap bang momen tor (PD tren goc servogear)."""

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.act_id = m.actuator(C.GRIPPER_ACTUATOR).id
        self.qadr = int(m.jnt_qposadr[m.joint("gripper_servo_gear").id])
        self.dadr = int(m.jnt_dofadr[m.joint("gripper_servo_gear").id])
        self.left = m.site("gripper_gripper_left_site").id
        self.right = m.site("gripper_gripper_right_site").id
        self.opening = 0.0
        self.pad0, self.pad1 = self._find_pads(m)

    @staticmethod
    def _find_pads(m: mujoco.MjModel) -> tuple[int, int]:
        """Hai mieng kep: geom va cham, manh (day 0.5mm) va cao (~50mm)."""
        root = int(m.body_rootid[m.body("gripper_Gripper").id])
        pads = [g for g in range(m.ngeom)
                if int(m.body_rootid[int(m.geom_bodyid[g])]) == root
                and int(m.geom_contype[g]) != 0
                and m.geom_size[g][0] < 0.001 and m.geom_size[g][2] > 0.02]
        if len(pads) == 2:
            return int(pads[0]), int(pads[1])
        return C.GAP_GEOM0, C.GAP_GEOM1

    def gap(self, d: mujoco.MjData) -> float:
        """Khoang cach thuc giua hai mieng kep (m)."""
        return float(mujoco.mj_geomDistance(self.m, d, self.pad0, self.pad1,
                                            1.0, None))

    def gap_command(self, d: mujoco.MjData, gap_des: float,
                    kp: float = C.GAP_PRE_KP, tmax: float = C.GAP_PRE_TMAX) -> float:
        tau = kp * (self.gap(d) - gap_des)
        tau = float(np.clip(tau, -tmax, tmax))
        d.ctrl[self.act_id] = tau
        return tau

    def update_opening(self, d: mujoco.MjData) -> float:
        self.opening = float(np.linalg.norm(
            d.site_xpos[self.left] - d.site_xpos[self.right]))
        return self.opening

    def command(self, d: mujoco.MjData, q_des: float, kp: float = C.GRIP_KP,
                kd: float = C.GRIP_KD, tmax: float = C.GRIP_TORQUE_MAX) -> float:
        tau = kp * (q_des - d.qpos[self.qadr]) - kd * d.qvel[self.dadr]
        tau = float(np.clip(tau, -tmax, tmax))
        d.ctrl[self.act_id] = tau
        return tau

    def hold(self, d: mujoco.MjData, torque: float = C.GRIP_HOLD_TORQUE) -> None:
        d.ctrl[self.act_id] = abs(torque)

    def release(self, d: mujoco.MjData) -> None:
        d.ctrl[self.act_id] = 0.0
