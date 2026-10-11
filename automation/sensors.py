"""Doc cam bien (Luc/Torque 6 truc, cham cam, vi tri/khop)."""
from __future__ import annotations

import mujoco
import numpy as np

from . import config as C


class SensorBus:
    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.ft_f = self._addr("ft_force")
        self.ft_t = self._addr("ft_torque")
        self.touch_adr = self._addr("gripper_touch")
        self.gear_qadr = int(m.jnt_qposadr[m.joint(C.GRIPPER_JOINT).id])
        self.gear_dadr = int(m.jnt_dofadr[m.joint(C.GRIPPER_JOINT).id])
        self.q = np.array([int(m.jnt_qposadr[m.joint(n).id]) for n in C.ARM_JOINTS])
        self.qd = np.array([int(m.jnt_dofadr[m.joint(n).id]) for n in C.ARM_JOINTS])
        self.f_base = np.zeros(3)
        self.t_base = np.zeros(3)

    def _addr(self, name: str) -> int:
        return int(self.m.sensor_adr[self.m.sensor(name).id])

    def calibrate(self, d: mujoco.MjData) -> None:
        self.f_base = d.sensordata[self.ft_f:self.ft_f + 3].copy()
        self.t_base = d.sensordata[self.ft_t:self.ft_t + 3].copy()

    def ft(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return (d.sensordata[self.ft_f:self.ft_f + 3].copy(),
                d.sensordata[self.ft_t:self.ft_t + 3].copy())

    def ft_ext(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        f, t = self.ft(d)
        return f - self.f_base, t - self.t_base

    def f_ext_norm(self, d: mujoco.MjData) -> float:
        f, _ = self.ft_ext(d)
        return float(np.linalg.norm(f))

    def contact_load(self, d: mujoco.MjData, approach: np.ndarray | None = None) -> float:
        """Do lon bien thien luc F/T so voi baseline (bat loan trong khi di chuyen)."""
        f, t = self.ft_ext(d)
        return float(np.linalg.norm(f) + np.linalg.norm(t) * 0.1)

    def touch(self, d: mujoco.MjData) -> float:
        return float(d.sensordata[self.touch_adr])

    def touch_raw(self, d: mujoco.MjData) -> float:
        return self.touch(d)

    def opening(self, d: mujoco.MjData) -> float:
        """Do mo ngam kep (mm) do tu 2 diem truc pivot 2 ngon."""
        m = self.m
        l = m.site(C.GRIP_FINGER0).id
        r = m.site(C.GRIP_FINGER1).id
        return float(1000.0 * np.linalg.norm(d.site_xpos[r] - d.site_xpos[l]))

    def arm_q(self, d: mujoco.MjData) -> np.ndarray:
        return d.qpos[self.q].copy()

    def arm_qd(self, d: mujoco.MjData) -> np.ndarray:
        return d.qvel[self.qd].copy()

    def gear_q(self, d: mujoco.MjData) -> float:
        return float(d.qpos[self.gear_qadr])

    def gear_qd(self, d: mujoco.MjData) -> float:
        return float(d.qvel[self.gear_dadr])

    def warning_count(self, d: mujoco.MjData) -> int:
        return int(sum(w.number for w in d.warning))
