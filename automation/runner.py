"""Vong chay chinh: mo phong + dieu khien + may trang thai + telemetry."""
from __future__ import annotations

import csv
import datetime
import time
from pathlib import Path

import mujoco
import numpy as np

from . import config as C
from .controller import Gripper, ImpedanceController
from .kinematics import Kinematics
from .mission import PickMission
from .sensors import SensorBus


class Runner:
    """Dua model mo phong chay tu dong qua cac chu ky gắp linh kien."""

    def __init__(self, model_path=None):
        self.model = mujoco.MjModel.from_xml_path(str(model_path or C.MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.kin = Kinematics(self.model)
        self.ctrl = ImpedanceController(self.model, self.kin)
        self.grip = Gripper(self.model)
        self.bus = SensorBus(self.model)
        self.mission = PickMission(self.model, self.kin, self.bus)
        self.mission.grip = self.grip
        self.arm_ids = np.array([self.model.actuator(n).id
                                 for n in C.ARM_ACTUATORS])
        self.steps = 0
        self.sim_t = 0.0
        self.sample = []
        self.results: list[dict] = []
        self.phase = "idle"

    def settle(self, steps=2500) -> None:
        mujoco.mj_forward(self.model, self.data)
        m, d = self.model, self.data
        for _ in range(steps):
            d.ctrl[self.arm_ids] = d.qfrc_bias[self.kin.dofs]
            self.grip.command(d, C.GRIP_OPEN)
            mujoco.mj_step(m, d)
        self.bus.calibrate(d)
        self.steps += steps
        self.sim_t = float(d.time)
        self.phase = "settled"

    def grip_mode(self, d, mode: str) -> None:
        if mode == "close":
            self.grip.command(d, C.GRIP_CLOSE_Q, tmax=C.GRIP_CLOSE_TMAX)
        elif mode == "softclose":
            self.grip.command(d, C.GRIP_CLOSE_Q, tmax=C.GRIP_SOFT_TMAX)
        elif mode == "squeeze":
            d.ctrl[self.grip.act_id] = C.GRIP_SQUEEZE_TORQUE
        elif mode == "hold":
            # momen kep HANG (neu PD ve q hien tai -> tau ~ 0 -> rot linh kien)
            self.grip.hold(d, C.GRIP_HOLD_TORQUE)
        elif mode == "preclose":
            self.grip.gap_command(d, self.mission.gap_des)
        elif mode == "none":
            # mo cham tai cho (xem GRIP_RELEASE_TORQUE)
            d.ctrl[self.grip.act_id] = C.GRIP_RELEASE_TORQUE
        else:
            self.grip.command(d, C.GRIP_OPEN, tmax=C.GRIP_TORQUE_MAX)

    def step_once(self) -> str:
        m, d = self.model, self.data
        now = self.sim_t
        status = self.mission.fsm(d, now)
        target, mode = self.mission.update(d, now)
        tau = self.ctrl.compute(d, target)
        d.ctrl[self.arm_ids] = tau
        self.grip_mode(d, mode)
        mujoco.mj_step(m, d)
        self.steps += 1
        self.sim_t = float(d.time)
        if self.steps % 10 == 0:
            self._sample()
        self.phase = self.mission.target_name or "idle"
        return status

    def _sample(self) -> None:
        d = self.data
        tip, a, c = self.kin.read_frame(d)
        f, t = self.bus.ft(d)
        self.sample.append({
            "t": self.sim_t,
            "phase": self.mission.phase,
            "target": self.mission.target_name,
            "tip_x": tip[0], "tip_y": tip[1], "tip_z": tip[2],
            "f_x": f[0], "f_y": f[1], "f_z": f[2],
            "touch": self.bus.touch(d),
            "opening": self.bus.opening(d),
            "gap": self.grip.gap(d) if self.grip is not None else -1.0,
            "warning": self.bus.warning_count(d),
        })

    def run_cycles(self, n=5, reset_first=True, settle_steps=2500,
                   max_steps=400_000) -> dict:
        m, d = self.model, self.data
        if reset_first:
            mujoco.mj_resetData(m, d)
            self.settle(settle_steps)
        results = []
        for k in range(n):
            cycle0 = self.steps
            t_start = d.time
            t0 = time.perf_counter()
            if not self.mission.start_cycle(d, self.sim_t):
                results.append({"cycle": k + 1, "outcome": "no_target"})
                break
            while True:
                status = self.step_once()
                if self.steps - cycle0 > max_steps:
                    results.append({"cycle": k + 1, "outcome": "timeout"})
                    break
                if (self.steps - cycle0) % 4000 == 0:
                    print(f"  [{self.mission.target_name}] t={self.sim_t:6.2f}s "
                          f"phase={self.mission.phase} ", flush=True)
                if status == "done":
                    r = self._cycle_summary(k + 1, t0)
                    r["t_start"] = t_start
                    r["sim_dt"] = d.time - t_start
                    r["probe"] = self.mission.probe_idx.get(self.mission.target, 0)
                    results.append(r)
                    break
        self.results = results
        return {"results": results, "sample": self.sample,
                "sim_time": self.sim_t, "wall_time": time.perf_counter()}

    def _cycle_summary(self, cycle, t0) -> dict:
        d = self.data
        p = d.xpos[self.mission.target]
        outcome = self.mission.cycle_outcome
        mark = getattr(self.mission, "drop_mark", None)
        if mark is None:
            mark = C.TARGET_MARK
        dxy = float(np.hypot(p[0] - mark[0], p[1] - mark[1]))
        if outcome == "ok":
            delivered = dxy <= C.PLACE_TOL and p[2] < 0.14
            outcome = "ok" if delivered else "off_mark"
        f, _ = self.bus.ft(d)
        return {
            "cycle": cycle,
            "part": self.mission.target_name,
            "outcome": outcome,
            "t_end": d.time,
            "part_x": float(p[0]), "part_y": float(p[1]), "part_z": float(p[2]),
            "drop_x": float(mark[0]), "drop_y": float(mark[1]),
            "d_mark": dxy,
            "f_z": float(f[2]),
            "wall": time.perf_counter() - t0,
        }

    def save_csv(self, path: str | Path, tag: str = "run") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["tag", "run", "t", "phase", "target", "tip_x", "tip_y", "tip_z",
                  "f_x", "f_y", "f_z", "touch", "opening", "gap", "warning"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for s in self.sample:
                row = {k: s.get(k) for k in ("t", "phase", "target", "tip_x",
                                             "tip_y", "tip_z", "f_x", "f_y",
                                             "f_z", "touch", "opening", "gap",
                                             "warning")}
                row = {"tag": tag, "run": 0, **row}
                w.writerow(row)
        return path

    def save_results(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["cycle", "part", "outcome", "t_start", "t_end", "sim_dt", "wall",
                  "part_x", "part_y", "part_z", "drop_x", "drop_y", "d_mark", "f_z",
                  "probe"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in self.results:
                row = {k: r.get(k, "") for k in fields}
                w.writerow(row)
        return path