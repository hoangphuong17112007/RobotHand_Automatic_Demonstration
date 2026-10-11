"""Thread RTC (Real Time Controller): chay Runner (mo phong + dieu khien + mission) va ban snapshot cho AI,
nhan proposal ve de dinh target moi chu ky.

Thiet ke an toan thread voi viewer: moi tac dong len MjData (mo phong + camera + sync)
xay ra TRONG MOT thread duy nhat (advance/tick). Viewer passive chi doc du lieu de ve,
khong bao gio copy-dong-thoi-voi-mj_step (loi `mj_copyDataVisual: stack is in use`).
"""
from __future__ import annotations

import threading
import time
from collections import deque

import mujoco
import numpy as np

from . import config as C
from .bridge import Bridge, Proposal, Snapshot
from .controller import PoseTarget
from .perception import Camera
from .graspai import build_patch
from .runner import Runner


class RtcNode(threading.Thread):
    """So huu mo hinh/diff; moi thao tac mujoco chi trong thread nay."""

    def __init__(self, bridge: Bridge, cycles: int = 5, task: int = 1,
                 settle_steps: int = 2500, snapshot_every: int = 0,
                 name: str = "rtc_node"):
        super().__init__(daemon=True, name=name)
        self.bridge = bridge
        self.cycles = cycles
        self.task = task
        self.settle_steps = settle_steps
        self.snapshot_every = snapshot_every      # moi N buoc ban 1 snapshot (0 = chi dau ky)
        self._stop = threading.Event()
        self.runner: Runner | None = None
        self.camera: Camera | None = None
        self.results: list[dict] = []
        self.ai_used: list[dict] = []
        self.failed: set[int] = set()
        self._surveyed = False
        self._survey_count = 0
        self._survey_max = int(6.5 / 0.001)
        self.summary: dict = {"results": [], "ai_used": [], "sim_time": 0.0,
                              "wall_time": 0.0}
        # Trang thai may chay (dung chung cho che do thread va che do tick/main):
        self._prepared = False
        self._cycle = 1
        self._cycling = False
        self._aim_state: str | None = None   # None | "waiting" | "decided"
        self._aim_ok = False
        self._prop: Proposal | None = None
        self._snap_ts = -1.0
        self._aim_win_start = 0.0
        self._ct = (0, 0.0, 0.0)   # (cycle0, t0, t_start)
        self._regrasps = 0         # so lan Deep Grasp tinh lai toa do kep (cuoi ky)
        self._poke_ct = 0          # so lan "gal" da chay trong chu ky hien tai
        # Lich su ngan han phuc vu bu tru do tre (Latency Compensation) va VFS
        self._tip_hist: deque = deque(maxlen=C.POSE_HIST_LEN)  # (ts, tip_x, tip_y)
        self._tgt_hist: deque = deque(maxlen=6)                # (ts, x, y) target

    def stop(self) -> None:
        self._stop.set()

    def prepared(self) -> bool:
        return self._prepared and self.runner is not None

    def prepare(self) -> None:
        """Khoi tao Runner + Camera va settle (goi 1 lan, tu thread goi advance)."""
        if self._prepared:
            return
        self.runner = Runner()
        self.camera = Camera(self.runner.model)
        print(f"[rtc] chuan bi, task={self.task}, cycles={self.cycles} ...", flush=True)
        self.runner.settle(self.settle_steps)
        self._prepared = True

    def finish(self) -> None:
        """Dong camera, thong bao AI dung, tong ket. An toan goi nhieu lan."""
        self.bridge.request_stop()
        if self.camera is not None:
            self.camera.close()
        self.summary = {
            "results": self.results,
            "ai_used": self.ai_used,
            "sim_time": self.runner.sim_t if self.runner else 0.0,
            "wall_time": time.perf_counter(),
        }

    # ------------------------------------------------------------------ tick

    def idle(self) -> bool:
        """True khi khong con chu ky nao de chay (dung goi advance them nua)."""
        return self._cycle > self.cycles

    def advance(self) -> str:
        """Mot buoc tien trinh: survey / chụp snapshot / cho aim / step 1 lic.
        Tra ve trang thai moi: "aim", "start", "step", "done", "done_all"."""
        if not self._prepared:
            self.prepare()
        r = self.runner
        if self._cycle > self.cycles:
            return "done_all"
        m, d = r.model, r.data

        # Survey tung buoc (khong block lock viewer lau): dua tay len bin hover.
        if not self._surveyed:
            self._survey_step(r)
            return "survey"

        if not self._cycling:
            return self._advance_cycle_setup(d)

        # Dang chay chu ky: step 1 lic.
        status = r.step_once()
        self._push_tip(r)
        cycle0, t0, t_start = self._ct
        if self._stop.is_set():
            raise SystemExit("interrupted")
        if C.VFS_ENABLE and r.mission.phase == r.mission.DESCEND \
                and (r.steps - cycle0) % C.VFS_EVERY == 0:
            self._maybe_vfs(r)
        if r.steps - cycle0 > 400_000:
            self.results.append({"cycle": self._cycle, "outcome": "timeout"})
            self._end_cycle()
            return "step"
        if self.snapshot_every and (r.steps - cycle0) % self.snapshot_every == 0:
            self._publish_snapshot(r.sim_t)
        if (r.steps - cycle0) % 4000 == 0:
            print(f"  [{r.mission.target_name}] t={r.sim_t:6.2f}s "
                  f"phase={r.mission.phase} ", flush=True)
        if status == "regrasp":
            return self._handle_regrasp()
        if status == "poked":
            return self._handle_poked()
        if status == "done":
            out = r._cycle_summary(self._cycle, t0)
            out["t_start"] = t_start
            out["sim_dt"] = d.time - t_start
            out["probe"] = r.mission.probe_idx.get(r.mission.target, 0)
            out["ai"] = bool(self._aim_ok)
            if out.get("outcome") == "loose":
                self.failed.add(r.mission.target)
            r.results.append(out)
            self.results.append(out)
            print(f"[rtc] cycle {self._cycle} {out.get('outcome')} {out.get('part')} "
                  f"sim={out.get('sim_dt', 0):.1f}s", flush=True)
            self._end_cycle()
            return "step"
        return "step"

    def _advance_cycle_setup(self, d) -> str:
        """Khoi tao dau chu ky: cap nhat blocked, snapshot, mo cua so cho aim."""
        r = self.runner
        r.mission.refiner = self._refine
        r.mission.blocked = set(self.failed)
        r.mission.external = None
        self.bridge.set_blocked(set(self.failed))

        if self._aim_state is None:
            snap = self._publish_snapshot(r.sim_t)
            self._snap_ts = snap.ts
            self._snap = snap
            self._aim_win_start = time.time()
            self._aim_state = "waiting"
            self._aim_ok = False
            self._prop = None
            return "aim"

        if self._aim_state == "waiting":
            prop = self.bridge.latest_proposal()
            if (prop is None or prop.ts < self._snap_ts
                    or (time.time() - prop.wall) > 8.0):
                prop = None
            if prop is not None and (prop.score > C.PROP_SCORE_MIN
                                     or prop.action == "poke"):
                # gal (poke) nham vao part ke goc/vach - diem gap thap la dieu
                # gan co, nen khong loc theo PROP_SCORE_MIN.
                self._aim_ok = True
                self._prop = prop
            elif time.time() - self._aim_win_start > 6.0:
                self._aim_ok = False
                self._prop = None
            else:
                # Van dang cho AI tinh — chua co gi de lam (viewer giu hinh).
                return "aim"
            self._aim_state = "decided"
            return "aim"

        # Da co ket luan aim: ap dung target va bat dau chu ky.
        if self._aim_ok and self._prop is not None:
            if self._prop.action == "poke":
                # Part ke goc/vach: gal truoc (AI de xuat vi tri + huong day).
                ok = r.mission.start_poke(d, r.sim_t, self._prop.poke_xy,
                                          self._prop.poke_dir,
                                          body=self._prop.body)
                if not ok:
                    self._aim_ok = False
                    self._prop = None
                    self._aim_state = None
                    return "step"
                self.ai_used.append({"cycle": self._cycle, **vars(self._prop)})
                print(f"[rtc] cycle {self._cycle} poke <- AI: {self._prop.name} "
                      f"dir=({self._prop.poke_dir[0]:+.2f},"
                      f"{self._prop.poke_dir[1]:+.2f})", flush=True)
                self._ct = (r.steps, time.perf_counter(), d.time)
                self._cycling = True
                self._aim_state = None
                return "start"
            # Bu tru do tre: ngoai suy vi tri linh kien tu thoi diem chup anh
            # sang thoi diem hien tai (Timestamping & Latency Compensation).
            px, py = self._predict_target(self._prop.ts, self._prop.x, self._prop.y)
            aim_ok = r.mission.aim_at(d, px, py,
                                      name=self._prop.name, yaw=self._prop.yaw)
            if aim_ok:
                self._track_target(r.sim_t, px, py)
                self.ai_used.append({"cycle": self._cycle, **vars(self._prop)})
                print(f"[rtc] cycle {self._cycle} aim <- AI: {self._prop.name} at "
                      f"({px:.3f},{py:.3f}) yaw={self._prop.yaw:.2f} "
                      f"clr={self._prop.clearance:.3f} ready={self._prop.grasp_ready} "
                      f"score={self._prop.score:.2f}", flush=True)
                # doi gat: cac linh kien lang gieng dang can tro lan gap nay
                if C.BLOCK_OBSTACLES and self._prop.blocked:
                    r.mission.blocked = set(self.failed) | set(self._prop.blocked)
                    self.bridge.set_blocked(r.mission.blocked)
            else:
                self._aim_ok = False
        if not r.mission.start_cycle(d, r.sim_t):
            self.results.append({"cycle": self._cycle, "outcome": "no_target"})
            print(f"[rtc] cycle {self._cycle}: no target", flush=True)
            self._end_cycle()
            return "start"
        self._ct = (r.steps, time.perf_counter(), d.time)
        self._cycling = True
        self._aim_state = None
        return "start"

    def _end_cycle(self) -> None:
        self._cycling = False
        self._aim_state = None
        self._aim_ok = False
        self._prop = None
        self._cycle += 1
        self._regrasps = 0
        self._poke_ct = 0
        self.bridge.set_poke_budget(C.POKE_MAX)

    def _handle_regrasp(self) -> str:
        """Co nguy co truot (part nghieng/nan kep) -> dung tay & goi Deep Grasp
        tinh lai toa do kep: tang probe (doi vi tri kep), chup snapshot moi de
        AI tinh lai. Tran lai qua nhieu lan cho cung 1 chu ky -> that bai.
        """
        r = self.runner
        target = r.mission.target
        self._regrasps += 1
        print(f"[rtc] cycle {self._cycle} slip-risk -> Deep Grasp tinh lai "
              f"toa do kep #{self._regrasps}", flush=True)
        if self._regrasps > C.GRASP_REGASPS_MAX:
            self.failed.add(target) if target is not None else None
            out = r._cycle_summary(self._cycle, self._ct[1])
            out["t_start"] = self._ct[2]
            out["sim_dt"] = r.data.time - self._ct[2]
            out["probe"] = r.mission.probe_idx.get(target, 0) if target is not None else 0
            out["ai"] = bool(self._aim_ok)
            out["outcome"] = "loose"
            r.results.append(out)
            self.results.append(out)
            print(f"[rtc] cycle {self._cycle} loose (het lan thu lai) "
                  f"{out.get('part')} sim={out['sim_dt']:.1f}s", flush=True)
            self._end_cycle()
            return "step"
        if target is not None:
            # doi vi tri kep cua Deep Grasp cho lan thu lai (probe tiep theo)
            r.mission.probe_idx[target] = r.mission.probe_idx.get(target, 0) + 1
        r.mission.regrasps = self._regrasps
        self._cycling = False
        self._aim_state = None
        self._aim_ok = False
        self._prop = None
        self._snap_ts = -1.0
        return "step"

    def _handle_poked(self) -> str:
        """Da gal xong 1 lan (part dich ra khoi goc/vach): giam poke budget, chup
        snapshot moi de AI tinh tiep (co the gal them hoac gap). Chua tinh 1 cycle."""
        self._poke_ct += 1
        budget = max(0, C.POKE_MAX - self._poke_ct)
        self.bridge.set_poke_budget(budget)
        print(f"[rtc] cycle {self._cycle} poked #{self._poke_ct} "
              f"(budget con {budget})", flush=True)
        self._cycling = False
        self._aim_state = None
        self._aim_ok = False
        self._prop = None
        self._snap_ts = -1.0
        return "step"

    # ------------------------------------------------------- ham ho tro khac

    def _survey_step(self, r) -> None:
        """Mot buoc survey: dua tay toi vi tri hover phia tren bin."""
        m, d = r.model, r.data
        goal = np.array([0.0, C.SOURCE_BIN[1], C.TIP_Z_HOVER])
        tip, _, _ = r.kin.read_frame(d)
        if self._survey_count >= self._survey_max or tip[2] < 0.30:
            self._surveyed = True
            return
        target = PoseTarget(goal, np.array(C.APPROACH_DOWN), np.array(C.CLOSE_X))
        tau = r.ctrl.compute(d, target)
        d.ctrl[r.arm_ids] = tau
        r.grip.command(d, C.GRIP_OPEN)
        mujoco.mj_step(m, d)
        r.steps += 1
        r.sim_t = float(d.time)
        self._survey_count += 1
        if (r.steps % 10 == 0):
            r._sample()
        if float(np.linalg.norm(tip - goal)) < 0.004:
            self._surveyed = True

    def _push_tip(self, r) -> None:
        """Ghi ham lich su vi tri canh tay/camera (Lam lich su ngan han)."""
        tip, _, _ = r.kin.read_frame(r.data)
        self._tip_hist.append((r.sim_t, float(tip[0]), float(tip[1])))

    def _track_target(self, ts: float, x: float, y: float) -> None:
        """Quan sat vi tri linh kien theo thoi gian (cho bu tru do tre)."""
        self._tgt_hist.append((float(ts), float(x), float(y)))

    def _predict_target(self, ts: float, x: float, y: float):
        """Bu tru do tre: ngoai suy vi tri linh kien tu (ts,x,y) sang hien tai.

        Giu mot mang lich su vi tri ngan han; van toc uoc tu 2 quan sat moii
        nhat; ngoai suy tuyen tinh nhung dong trong C.LATCOMP_MAX_AGE.
        """
        now = self.runner.sim_t if self.runner is not None else ts
        age = now - ts
        if age <= 1e-4 or len(self._tgt_hist) < 2:
            return x, y
        (t1, x1, y1), (t2, x2, y2) = self._tgt_hist[-2], self._tgt_hist[-1]
        if t2 - t1 < 1e-5:
            return x, y
        vx, vy = (x2 - x1) / (t2 - t1), (y2 - y1) / (t2 - t1)
        dt = min(age, C.LATCOMP_MAX_AGE)
        return x + vx * dt, y + vy * dt

    def _maybe_vfs(self, r) -> None:
        """Visual-Force Servoing: refresh vi tri linh kien khi ha xuong (micro-adjust)."""
        trg = r.mission.target
        if trg is None:
            return
        try:
            snap = self._publish_snapshot(r.sim_t)
        except Exception:
            return
        ex = ey = None
        for pc in snap.candidates:
            if pc.body == trg:
                ex, ey = self._predict_target(snap.ts, pc.x, pc.y)
                break
        if ex is None:
            return
        self._track_target(r.sim_t, ex, ey)
        grp = getattr(r.mission, "grasp", None)
        if grp is None:
            return
        off = np.zeros(2)
        off[0] = ex - float(grp[0])
        off[1] = ey - float(grp[1])
        n = float(np.hypot(off[0], off[1]))
        if n < 0.0015:
            r.mission.vfs_off = np.zeros(2)
            return
        if n > C.VFS_MAX_OFF:
            off *= C.VFS_MAX_OFF / n
        cur = getattr(r.mission, "vfs_off", None)
        cur = np.zeros(2) if cur is None else np.asarray(cur, dtype=float)
        r.mission.vfs_off = C.VFS_EMA * off + (1.0 - C.VFS_EMA) * cur

    def _publish_snapshot(self, snap_ts: float) -> Snapshot:
        cam = self.camera
        d = self.runner.data
        cands = cam.candidates(d)
        depth = seg = None
        patches = []
        clouds = []
        if cands:
            rgb, depth, seg = cam.snapshot(d)
            geom_of = cam.part_of_body
            for pc in cands:
                g = geom_of.get(pc.body)
                if g is not None:
                    patches.append(build_patch(depth, seg, g, pc.pcy, pc.pcx))
                    clouds.append(cam.part_cloud(d, seg, depth, pc.body))
        snap = Snapshot(ts=snap_ts, wall=time.time(), candidates=cands,
                        patches=patches, clouds=clouds, depth=depth, seg=seg)
        self.bridge.publish_snapshot(snap)
        return snap

    def _refine(self, d, body: int):
        """Camera chup lai tai hover -> (x,y, yaw) cua part đa chon (neu thay) de DESCEND chinh xac."""
        try:
            cands = self.camera.candidates(d)
        except Exception as e:
            print(f"[rtc] refine fail: {e}", flush=True)
            return None
        goal = self.runner.mission.hover[:2]
        for pc in cands:
            if pc.body == int(body):
                px, py = self._predict_target(d.time, pc.x, pc.y)
                self._track_target(d.time, px, py)
                return (px, py)
        best, bd, best_yaw = None, None, 0.0
        for pc in cands:
            dst = float(np.hypot(pc.x - goal[0], pc.y - goal[1]))
            if bd is None or dst < bd:
                bd, best = dst, (pc.x, pc.y)
                best_yaw = pc.yaw
        return best if best is None else (best[0], best[1])

    # ----------------------------------------------------------------- thread

    def run(self) -> None:
        """Che do thread: chay tat ca chu ky roi dung."""
        try:
            self.prepare()
            while (self._cycle <= self.cycles or self._cycling) \
                    and not self._stop.is_set() and not self.bridge.stopped:
                burst = 0
                try:
                    while burst < 40 and not self._stop.is_set():
                        status = self.advance()
                        burst += 1
                        if status == "aim":
                            # Dang cho AI tinh: nghi mot lat de AI kip xu ly.
                            break
                except SystemExit:
                    break
                time.sleep(0.002)
        finally:
            self.finish()

    def summary_report(self) -> dict:
        return self.summary