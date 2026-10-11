"""Trajectory (quintic + slerp giua cac waypoint) va may trang thai pick-and-place."""
from __future__ import annotations

import mujoco
import numpy as np

from . import config as C
from .controller import PoseTarget
from .kinematics import Kinematics, frame_quat, quat_apply, quat_slerp
from .sensors import SensorBus

DOWN = np.array(C.APPROACH_DOWN)
CLOSE_X = np.array(C.CLOSE_X)


class MoveSegment:
    """Doan thang quintic giua 2 pose (vi tri moc + huong dung slerp)."""

    def __init__(self, p0, a0, c0, p1, a1, c1, vmax, amax=0.6):
        self.p0 = np.asarray(p0, float).copy()
        self.p1 = np.asarray(p1, float).copy()
        self.a0 = np.asarray(a0, float) / np.linalg.norm(a0)
        self.a1 = np.asarray(a1, float) / np.linalg.norm(a1)
        self.c0 = np.asarray(c0, float)
        self.c1 = np.asarray(c1, float)
        self.q0 = frame_quat(self.a0, self.c0)
        self.q1 = frame_quat(self.a1, self.c1)
        dist = float(np.linalg.norm(self.p1 - self.p0))
        t_ramp = vmax / amax
        t_const = max(0.0, dist / vmax - t_ramp)
        self.T = max(2 * t_ramp + t_const, 0.004)

    def dist(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    def pose_at(self, t: float) -> PoseTarget:
        tau = float(np.clip(t / self.T, 0.0, 1.0))
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / self.T
        p = self.p0 + s * (self.p1 - self.p0)
        q = quat_slerp(self.q0, self.q1, s)
        approach = quat_apply(q, np.array([1.0, 0.0, 0.0]))
        close = quat_apply(q, np.array([0.0, 1.0, 0.0]))
        v = ds * (self.p1 - self.p0)
        return PoseTarget(p, approach, close, v=v)


class PickMission:
    """May trang thai: hover -> descend (F/T nương) -> grip -> lift -> transfer -> drop."""

    HOVER, DESCEND, GRIP, LIFT, TRANSFER, DROP, RELEASE, RETURN, POKE = range(9)

    def __init__(self, m: mujoco.MjModel, kin: Kinematics, bus: SensorBus):
        self.m = m
        self.kin = kin
        self.bus = bus
        self.part_ids = [i for i in range(m.nbody)
                         if m.body(i).name.startswith(C.PART_BODIES_PREFIX)]
        self.phase = self.HOVER
        self.seg: MoveSegment | None = None
        self.t0 = 0.0
        self.t_phase = 0.0
        self.target = None
        self.target_name = ""
        self.attempts = 0
        self.picked = set()
        self.probe_idx: dict[int, int] = {}
        self.results: list[dict] = []
        self.cycle_outcome = 'pending'
        self.grip_open0 = self.grip_open_min = 100.0
        self.grip_touch_max = 0.0
        self.lift_from = 0.0
        self.gap_des = 0.06
        self.squeezing = False
        self.squeeze_z = 0.0
        self.held_off = np.zeros(2)
        self.external = None
        self.aim_used = None
        self.refiner = None
        self.blocked: set[int] = set()
        self.yaw = 0.0
        self.close_axis = np.array(CLOSE_X, float)
        self.vfs_off = np.zeros(2)
        # vi tri tha tren khay dich - chon lai moi chu ky (xem _pick_drop_xy)
        self.drop_mark = np.array(C.TARGET_MARK, float)
        # part dang duoc kep/tha - loai khoi vat can o khay dich de marker
        # (va vi tri tha ke tiep) khong nhay som khi part chua duoc tha xong.
        self.carried = None
        # danh dau cho buoc "mo hoan toan cang kep" dau moi chu ky
        self._preopen_done = True
        self._preopen_t0 = -1.0
        self._load_high_t = -1.0
        self._hover_t0 = -1.0
        self._drop_t = -1.0
        self._tr_t = -1.0
        self.red_ids = self._red_parts()
        self.grip = None
        # "gal" (poke) + goi lai Deep Grasp (regrasp) khi co nguy co truot
        self.retry_request = False
        self.regrasps = 0
        self._poke_pts: list[tuple[np.ndarray, float]] = []
        self._poll = 0            # index seg trong chuoi gal
        self.poke_z = 0.0
        self._poke_t0 = -1.0

    def aim_at(self, d: mujoco.MjData, x: float, y: float, name: str = "",
               yaw: float = 0.0) -> bool:
        """Gan target ngoai lai (cua he thi giac/AI) theo toa do (x,y) trong he world."""
        loaded = False
        best: tuple[float, int] | None = None
        bin_ok = lambda px, py: abs(px) <= 0.165 and 0.295 <= py <= 0.435
        for b in self._available_parts(d):
            if b in self.blocked:
                continue
            p = d.xpos[b]
            dst = float(np.hypot(p[0] - x, p[1] - y))
            if dst > 0.07:
                continue
            if best is None or dst < best[0]:
                best = (dst, b)
                loaded = True
        if best is None:
            return False
        self.external = {"x": x, "y": y, "body": best[1],
                         "name": name or self.m.body(best[1]).name,
                         "yaw": float(yaw)}
        self.aim_used = self.external
        return True

    def _red_parts(self) -> set[int]:
        """Cac part co material TARGET_MATERIAL (linh kien do) - rong; rong = tat ca."""
        want = C.TARGET_MATERIAL
        out: set[int] = set()
        if not want:
            return out
        for g in range(self.m.ngeom):
            b = int(self.m.geom_bodyid[g])
            mat = int(self.m.geom_matid[g])
            if b in self.part_ids and mat >= 0 and self.m.mat(mat).name == want:
                out.add(b)
        return out

    def _available_parts(self, d: mujoco.MjData) -> list[int]:
        out = []
        for b in self.part_ids:
            p = d.xpos[b]
            if self.red_ids and b not in self.red_ids:
                continue
            if (p[1] > 0.29 and p[2] < 0.25 and b not in self.picked
                    and self.probe_idx.get(b, 0) < C.GRASP_MAX_PROBES):
                out.append(b)
        return out

    def select_target(self, d: mujoco.MjData) -> int | None:
        cand = []
        for b in self._available_parts(d):
            if b in self.blocked:
                continue
            p = d.xpos[b]
            if abs(p[0]) <= 0.145 and 0.305 <= p[1] <= 0.425:
                cand.append((float(p[2]), b))
        if not cand:
            return None
        cand.sort(reverse=True)
        return cand[0][1]

    def _tray_parts(self, d: mujoco.MjData) -> list[int]:
        """Cac linh kien dang nam o khu vuc khay dich (da tha truoc do)."""
        out = []
        for b in self.part_ids:
            if b == self.carried:
                continue
            p = d.xpos[b]
            if p[1] < C.TRAY_OCCUPIED_Y and p[2] < 0.20:
                out.append(b)
        return out

    def _drop_free(self, d: mujoco.MjData, xy, occupied,
                   clearance: float) -> bool:
        """True neu (xy) nam trong khay va du khoang cach voi moi part da tha."""
        if abs(xy[0]) > C.TRAY_X_MAX:
            return False
        if not (C.TRAY_Y_MIN <= xy[1] <= C.TRAY_Y_MAX):
            return False
        for b in occupied:
            p = d.xpos[b]
            if float(np.hypot(p[0] - xy[0], p[1] - xy[1])) < clearance:
                return False
        return True

    def _pick_drop_xy(self, d: mujoco.MjData) -> np.ndarray:
        """Chon vi tri tha moi chu ky tren khay dich - dam bao thong thao.

        Vi tri cu se bi linh kien vua tha chiem -> chu ky tiep theo tha cham
        len no (kep/linh kien ram nhau, that bai). Duyet ung vien theo vong
        tang dan quanh TARGET_MARK, chon vi tri dau tien du dieu kien:
        nam trong khay dich va cach it nhat DROP_CLEARANCE moi linh kien
        da nam o khay. Neu het cho thong thao thi giam dan nguong (van
        uu tien vi tri gan nhan), cuoi cung chon vi tri co khoang cach lon nhat.
        """
        occupied = self._tray_parts(d)
        cands = [np.array(C.TARGET_MARK, float)]
        for r in C.DROP_RINGS:
            for i in range(C.DROP_RING_SAMPLES):
                a = 2.0 * np.pi * i / C.DROP_RING_SAMPLES
                cands.append(np.array([C.TARGET_MARK[0] + r * np.cos(a),
                                       C.TARGET_MARK[1] + r * np.sin(a)]))
        clearance = C.DROP_CLEARANCE
        while clearance > 0.02:
            for xy in cands:
                if self._drop_free(d, xy, occupied, clearance):
                    return xy
            clearance *= 0.85
        # het cho: chon ung vien co khoang cach toi da lon nhat
        best, bd = cands[0], -1.0
        for xy in cands:
            if abs(xy[0]) > C.TRAY_X_MAX:
                continue
            if not (C.TRAY_Y_MIN <= xy[1] <= C.TRAY_Y_MAX):
                continue
            dmin = min((float(np.hypot(d.xpos[b][0] - xy[0],
                                       d.xpos[b][1] - xy[1]))
                        for b in occupied), default=9.9)
            if dmin > bd:
                bd, best = dmin, xy
        return np.array(best, float)

    def _jaw_offset(self, d: mujoco.MjData) -> np.ndarray:
        """Lech world giua tam hai mieng kep va tip (ee_site).

        ee_site KHONG nam giua hai mieng kep (lech ~11mm theo truc kep,
        ~8mm theo huong doc) -> neu dat tip len tam linh kien thi mieng
        kep lech qua mot ben, cham vao goc linh kien (lam no lat) va kep
        day linh kien ra khi dong.  Bo sung offset nay vao target.
        """
        tip = d.site_xpos[self.kin.site_id].copy()
        c = 0.5 * (d.geom_xpos[C.GAP_GEOM0] + d.geom_xpos[C.GAP_GEOM1])
        return c - tip

    def _part_geom(self, b: int) -> int | None:
        g = None
        for gi in range(self.m.ngeom):
            if int(self.m.geom_bodyid[gi]) == b:
                if g is None or int(self.m.geom_contype[gi]) != 0:
                    g = gi
                if int(self.m.geom_contype[gi]) != 0:
                    return gi
        return g

    def _part_yaw(self, d: mujoco.MjData, b: int) -> float:
        """Goc quay (rad, quanh truc dung) cua truc dai cua box linh kien."""
        g = self._part_geom(b)
        if g is None:
            return 0.0
        R = d.geom_xmat[g].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _pads_dbg(self, d: mujoco.MjData, tip: np.ndarray) -> str:
        """Vi tri 2 mieng kep (geom 75/90) so voi tip va linh kien muc tieu."""
        out = []
        for g in (C.GAP_GEOM0, C.GAP_GEOM1):
            p = d.geom_xpos[g]
            hz = float(self.m.geom_size[g][2])
            out.append(f"g{g}[x{p[0]-tip[0]:+.3f} y{p[1]-tip[1]:+.3f} "
                       f"z{p[2]-hz-tip[2]:+.3f}..{p[2]+hz-tip[2]:+.3f}]")
        if self.target is not None:
            hz = self._part_hz(d, self.target)
            pz = float(d.xpos[self.target][2])
            out.append(f"part[{pz-hz-tip[2]:+.3f},{pz+hz-tip[2]:+.3f}]")
        out.append(self._finger_slide_dbg(d))
        return " ".join(out)

    def _finger_slide_dbg(self, d: mujoco.MjData) -> str:
        """qpos slide cua 2 co ngon kep (x,y,z) + goc servo/idol gear."""
        rows = []
        for g in (C.GAP_GEOM0, C.GAP_GEOM1):
            b = int(self.m.geom_bodyid[g])
            qs = []
            for j in range(self.m.njnt):
                if int(self.m.jnt_bodyid[j]) == b and int(self.m.jnt_type[j]) == 2:
                    qs.append(float(d.qpos[int(self.m.jnt_qposadr[j])]))
            rows.append("[" + ",".join(f"{v:+.4f}" for v in qs) + "]")
        gears = ""
        try:
            if not hasattr(self, "_gear_jids"):
                self._gear_jids = (int(self.m.joint("gripper_servo_gear").id),
                                   int(self.m.joint("gripper_idol_gear").id))
            sv = float(d.qpos[int(self.m.jnt_qposadr[self._gear_jids[0]])])
            idl = float(d.qpos[int(self.m.jnt_qposadr[self._gear_jids[1]])])
            gears = f" gear(sv={sv:+.3f},id={idl:+.3f})"
        except (ValueError, KeyError):
            pass
        return f"slide{rows[0]}{rows[1]}{gears}"

    def _contact_dbg(self, d: mujoco.MjData) -> str:
        """Liet ke contact dang cham giua kep/linh kien muc tieu (debug DESCEND)."""
        if not hasattr(self, "_grip_root"):
            bid = int(self.m.body("gripper_Gripper").id)
            self._grip_root = int(self.m.body_rootid[bid])
        rows = []
        seen: set[tuple[int, int]] = set()
        for ci in range(int(d.ncon)):
            c = d.contact[ci]
            ga, gb = int(c.geom1), int(c.geom2)
            ba = int(self.m.geom_bodyid[ga])
            bb = int(self.m.geom_bodyid[gb])
            hit = (ba == self._grip_root or bb == self._grip_root
                   or ba == self.target or bb == self.target)
            if not hit:
                continue
            key = (min(ga, gb), max(ga, gb))
            if key in seen:
                continue
            seen.add(key)
            def lbl(gi: int) -> str:
                bi = int(self.m.geom_bodyid[gi])
                nm = self.m.body(bi).name or f"b{bi}"
                gn = self.m.geom(gi).name or f"g{gi}"
                z = float(d.geom_xpos[gi][2])
                return f"{nm}:{gn}(z={z:.3f})"
            cp = c.pos
            rows.append(f"{lbl(ga)}~{lbl(gb)} d={c.dist*1000:+.1f}mm "
                        f"at=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f})")
        return " | ".join(rows) if rows else "(khong co)"

    def _part_extent(self, d: mujoco.MjData, b: int,
                     axis: np.ndarray | None = None) -> float:
        """Do rong linh kien theo truc kep cua dap (axis world, m)."""
        g = self._part_geom(b)
        if g is None:
            return 0.05
        if axis is None:
            axis = np.array([1.0, 0.0, 0.0])
        R = d.geom_xmat[g].reshape(3, 3)
        a, c, h = self.m.geom_size[g]
        return 2.0 * (abs(float(axis @ R[:, 0])) * a
                      + abs(float(axis @ R[:, 1])) * c
                      + abs(float(axis @ R[:, 2])) * h)

    def _part_hz(self, d: mujoco.MjData, b: int) -> float:
        """Nua chieu cao (z) cua linh kien theo the gioi han AABB world."""
        g = self._part_geom(b)
        if g is None:
            return 0.005
        R = d.geom_xmat[g].reshape(3, 3)
        a, c, h = self.m.geom_size[g]
        return float(abs(R[2, 0]) * a + abs(R[2, 1]) * c + abs(R[2, 2]) * h)

    def _part_top_z(self, d: mujoco.MjData, b: int) -> float:
        return float(d.xpos[b][2]) + self._part_hz(d, b)

    def _grasp_z(self, d: mujoco.MjData, b: int) -> float:
        """Z cua tip khi kep: day mieng kep cham duoi day linh kien.

        Khong cham sau hon nua (truoc day da day them vao dong phan thanh
        lam lac linh kien khi xuong).
        """
        p = d.xpos[b]
        bottom = float(p[2]) - self._part_hz(d, b)
        return max(bottom - C.GRASP_PAD_ABOVE, C.BIN_FLOOR_Z + 0.003)

    def _z_correct(self, d: mujoco.MjData, grasp_z: float) -> float:
        """Ha grasp_z khi mieng kep lech z -> mieng CAO cham qua tren linh kien.

        Co kep (servo/idol gear lech lich su) lam 2 ngon truot slide_x khac
        nhau -> 2 mieng kep lech nhau theo huong tiep can (world z khi xuong).
        jaw_offset chi bo sung xy; lech z khong bo sung duoc (di chuyen ca 2
        mieng cung luc la khong doi xung duoc). Neu mieng cao cham ca hon day
        linh kien -> kep dong mot ben, day linh kien ra khoi -> loose.  Ha tip
        de duoi mieng cao cham xuong day linh kien (vi tri thiet ke).
        """
        tip_z = float(d.site_xpos[self.kin.site_id][2])
        hb = []
        for g in (C.GAP_GEOM0, C.GAP_GEOM1):
            hz = float(self.m.geom_size[g][2])
            hb.append(float(d.geom_xpos[g][2]) - hz - tip_z)
        bottom = float(d.xpos[self.target][2]) - self._part_hz(d, self.target)
        z = min(grasp_z, bottom - max(hb))
        # khong cham sau: mieng thap va tip khong cham day khay
        z = max(z, C.BIN_FLOOR_Z + 0.003,
                C.BIN_FLOOR_Z + 0.002 - min(hb))
        z = max(z, grasp_z - C.PAD_Z_MAX_DROP)
        return z

    def _nearest_part(self, d: mujoco.MjData, x: float, y: float,
                      max_d: float = 0.08) -> int | None:
        best = None
        bd = max_d
        for b in self._available_parts(d):
            p = d.xpos[b]
            dd = float(np.hypot(p[0] - x, p[1] - y))
            if dd < bd:
                bd, best = dd, b
        return best

    def start_poke(self, d: mujoco.MjData, now: float, xy, dir_,
                   body: int | None = None) -> bool:
        """Chuoi "gal" (phe quet): day linh kien ke goc ra khoi vach truoc khi gap.

        Khoa khay xanh co vach; part nam sat vach khong the gap (pads dap vach).
        Huong day do Transporter tinh (song song vach, ra khoi goc). Kep mo
        khienh 'gap_des' = do rong part + luoi; 2 mieng kep "khung" part; quet
        ngang nguc part de day no dich ra khoi goc; quet xong AI snap lai.
        """
        if xy is None or dir_ is None:
            return False
        dir_ = np.asarray(dir_, float)
        n = float(np.linalg.norm(dir_))
        if n < 1e-9:
            return False
        dir3 = np.array([float(dir_[0] / n), float(dir_[1] / n), 0.0])
        b = body if body is not None else self._nearest_part(d, xy[0], xy[1])
        w = 0.03
        center = np.array([float(xy[0]), float(xy[1]), 0.0])
        if b is not None:
            p = d.xpos[b]
            center = np.array([float(p[0]), float(p[1]), 0.0])
            w = max(self._part_extent(d, b, dir3), 0.02)
        self.target = b
        self.close_axis = dir3
        self.gap_des = w + C.POKE_GAP_MARGIN
        self.poke_z = max(C.BIN_FLOOR_Z + 0.003,
                          self._grasp_z(d, b) if b is not None else center[2])
        back_face = center - dir3 * (w / 2.0)
        hand_start = back_face + dir3 * (self.gap_des / 2.0)
        hand_safe = hand_start + dir3 * 0.006            # menh lop duoc truoc khi cham
        hand_end = hand_start + dir3 * C.POKE_SWEEP_DIST
        settle = np.array([0.0, C.SOURCE_BIN[1], C.TIP_Z_HOVER])
        self._poke_pts = [
            (np.array([hand_start[0], hand_start[1], C.TIP_Z_HOVER]), 0.35),
            (np.array([hand_safe[0], hand_safe[1], C.TIP_Z_HOVER]), 0.15),
            (np.array([hand_safe[0], hand_safe[1], self.poke_z]), 0.05),
            (np.array([hand_start[0], hand_start[1], self.poke_z]), 0.05),
            (np.array([hand_end[0], hand_end[1], self.poke_z]), C.POKE_SPEED),
            (np.array([hand_end[0], hand_end[1], C.TIP_Z_HOVER]), 0.15),
            (np.array([settle[0], settle[1], C.TIP_Z_HOVER]), 0.25),
        ]
        self._poll = 0
        self._poke_t0 = -1.0
        self.phase = self.POKE
        self.seg = None
        self.t_phase = now
        self.t0 = now
        self.cycle_outcome = 'poked'
        self.retry_request = False
        print(f"[poke] gal part=({center[0]:.3f},{center[1]:.3f}) "
              f"dir=({dir3[0]:+.2f},{dir3[1]:+.2f}) w={w*1000:.0f}mm "
              f"poke_z={self.poke_z:.3f} sweep={C.POKE_SWEEP_DIST:.3f}", flush=True)
        return True

    def _gripped_retry(self, d: mujoco.MjData) -> bool:
        """Nhan dien NGUY CO truot: linh kien nghieng qua sau khi kep cham."""
        if self.target is None:
            return False
        pg = self._part_geom(self.target)
        if pg is None:
            return False
        Rg = d.geom_xmat[pg].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(Rg[2, 2], -1.0, 1.0))))
        return tilt > C.SLIP_TILT_DEG

    def planned_drop_xy(self, d: mujoco.MjData) -> np.ndarray:
        """Vi tri tha du kien cho chu ky ke tiep (de danh dau tren viewer).

        Tra ve cung vi tri ma start_cycle se chon: dua tren cac linh kien
        dang nam o khay dich o thoi diem hien tai.
        """
        return self._pick_drop_xy(d)

    def start_cycle(self, d: mujoco.MjData, now: float) -> bool:
        b = None
        cx = cy = 0.0
        if self.external is not None:
            b = self.external.get("body")
        if b is None:
            b = self.select_target(d)
        if b is False or b is None:
            return False
        self.attempts += 1
        self.target = b
        self.target_name = self.m.body(b).name
        # bat dau chu ky: part nay dang duoc kep/tha; va chac chan cang kep
        # phai mo hoan toan truoc khi ha xuong gap
        self.carried = b
        self._preopen_done = False
        self._preopen_t0 = -1.0
        k = min(self.probe_idx.get(b, 0), len(C.GRASP_PROBES) - 1)
        dx, dy = C.GRASP_PROBES[k]
        p = d.xpos[b]
        if self.external is not None:
            cx, cy = self.external["x"], self.external["y"]
        else:
            cx, cy = p[0], p[1]
        self.probe = np.array([dx, dy, 0.0])
        off = self._jaw_offset(d)
        # huong kep: xoay truc kep (close) trung truc dai cua linh kien (tranh
        # cham goc lam lat linh kien khi dong mieng kep)
        yaw = float(self.external["yaw"]) if (self.external is not None
                                              and self.external.get("yaw") is not None) \
            else self._part_yaw(d, b)
        self.yaw = yaw
        self.close_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        self.vfs_off = np.zeros(2)
        self.hover = np.array([cx + dx - off[0], cy + dy - off[1],
                               C.TIP_Z_HOVER])
        self.grasp_z = self._grasp_z(d, b)
        self.squeeze_z = self._part_top_z(d, b) - C.PAD_BELOW_TIP
        self.squeezing = False
        self.gap_des = self._part_extent(d, b) + C.GAP_PRE_MARGIN
        self.grasp = np.array([cx + dx - off[0], cy + dy - off[1],
                               self.grasp_z])
        # Vi tri tren khay dich cho chu ky nay: chon lai, tranh linh kien
        # da tha truoc do (neu van dung vi tri cu se tha cham len no).
        self.drop_mark = self._pick_drop_xy(d)
        self.transfer = np.array([self.drop_mark[0], self.drop_mark[1],
                                  C.TIP_Z_HOVER])
        self.drop = np.array([self.drop_mark[0], self.drop_mark[1],
                              C.TIP_Z_DROP])
        print(f"[dbg] DROP_MARK moi chu ky -> "
              f"({self.drop_mark[0]:.3f},{self.drop_mark[1]:.3f})", flush=True)
        self.phase = self.HOVER
        self.seg = None
        self.t_phase = now
        self.cycle_outcome = 'pending'
        self.retry_request = False
        self.grip_open0 = self.grip_open_min = 100.0
        self.grip_touch_max = 0.0
        self.lift_from = 0.0
        self._load_high_t = -1.0
        self._hover_t0 = -1.0
        self._drop_t = -1.0
        self._drop_dbg = 0.0
        self._tr_t = -1.0
        return True

    def _go(self, now: float, p, vmax, amax=0.6, close=None) -> None:
        tip, a, c = self.kin.read_frame(self.d)
        close = self.close_axis if close is None else close
        self.seg = MoveSegment(tip, a, c, p, DOWN, close, vmax=vmax, amax=amax)
        self.t0 = now
        self.t_phase = now
        if self.bus is not None:
            self.bus.calibrate(self.d)

    def _hold_current(self, d: mujoco.MjData) -> PoseTarget:
        tip, a, c = self.kin.read_frame(d)
        return PoseTarget(tip, a, c)

    def _grip_ready(self, d: mujoco.MjData, now: float) -> bool:
        """Cang kep bat buoc mo du rong truoc khi ha xuong gắp.

        Tuyet doi khong kep (mieng kep hep/dong) khi bat dau thao tac gap:
        linh kien se khong loot duoc vao giua hai mieng -> loose chac chan.
        """
        if self.grip is None:
            return True
        gap = self.grip.gap(d)
        need = min(C.GRIP_OPEN_GAP, self.gap_des)
        if gap >= need - 0.002:
            return True
        if now - self._preopen_t0 >= C.GRIP_OPEN_TIMEOUT:
            # het thoi gian danh cho viec mo - quyet dinh tien hanh (best effort)
            print(f"[grip] BAN THIEU: gap={gap*1000:.0f}mm < need={need*1000:.0f}mm "
                  f"nhung het thoi gian mo", flush=True)
            return True
        return False

    def _reached(self, d: mujoco.MjData) -> bool:
        if self.seg is None:
            return True
        if self.phase == self.HOVER:
            tol = C.HOVER_TOL
        elif self.phase in (self.LIFT, self.TRANSFER, self.RETURN):
            tol = C.LIFT_TOL
        else:
            tol = C.WAYPT_TOL
        tip, _, _ = self.kin.read_frame(d)
        return bool(np.linalg.norm(tip - self.seg.p1) < tol)

    def fsm(self, d: mujoco.MjData, now: float) -> str:
        """Buoc may trang thai; tra 'run' | 'done'."""
        self.d = d
        ph = self.phase
        dt = now - self.t_phase
        if ph == self.HOVER:
            if self.seg is None:
                self._go(now, self.hover, 0.4)
            else:
                if self._reached(d):
                    if self._hover_t0 < 0:
                        self._hover_t0 = now
                else:
                    self._hover_t0 = -1.0
                stable = (self._hover_t0 >= 0
                          and (now - self._hover_t0 >= C.HOVER_DWELL))
                timeout = now - self.t0 > self.seg.T + 1.0
                if stable or timeout:
                    if self.refiner is not None and stable:
                        r_xy = self.refiner(d, self.target)
                        if r_xy is not None:
                            off = self._jaw_offset(d)
                            self.grasp = np.array([r_xy[0] - off[0],
                                                   r_xy[1] - off[1],
                                                   self.grasp_z])
                            print(f"[dbg] HOVER refine aim -> "
                                  f"({r_xy[0]:.3f},{r_xy[1]:.3f})",
                                  flush=True)
                    elif self.external is None:
                        # linh kien co the da lac/rot khi vy dung xuong ->
                        # lay lai vi tri hien tai truoc khi xuong
                        p_now = d.xpos[self.target]
                        _, _, cax = self.kin.read_frame(d)
                        self.gap_des = (self._part_extent(d, self.target, cax)
                                        + C.GAP_PRE_MARGIN)
                        self.grasp_z = self._grasp_z(d, self.target)
                        self.squeeze_z = (self._part_top_z(d, self.target)
                                          - C.PAD_BELOW_TIP)
                        self.squeezing = False
                        off = self._jaw_offset(d)
                        self.grasp = np.array([p_now[0] + self.probe[0]
                                               - off[0],
                                               p_now[1] + self.probe[1]
                                               - off[1],
                                               self.grasp_z])
                        print(f"[dbg] REAIM part=({p_now[0]:.3f},{p_now[1]:.3f},"
                              f"{p_now[2]:.3f}) -> grasp=({self.grasp[0]:.3f},"
                              f"{self.grasp[1]:.3f},{self.grasp[2]:.3f}) "
                              f"gap_des={self.gap_des*1000:.1f} "
                              f"jaw_off=({off[0]:+.3f},{off[1]:+.3f})",
                              flush=True)
                    z0 = self.grasp_z
                    self.grasp_z = self._z_correct(d, self.grasp_z)
                    self.grasp[2] = self.grasp_z
                    if abs(self.grasp_z - z0) > 0.001:
                        print(f"[dbg] z_corr grasp_z {z0:.3f} -> "
                              f"{self.grasp_z:.3f}", flush=True)
                    if not self._grip_ready(d, now):
                        # chua mo du rong (hoặc dang mo) -> giu HOVER, update()
                        # se tra lenh 'open' toi luc gap duoc / het thoi gian mo
                        return 'run'
                    self.phase = self.DESCEND
                    self._go(now, self.grasp, 0.06, 0.2)
        elif ph == self.DESCEND:
            tip, appr_v, close_v = self.kin.read_frame(d)
            load = 0.0 if self.bus is None else self.bus.contact_load(d)
            if not self.squeezing:
                trig = min(self.squeeze_z - C.SQUEEZE_ENGAGE,
                           self.grasp_z + C.SQUEEZE_LATE)
                if tip[2] <= trig:
                    self.squeezing = True
                    gp = ((self.grip.gap(d) * 1000.0)
                          if self.grip is not None else -1.0)
                    print(f"  [sqz] t={now:6.3f} tip_z={tip[2]:.3f} "
                          f"trig={trig:.3f} gap={gp:.1f}", flush=True)
            # load phai vuot nguong lien tuc mot luc thi moi la "chan that su"
            if load > C.FT_CONTACT_THRESHOLD:
                if self._load_high_t < 0:
                    self._load_high_t = now
            else:
                self._load_high_t = -1.0
            persisted = (self._load_high_t >= 0
                         and (now - self._load_high_t) >= C.FT_PERSIST_TIME)
            near = tip[2] <= self.grasp_z + C.FT_NEAR_Z
            safety = load > C.FT_SAFETY_THRESHOLD
            blocked = (safety or (persisted and near)) and tip[2] < self.hover[2] - 0.02
            if now - self._tr_t >= 0.01:
                self._tr_t = now
                pp = d.xpos[self.target]
                gpx = (self.grip.gap(d) * 1000.0) if self.grip is not None else -1.0
                g75 = d.geom_xpos[C.GAP_GEOM0]
                g90 = d.geom_xpos[C.GAP_GEOM1]
                tilt = 0.0
                pg = self._part_geom(self.target)
                if pg is not None:
                    Rg = d.geom_xmat[pg].reshape(3, 3)
                    tilt = float(np.degrees(np.arccos(
                        np.clip(Rg[2, 2], -1.0, 1.0))))
                print(f"  [desc] t={now:6.3f} tip=({tip[0]:.3f},{tip[1]:.3f},"
                      f"{tip[2]:.3f}) gap={gpx:.1f} "
                      f"part=({pp[0]:.3f},{pp[1]:.3f},{pp[2]:.3f}) "
                      f"dx={tip[0]-pp[0]:+.3f} dy={tip[1]-pp[1]:+.3f} "
                      f"g75x={g75[0]:.3f} g90x={g90[0]:.3f} "
                      f"tilt={tilt:.0f} load={load:.1f}", flush=True)
                if True:
                    print(f"       [con] {self._contact_dbg(d)}", flush=True)
            if (tip[2] <= self.grasp_z + 0.004 or blocked
                    or (self.seg is not None and now - self.t0 > self.seg.T + 0.4)):
                tip_p = d.xpos[self.target]
                gp = (self.grip.gap(d) * 1000.0) if self.grip is not None else -1.0
                cyaw = float(np.degrees(np.arctan2(close_v[1], close_v[0])))
                pg = self._part_geom(self.target)
                pyaw = 0.0
                if pg is not None:
                    Rg = d.geom_xmat[pg].reshape(3, 3)
                    pyaw = float(np.degrees(np.arctan2(Rg[1, 0], Rg[0, 0])))
                print(f"[dbg] DESCEND->GRIP tip=({tip[0]:.3f},{tip[1]:.3f},"
                      f"{tip[2]:.3f}) target=({self.grasp[0]:.3f},"
                      f"{self.grasp[1]:.3f}) grasp_z={self.grasp_z:.3f} "
                      f"gap={gp:.1f} gap_des={self.gap_des*1000:.1f} "
                      f"cyaw={cyaw:+.0f} pyaw={pyaw:+.0f} "
                      f"appr=({appr_v[0]:+.2f},{appr_v[1]:+.2f},"
                      f"{appr_v[2]:+.2f}) "
                      f"close=({close_v[0]:+.2f},{close_v[1]:+.2f},"
                      f"{close_v[2]:+.2f}) "
                      f"part=({tip_p[0]:.3f},{tip_p[1]:.3f},{tip_p[2]:.3f}) "
                      f"blocked={blocked} load={load:.1f}", flush=True)
                print(f"       [pad] {self._pads_dbg(d, tip)}", flush=True)
                self.phase = self.GRIP
                self.t_phase = now
                self._load_high_t = -1.0
                self.lift_from = float(d.xpos[self.target][2])
        elif ph == self.GRIP:
            dc = now - self.t_phase
            if dc >= C.GRIP_CLOSE_DURATION and self._gripped_retry(d):
                # Nguy co truot phat hien NGAY KHI KEP: part bi nghieng (nan kep
                # dang lop goc/mat) -> dung tay, mo kep va goi Deep Grasp tinh lai
                # toa do kep (re-snapshot + re-aim o phan ipc_node).
                tip, _, _ = self.kin.read_frame(d)
                pp = d.xpos[self.target]
                print(f"[SLIP] NGUY CO truot khi kep (tilt>{C.SLIP_TILT_DEG:.0f} do) "
                      f"tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) "
                      f"part=({pp[0]:.3f},{pp[1]:.3f},{pp[2]:.3f}) -> regrasp",
                      flush=True)
                print(f"       [con] {self._contact_dbg(d)}", flush=True)
                self.carried = None
                self.retry_request = True
                self.cycle_outcome = 'regrasp'
                self.phase = self.RETURN
                self._go(now, np.array([0.0, C.SOURCE_BIN[1], C.TIP_Z_HOVER]),
                         0.12, 0.35)
            elif dc > C.GRIP_CLOSE_DURATION + C.GRIP_SQUEEZE_DURATION + 0.05:
                tip, _, _ = self.kin.read_frame(d)
                q_end = self.bus.gear_q(d) if self.bus is not None else 0.0
                gp = (self.grip.gap(d) * 1000.0) if self.grip is not None else -1.0
                pp = d.xpos[self.target]
                print(f"[dbg] GRIP end q={q_end:.2f} gap={gp:.1f} "
                      f"tip_z={tip[2]:.3f} part=({pp[0]:.3f},{pp[1]:.3f},"
                      f"{pp[2]:.3f}) open_min={self.grip_open_min:.0f}mm "
                      f"touch_max={self.grip_touch_max:.2f}", flush=True)
                print(f"       [pad] {self._pads_dbg(d, tip)}", flush=True)
                self.cycle_outcome = 'gripped'
                self.phase = self.LIFT
                self._go(now, np.array([self.grasp[0], self.grasp[1], C.TIP_Z_HOVER]),
                         0.09, 0.25)
        elif ph == self.LIFT:
            p = d.xpos[self.target]
            dz = p[2] - self.lift_from
            tl = now - self.t_phase
            load = 0.0 if self.bus is None else self.bus.contact_load(d)
            if p[2] < self.lift_from - C.SLIP_FALLEN:
                # Linh kien TRUOT HAN ra khoi kep -> that bai (loose).
                print(f"[SLIP] part truot han: dz={dz:+.3f} < -{C.SLIP_FALLEN:.3f}"
                      f" -> loose", flush=True)
                self.cycle_outcome = 'loose'
                self.carried = None
                self.probe_idx[self.target] = self.probe_idx.get(self.target, 0) + 1
                if self.probe_idx[self.target] >= C.GRASP_MAX_PROBES:
                    self.picked.add(self.target)
                self.phase = self.RETURN
                self._go(now, np.array([0.0, C.SOURCE_BIN[1], C.TIP_Z_HOVER]),
                         0.15, 0.4)
            elif (tl > C.SLIP_LIFT_WAIT and dz < C.SLIP_LIFT_PROGRESS
                  and load > C.FT_CONTACT_THRESHOLD and self._gripped_retry(d)):
                # Dang kep NHAN part (co load) nhung part khong len duoc sau
                # SLIP_LIFT_WAIT va bi nghieng -> nguy co truot: dung lai va goi
                # Deep Grasp tinh lai toa do kep (regrasp).
                print(f"[SLIP] NGUY CO truot khi nang (part khong len sau "
                      f"{C.SLIP_LIFT_WAIT}s, dz={dz:+.3f}) -> regrasp", flush=True)
                self.cycle_outcome = 'regrasp'
                self.carried = None
                self.retry_request = True
                self.phase = self.RETURN
                self._go(now, np.array([0.0, C.SOURCE_BIN[1], C.TIP_Z_HOVER]),
                         0.12, 0.35)
            elif self._reached(d):
                print(f"[dbg] LIFT check dz={dz:+.3f} "
                      f"part=({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f})", flush=True)
                if dz > 0.02:
                    self.cycle_outcome = 'ok'
                    tip, _, _ = self.kin.read_frame(d)
                    # linh kien thuong nam lech so voi tip -> compens khi dat
                    self.held_off = np.array([p[0] - tip[0], p[1] - tip[1]])
                    print(f"[dbg] LIFT held_off=({self.held_off[0]:+.3f},"
                          f"{self.held_off[1]:+.3f}) tip=({tip[0]:.3f},"
                          f"{tip[1]:.3f})", flush=True)
                    self.transfer = np.array([self.drop_mark[0] - self.held_off[0],
                                              self.drop_mark[1] - self.held_off[1],
                                              C.TIP_Z_HOVER])
                    self.drop = np.array([self.drop_mark[0] - self.held_off[0],
                                          self.drop_mark[1] - self.held_off[1],
                                          C.TIP_Z_DROP])
                    self.phase = self.TRANSFER
                    self._go(now, self.transfer, 0.2, 0.5)
                else:
                    self.cycle_outcome = 'loose'
                    self.carried = None
                    self.probe_idx[self.target] = self.probe_idx.get(self.target, 0) + 1
                    if self.probe_idx[self.target] >= C.GRASP_MAX_PROBES:
                        self.picked.add(self.target)
                    self.phase = self.RETURN
                    self._go(now, np.array([0.0, C.SOURCE_BIN[1], C.TIP_Z_HOVER]),
                             0.15, 0.4)
        elif ph == self.TRANSFER:
            p = d.xpos[self.target]
            tip, _, _ = self.kin.read_frame(d)
            if p[2] < tip[2] - C.TRANSFER_SLIP_DROP:
                # Linh kien TRUOT KHOI KEP giua duong van chuyen -> that bai.
                print(f"[SLIP] part rot khoi kep khi van chuyen: "
                      f"part_z={p[2]:.3f} tip_z={tip[2]:.3f} -> loose", flush=True)
                self.cycle_outcome = 'loose'
                self.carried = None
                self.probe_idx[self.target] = self.probe_idx.get(self.target, 0) + 1
                if self.probe_idx[self.target] >= C.GRASP_MAX_PROBES:
                    self.picked.add(self.target)
                self.phase = self.RETURN
                self._go(now, np.array([0.0, C.SOURCE_BIN[1], C.TIP_Z_HOVER]),
                         0.2, 0.5)
            elif self._reached(d):
                self.cycle_outcome = 'ok' if self.cycle_outcome == 'gripped' else self.cycle_outcome
                self.phase = self.DROP
                self._go(now, self.drop, 0.1, 0.25)
        elif ph == self.DROP:
            arrived = (self._reached(d)
                       or (self.seg is not None and now - self.t0 > self.seg.T))
            if arrived:
                if self._drop_t < 0:
                    self._drop_t = now
                    self._drop_dbg = 0.0
                    tip, _, _ = self.kin.read_frame(d)
                    pp = d.xpos[self.target]
                    gp = (self.grip.gap(d) * 1000.0) if self.grip is not None else -1.0
                    print(f"[dbg] DROP arr part=({pp[0]:.3f},{pp[1]:.3f},"
                          f"{pp[2]:.3f}) tip=({tip[0]:.3f},{tip[1]:.3f},"
                          f"{tip[2]:.3f}) off=({pp[0]-tip[0]:+.3f},"
                          f"{pp[1]-tip[1]:+.3f}) gap={gp:.0f}", flush=True)
                    print(f"       [pad] {self._pads_dbg(d, tip)}", flush=True)
                    print(f"       [grip] ctrl={d.ctrl[self.grip.act_id]:+.2f}",
                          flush=True)
                elif now - self._drop_t >= self._drop_dbg + 0.05:
                    self._drop_dbg += 0.05
                    tip, _, _ = self.kin.read_frame(d)
                    pp = d.xpos[self.target]
                    gp = (self.grip.gap(d) * 1000.0) if self.grip is not None else -1.0
                    sv = (float(d.qpos[int(self.m.jnt_qposadr[
                              self._gear_jids[0]])])
                          if hasattr(self, "_gear_jids") else -9.0)
                    print(f"  [dw] dt={now-self._drop_t:.2f} gap={gp:.0f} "
                          f"sv={sv:+.3f} part=({pp[0]:.3f},{pp[1]:.3f},"
                          f"{pp[2]:.3f}) off=({pp[0]-tip[0]:+.3f},"
                          f"{pp[1]-tip[1]:+.3f},{pp[2]-tip[2]:+.3f})", flush=True)
                # giu them mot luc tai vi tri thap cho linh kien on dinh moi mo kep
                if now - self._drop_t >= 0.20:
                    self.phase = self.RELEASE
                    self.t_phase = now
            else:
                self._drop_t = -1.0
        elif ph == self.RELEASE:
            if dt > C.RELEASE_DWELL:
                tip, _, _ = self.kin.read_frame(d)
                pp = d.xpos[self.target]
                dxy = float(np.hypot(pp[0] - self.drop_mark[0],
                                     pp[1] - self.drop_mark[1]))
                print(f"[dbg] RELEASE part=({pp[0]:.3f},{pp[1]:.3f},"
                      f"{pp[2]:.3f}) tip=({tip[0]:.3f},{tip[1]:.3f},"
                      f"{tip[2]:.3f}) d_mark={dxy*1000:.0f}mm", flush=True)
                print(f"       [pad] {self._pads_dbg(d, tip)}", flush=True)
                print(f"       [grip] ctrl={d.ctrl[self.grip.act_id]:+.2f}",
                      flush=True)
                self.picked.add(self.target)
                self.carried = None
                self.phase = self.RETURN
                self._go(now, np.array([0.0, C.SOURCE_BIN[1], C.TIP_Z_HOVER]),
                         0.2, 0.5)
        elif ph == self.POKE:
            # Chuoi gal: hover -> (dung sau part) -> quet -> nang -> ve bin.
            # Xong chuoi -> 'poked' de ipc_node quet cam moi (AI tinh tiep).
            timeout = (self._poll >= len(self._poke_pts) and self._poke_t0 >= 0
                       and now - self._poke_t0 > 0.4)
            if timeout:
                self.gap_des = 0.06
                return 'poked'
            if self.seg is None or not self._reached(d):
                if self.seg is not None and now - self.t0 > self.seg.T + 1.5:
                    self._poll += 1                        # bi ket: bo qua seg
                    if self._poll < len(self._poke_pts):
                        p, v = self._poke_pts[self._poll]
                        self._go(now, p, v)
                    self._poke_t0 = now
                elif self.seg is None and self._poll < len(self._poke_pts):
                    p, v = self._poke_pts[self._poll]
                    self._go(now, p, v)
            elif self._poll < len(self._poke_pts):
                self._poll += 1
                self._poke_t0 = now if self._poll == len(self._poke_pts) else -1.0
                if self._poll < len(self._poke_pts):
                    p, v = self._poke_pts[self._poll]
                    self._go(now, p, v)
            if self._poll >= len(self._poke_pts) and self._poke_t0 >= 0 \
                    and now - self._poke_t0 > 0.4:
                self.gap_des = 0.06
                return 'poked'
        elif ph == self.RETURN:
            if self._reached(d) and now - self.t_phase > 0.3:
                if self.retry_request:
                    self.retry_request = False
                    return 'regrasp'
                return 'done'
        return 'run'

    def update(self, d: mujoco.MjData, now: float) -> tuple[PoseTarget, str]:
        """Tra ve (PoseTarget, grip_mode) cho vong dieu khien moi buoc."""
        ph = self.phase
        if ph in (self.GRIP, self.RELEASE):
            target = self._hold_current(d)
        else:
            t = now - self.t0 if self.seg is not None else 0.0
            target = self.seg.pose_at(t) if self.seg is not None else self._hold_current(d)
        if ph == self.DESCEND:
            # Vision Feedback Servoing: dich đeu vi tri ngang theo vfs_off
            # (duoc ipc_node tinh khi quan sat lai linh kien trong luc ha).
            if np.any(self.vfs_off):
                target.tip = target.tip + np.array([self.vfs_off[0], self.vfs_off[1], 0.0])
        if ph == self.HOVER and not self._preopen_done:
            g = self.grip.gap(d) if self.grip is not None else 1.0
            if self._preopen_t0 < 0.0:
                self._preopen_t0 = now
            if (g >= C.GRIP_OPEN_GAP
                    or now - self._preopen_t0 >= C.GRIP_OPEN_TIMEOUT):
                self._preopen_done = True
            else:
                # mo hoan toan cang kep voi luc lon truoc khi ha xuong
                return target, 'open'
        if ph == self.GRIP:
            dtg = now - self.t_phase
            mode = 'close' if dtg < C.GRIP_CLOSE_DURATION else 'squeeze'
        elif ph in (self.LIFT, self.TRANSFER, self.DROP):
            mode = 'hold'
        elif ph == self.RELEASE:
            mode = 'none'
        elif ph == self.POKE:
            # gal: mo kep dien 'gap_des' (khung part) trong ca chuoi quet - khi
            # handle o do cao quet se day mat lop tren part doc theo close_axis
            # (huong quet).
            mode = 'preclose'
        elif ph in (self.HOVER, self.DESCEND):
            # kep doi hinh truoc; khi mieng kep da cham duoc vao mat ben
            # linh kien -> kep nhe trong luc van xuong (tu can chinh)
            mode = ('softclose' if (ph == self.DESCEND and self.squeezing)
                    else 'preclose')
        else:
            mode = 'open'
        return target, mode