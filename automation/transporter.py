"""Transporter Network (dang danh gia kha nang gap) — danh gia cho thong bin lon xon.

Vai tro trong pipeline AI:
  Mask R-CNN cung cap UNG VIEN (segment). Tranh nay KHAO SAT kha nang gap cua
  tung ung vien trong cum linh kien: tinh khe gap (clearance) quanh vi tri gap
  theo huong kep (close axis da xoay theo yaw cua linh kien), phat hien cac linh
  kien LAN CAN DANG CAN TRO (blockers), va chi con trong lai nhung ung vien
  co du khoang trong de 2 ngon kep luot xuong ben canh. Cac vat can duoc danh dau
  de RTC "doi gat" (blocked) cho den khi khu vuc thong thoang.

Vach khay (walls): vach KHONG cho la vat can ao khi band kep cham tuong - part
  sat vach bi coi la KHONG the gap (corner) truoc khi duoc "gal" ra khoi goc.
  `poke_dir` (. vach gan nhat, ra khoi goc) la huong AI de xuat de gal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C


@dataclass
class GraspEval:
    index: int = -1
    clearance: float = 1.0
    graspable: bool = True
    blockers: set = field(default_factory=set)    # set[int] body id linh kien can tro
    corner: bool = False          # part ke vach khay (gap truc diem se dung vach)
    wall_gap: float = 9.9         # khe gap nho nhat den cac vach (m; <0 = cham vach)
    poke_dir: tuple = None        # (dx, dy) don vi: huong day part ra khoi goc


class TransporterNetwork:
    """Xep hang ung vien theo kha nang gap trong cum (transporter-style)."""

    def __init__(self, clear_req: float | None = None, clear_good: float | None = None,
                 band_extra: float | None = None, pad_win: float | None = None):
        self.clear_req = clear_req if clear_req is not None else C.GRASP_CLEAR_REQ
        self.clear_good = clear_good if clear_good is not None else C.GRASP_CLEAR_GOOD
        self.band_extra = band_extra if band_extra is not None else C.GRASP_BAND_EXTRA
        self.pad_win = pad_win if pad_win is not None else C.GRASP_PAD_WINDOW
        self.wall_block = bool(getattr(C, "TRANSPORTER_WALL_BLOCK", True))
        self.bin_cx = 0.0
        self.bin_cy = C.SOURCE_BIN[1]
        self.x_wall = float(getattr(C, "SOURCE_BIN_X_MAX", 0.196))
        self.y_lo = float(getattr(C, "SOURCE_BIN_Y_LO", 0.298))
        self.y_hi = float(getattr(C, "SOURCE_BIN_Y_HI", 0.432))

    def _neighbor_gap(self, dxy: np.ndarray, ha: float, hb: float,
                      aj: float, bj: float) -> float:
        """Khoang cach giua 2 hinh chu nhat AA (local frame); 0 = dang cham nhau."""
        u, v = float(dxy[0]), float(dxy[1])
        dx = abs(u) - (ha + aj)
        dy = abs(v) - (hb + bj)
        if dx > 0.0 and dy > 0.0:
            return float(np.hypot(dx, dy))
        if dx > 0.0:
            return dx
        if dy > 0.0:
            return dy
        return 0.0

    def _clearance(self, me, others) -> tuple[float, set[int]]:
        """Khe gap nho nhat (m) giua 'me' va cac linh kien lan can (am = cham nhau)."""
        a, b = float(me.half[0]), float(me.half[1])
        cy, sy = np.cos(me.yaw), np.sin(me.yaw)
        R = np.array([[cy, sy], [-sy, cy]])
        c = np.array([me.x, me.y])
        z_thr = float(me.z_top) - C.GRASP_ZNECK
        best = 9.9
        blockers: set[int] = set()
        for j in others:
            if j.body == me.body:
                continue
            if float(j.z_top) < z_thr:                 # thap hon: kep vuot qua duoc
                continue
            d = np.array([j.x, j.y]) - c
            loc = R @ d                                # frame cua 'me'
            aj, bj = float(j.half[0]), float(j.half[1])
            hav = a + self.band_extra                      # vung 2 ngon kep o hai ben
            hbv = b + self.pad_win
            gap = self._neighbor_gap(loc, hav, hbv, aj, bj)
            if gap < best:
                best = gap
            if gap < self.clear_req:
                blockers.add(int(j.body))
        return best, blockers

    # ------------------------------------------------------------- vach khay

    def _part_extents(self, me) -> tuple[float, float]:
        """Do ra the gioi (m) cua PHAN THAN linh kien (rotated) + mot chut luoi."""
        a = float(me.half[0])
        b = float(me.half[1])
        cy, sy = np.cos(me.yaw), np.sin(me.yaw)
        hx = a * abs(cy) + b * abs(sy) + 0.003
        hy = a * abs(sy) + b * abs(cy) + 0.003
        return hx, hy

    def _wall_gaps(self, me) -> dict[str, float]:
        """Khe gap toi 4 vach trong (m; am = part dung SAt vach).

        Chi tinh the duong the o PHAN THAN linh kien (khong ke band kep): mieng
        kep manh 0.5mm chi can them ~3mm la du de luot vao sau part; neu danh dau
        band kep + pad_win se "ke goc" QUA NHIEU part binh thuong.
        """
        hx, hy = self._part_extents(me)
        x, y = float(me.x), float(me.y)
        wg = {
            "x+": self.x_wall - x - hx,
            "x-": self.x_wall + x - hx,
            "y+": self.y_hi - y - hy,
            "y-": y - self.y_lo - hy,
        }
        # Part da bi day RA NGOAI khay phia vach nao thi vach do khong con chan
        # (mieng kep xuong tu tren, khong dung vao vach o phia ngoai) -> bo qua
        # de part ngoai khay khong bi danh dau "ke goc" roi bi gal vo ich.
        if x > self.x_wall:
            wg["x+"] = 9.9
        if x < -self.x_wall:
            wg["x-"] = 9.9
        if y > self.y_hi:
            wg["y+"] = 9.9
        if y < self.y_lo:
            wg["y-"] = 9.9
        return wg

    def _push_dir(self, me, sides, wg) -> tuple[float, float]:
        """Huong day part ra khoi goc/vach (don vi; doc vach hoac truc goc).

        - SAT MOT VACH (khong phai goc): do ra vach theo phap tuyen (huong
          vao trong bin) - mieng kep manh (0.5mm) con khoang 2-5mm luot sau.
        - KE HAI VACH (goc that su): day DOC THEO 1 trong 2 vach (truoc tien)
          de ra khoi chu dinh cac goc; chon vach/chieu co khoang de trong
          phia sau (cho mieng kep) lon hon. Quet xong AI snap lai de tinh tiep.
        """
        x, y = float(me.x), float(me.y)
        cx, cy_ = self.bin_cx, self.bin_cy
        if len(sides) == 1:
            s = sides[0]
            if s == "y-":
                return (0.0, 1.0)
            if s == "y+":
                return (0.0, -1.0)
            if s == "x-":
                return (1.0, 0.0)
            return (-1.0, 0.0)
        # Goc: ung vien day doc vach (d dọc x hoặc dọc y huong ve trung tam)
        cand: list[tuple[tuple[float, float], float]] = []
        if "y+" in sides or "y-" in sides:
            d = (float(np.sign(cx - x)), 0.0)
            back = wg["x+"] if d[0] <= 0 else wg["x-"]
            cand.append((d, back))
        if "x+" in sides or "x-" in sides:
            d = (0.0, float(np.sign(cy_ - y)))
            back = wg["y+"] if d[1] <= 0 else wg["y-"]
            cand.append((d, back))
        if not cand:
            d = np.array([cx - x, cy_ - y])
            n = float(np.hypot(d[0], d[1]))
            return (1.0, 0.0) if n < 1e-9 else (float(d[0] / n), float(d[1] / n))
        best_d, best_b = cand[0][0], cand[0][1]
        for d, back in cand[1:]:
            if back > best_b:
                best_b, best_d = back, d
        return (float(best_d[0]), float(best_d[1]))

    def rank(self, cands) -> list[tuple[int, GraspEval]]:
        """Ung vien -> (index, GraspEval) xep giam theo kha nang gap."""
        out = []
        for i, me in enumerate(cands):
            half = getattr(me, "half", None)
            if half is None:
                half = (0.02, 0.015, 0.01)
            cl, blockers = self._clearance(me, cands)
            graspable = cl >= self.clear_req and not blockers
            corner = False
            wall_gap = 9.9
            sides: list[str] = []
            poke_dir = None
            if self.wall_block:
                wg = self._wall_gaps(me)
                wall_gap = float(min(wg.values()))
                sides = [s for s, g in wg.items() if g < self.clear_req]
                corner = wall_gap < self.clear_req
                if corner:
                    graspable = False
                    poke_dir = self._push_dir(me, sides, wg)
            out.append((i, GraspEval(index=i, clearance=cl,
                                     graspable=graspable, blockers=blockers,
                                     corner=corner, wall_gap=wall_gap,
                                     poke_dir=poke_dir)))
        out.sort(key=lambda t: (t[1].graspable, t[1].clearance), reverse=True)
        return out

    def _is_buried(self, me, cands) -> bool:
        """Part co part khac dang nam TREN (trung x,y va cao hon) -> khong gal
        (day se xoc do part ben tren -> loi khay/stack). Chi gal part tren cung."""
        a, b = self._part_extents(me)
        x, y = float(me.x), float(me.y)
        zt = float(me.z_top)
        for j in cands:
            if j is me:
                continue
            aj, bj = self._part_extents(j)
            if (abs(float(j.x) - x) < a + aj and abs(float(j.y) - y) < b + bj
                    and float(j.z_top) > zt + 0.004):
                return True
        return False

    def pick_poke(self, evs: list[tuple[int, GraspEval]],
                  cands) -> tuple[int, GraspEval] | None:
        """Chon ung vien "ke goc" de gal: corner + khe gap nho nhat.

        Chi gal part TREN CUNG (khong bi part khac ban len tren) de tranh xoc
        do stack. Day la "cuoi cung" - chi chay khi khong con ung vien nao gap duoc.
        """
        best_i, best_ev = None, None
        for i, ev in evs:
            if not ev.corner or ev.poke_dir is None:
                continue
            if cands and cands[ev.index] is not None and \
                    self._is_buried(cands[ev.index], cands):
                continue
            if best_ev is None or ev.wall_gap < best_ev.wall_gap:
                best_i, best_ev = i, ev
        if best_ev is None:
            return None
        return best_i, best_ev

    def clearance_score(self, cl: float) -> float:
        """0..1 the hien do thong thoang."""
        if cl >= self.clear_good:
            return 1.0
        if cl <= self.clear_req:
            return 0.0
        return float((cl - self.clear_req) / (self.clear_good - self.clear_req))