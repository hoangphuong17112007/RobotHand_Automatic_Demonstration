"""Thread AI (Chi huy Vĩ mô): nhan snapshot tu Bridge, chay pipeline 3 tang, pho Proposal.

Pipeline moi chu ky (xem README cua automation):
  1) Mask R-CNN           : nhan dang linh kien tu phan doan seg channel (ung vien).
  2) Transporter + ICP    : khao sat kha nang gap trong cum (clearance, phat hien vat
                            can) va kiem chung/loc pose 6D (BoxIcp). 
  3) Deep Grasping        : cham diem CNN tren patch depth goi den he so cuoi.
Dau ra CHI LA toa do/moc (Waypoint/Pose) + co GRASP_READY — khong bao gio them
bat cu truy xuat nao vao d.ctrl / momen dong co (Phan cap Quyen han).
"""
from __future__ import annotations

import threading
import time

import numpy as np

from . import config as C
from .bridge import Bridge, Proposal, Snapshot
from .icp import BoxIcp
from .graspai import CnnScorer, GraspProfile
from .transporter import TransporterNetwork


class AiNode(threading.Thread):
    """Chay doc lap voi thread mo phong. Gia mo thoi gian inference bang 'infer_sleep'."""

    def __init__(self, bridge: Bridge, task: int = 1, infer_sleep: float = 0.0,
                 min_move: float = 0.008, name: str = "ai_node"):
        super().__init__(daemon=True, name=name)
        self.bridge = bridge
        self.task = task
        self.infer_sleep = infer_sleep
        self.min_move = min_move
        self.scorer = CnnScorer()
        self.profile: GraspProfile = GraspProfile(self.scorer)
        self.transporter = TransporterNetwork()
        self.icp = BoxIcp()
        self.log: list[dict] = []
        self.last_prop: Proposal | None = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _fuse_score(self, base: float, ev, resid: float, inlier: float) -> float:
        s = float(base)
        if ev is not None:
            s += 0.30 * self.transporter.clearance_score(ev.clearance)
            if ev.graspable:
                s += 0.25
        r = float(np.clip(1.0 - resid / max(C.GRASP_ICP_RMSE_MAX, 1e-6), 0.0, 1.0))
        s += 0.15 * r * float(np.clip(inlier / 1.0, 0.0, 1.0))
        return s

    def _run_inference(self, snap: Snapshot) -> Proposal | None:
        cands = snap.candidates
        batches = snap.patches
        if not cands:
            return None
        blocked = self.bridge.blocked_ids()
        base = self.profile._scores(cands, batches, snap.depth, snap.seg)

        # Tang 2: Transporter + ICP (xep hang theo kha nang gap trong cum)
        subs_idx = [i for i, pc in enumerate(cands) if pc.body not in blocked]
        subs = [cands[i] for i in subs_idx]
        ev_by: dict[int, object] = {}
        poke_tgt: tuple[int, any] | None = None
        if subs:
            _ranked = self.transporter.rank(subs)
            for _c, (lid, ev) in enumerate(_ranked):
                ev_by[subs_idx[lid]] = ev
            poke_tgt = self.transporter.pick_poke(_ranked, subs)
            if poke_tgt is not None:
                poke_tgt = (subs_idx[poke_tgt[0]], poke_tgt[1])

        fin = []
        for ci, pc in enumerate(cands):
            if pc.body in blocked:
                continue
            ev = ev_by.get(ci)
            fin.append((self._fuse_score(float(base[ci]) if ci < len(base) else 0.0,
                                         ev, 9.9, 0.0), ci, ev))
        if not fin:
            fin = []
        any_graspable = any(
            t[2] is not None and t[2].graspable for t in fin
        )
        if (not any_graspable) and poke_tgt is not None and \
                self.bridge.poke_budget() > 0:
            # KHONG con ung vien nao gap duoc (tat ca ke goc/vach hoac bi can)
            # -> moi "gal" (last resort). Neuro khong gal khi van con part gap duoc
            # vi gal xoc don stack/truot len vach khay.
            return self._poke_proposal(snap, cands, poke_tgt, base)
        if not fin:
            return None
        if any_graspable:
            # uu tien ung vien gap duoc nhat (diem cao nhat) khi co it nhat 1.
            fin = [t for t in fin if t[2] is not None and t[2].graspable]
        fin.sort(key=lambda t: t[0], reverse=True)
        _, ci, ev = fin[0]
        pc = cands[ci]

        # Tang 3 + kiem chung pose (ICP) bang cloud diem cua linh kien da chon
        ix, iy, iyaw = pc.x, pc.y, pc.yaw
        resid, inlier = 9.9, 0.0
        cloud = snap.clouds[ci] if (snap.clouds and ci < len(snap.clouds)) else None
        if cloud is not None and len(cloud) >= 3:
            try:
                ix, iy, iyaw, resid, inlier = self.icp.fit(
                    cloud, pc.half, (pc.x, pc.y, pc.yaw), z_top=pc.z_top)
            except Exception:
                pass
        pose_ok = (resid <= C.GRASP_ICP_RMSE_MAX) and (inlier >= C.GRASP_ICP_INLIER_MIN)
        grasp_ready = bool(ev.graspable) if ev is not None else True
        grasp_ready = grasp_ready and pose_ok
        if self.infer_sleep > 0:
            time.sleep(self.infer_sleep)                      # gia thoi gian chay model
        final = self._fuse_score(float(base[ci]), ev, resid, inlier)
        if pc.z_top > 0.20 or pc.y < 0.30:                    # ngoai vung bin
            final -= 1.0
            grasp_ready = False
        blocked_obs: set[int] = set()
        if ev is not None and ev.blockers and C.BLOCK_OBSTACLES:
            blocked_obs = set(ev.blockers)
        prop = Proposal(ts=snap.ts, wall=time.time(),
                        x=ix, y=iy, z_top=pc.z_top, yaw=iyaw,
                        clearance=float(ev.clearance) if ev is not None else 1.0,
                        grasp_ready=grasp_ready, blocked=blocked_obs,
                        body=pc.body, name=pc.name, score=float(final))
        self.log.append({"snap_ts": snap.ts, "wall_ms": float('nan'),
                         "name": pc.name, "x": prop.x, "y": prop.y,
                         "yaw": prop.yaw, "clr": prop.clearance,
                         "ready": grasp_ready, "resid": resid,
                         "score": prop.score, "n_cands": len(cands)})
        return prop

    def _poke_proposal(self, snap: Snapshot, cands, ci_ev, base) -> Proposal | None:
        """De xuat "gal" (poke): day part ke goc ra khoi vach roi moi gap thuc su.

        Vi tri/do day do Transporter tinh (poke_xy/poke_dir) - RTC chay chuoi
        gal va SNAPSHOT lai de AI tinh tiep, so lan bi gioi han boi poke budget.
        """
        ci, ev = ci_ev
        pc = cands[ci]
        if snap.clouds and ci < len(snap.clouds) and self.infer_sleep > 0:
            time.sleep(self.infer_sleep)
        if ev is None or ev.poke_dir is None:
            return None
        score = self._fuse_score(float(base[ci]) if ci < len(base) else 0.0,
                                 ev, 9.9, 0.0)
        prop = Proposal(ts=snap.ts, wall=time.time(),
                        x=float(pc.x), y=float(pc.y), z_top=float(pc.z_top),
                        yaw=float(pc.yaw),
                        clearance=float(ev.clearance),
                        grasp_ready=False, blocked=set(),
                        body=pc.body, name=pc.name, score=float(score),
                        action="poke", poke_xy=(float(pc.x), float(pc.y)),
                        poke_dir=(float(ev.poke_dir[0]), float(ev.poke_dir[1])))
        self.log.append({"snap_ts": snap.ts, "wall_ms": float('nan'), "name": pc.name,
                         "x": pc.x, "y": pc.y, "yaw": pc.yaw, "clr": ev.clearance,
                         "ready": False, "resid": 9.9, "score": prop.score,
                         "action": "poke", "n_cands": len(cands)})
        return prop

    def run(self) -> None:
        last_snap_ts = -1.0
        loop = 0
        while not self.bridge.stopped and not self._stop.is_set():
            snap = self.bridge.snapshot_after(last_snap_ts)   # O(1): khong block
            tn = time.time()
            if snap is None:
                time.sleep(0.01)
                loop += 1
                continue
            last_snap_ts = snap.ts
            try:
                prop = self._run_inference(snap)
            except Exception:
                import traceback
                traceback.print_exc()
                prop = None
            if prop is None:
                continue
            if self.log:
                self.log[-1]["wall_ms"] = (time.time() - tn) * 1000
            prev = self.last_prop
            if prev is not None and prev.ts == prop.ts and \
               abs(prev.x - prop.x) < self.min_move and abs(prev.y - prop.y) < self.min_move:
                continue                                     # khong doi nhieu -> giu nguyen
            self.bridge.publish_proposal(prop)
            self.last_prop = prop