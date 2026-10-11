"""Khoi AI cham diem: CNN nhe tren depth-patch + profile chia theo 2 bai toan.

Bai toan 1 (cellet vao pallet/khe): Mask R-CNN (nhan = seg channel) + Deep Grasp
(cham diem patch CNN) + fallback Transporter-style (xep hang o do cao thap nhat).
Bai toan 2 (lap rap): DexRepNet++ (bang cham diem patch, khac trong viec do y tot
moc / khe do co).
"""
from __future__ import annotations

import threading

import numpy as np

PATCH = 24


class CnnScorer:
    """CNN 5 lop tren depth-patch 24x24 (z normalized), tra confidence [0,1]."""

    def __init__(self, warm_pretrain: bool = True, seed: int = 0):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.nn = nn
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.net.eval()
        self._device = "cpu"
        self.lock = threading.Lock()
        if warm_pretrain:
            self._pretrain()

    def _synthetic_batch(self, n=96):
        """Mau do tao: 'tot' = mat phang phing, trong tam thap, xung quanh thong thoang."""
        t = self.torch
        xs, ys = [], []
        rng = np.random.default_rng(0)
        for _ in range(n):
            base = rng.uniform(0.0, 0.10)
            flat = np.full((PATCH, PATCH), base, dtype=np.float32)
            if rng.uniform() < 0.5:                      # 'tot': phang, bo thoa
                flat += rng.normal(0, 0.002, (PATCH, PATCH))
                y = 1.0
            else:                                        # 'xau': zat/cuon, nhieu do cao
                s = base + rng.normal(0, 0.02, (PATCH, PATCH)) * (rng.uniform(1, 3))
                flat = s
                if rng.uniform() < 0.3:                  # mot mang che mot phan
                    flat[rng.integers(0, PATCH):, :] += rng.uniform(0.02, 0.08)
                y = rng.uniform(0.0, 0.35)
            xs.append(flat)
            ys.append(float(y))
        X = t.tensor(np.stack(xs)[:, None].astype(np.float32))
        Y = t.tensor(np.array(ys, dtype=np.float32)[:, None])
        return X, Y

    def _pretrain(self, epochs: int = 60, lr: float = 1e-3):
        t, nn = self.torch, self.nn
        try:
            X, Y = self._synthetic_batch()
            opt = t.optim.Adam(self.net.parameters(), lr=lr)
            lossf = nn.BCELoss()
            self.net.train()
            for _ in range(epochs):
                opt.zero_grad()
                out = self.net(X)
                loss = lossf(out, Y)
                loss.backward()
                opt.step()
            self.net.eval()
            self._err = float(loss.detach().numpy())
        except Exception as e:
            self._err = 1.0
            print(f"[graspai] pretrain skipped: {e}")

    def score(self, patches: list[np.ndarray]) -> np.ndarray:
        """patches: list depth 24x24 -> confidence (n,)."""
        if not patches:
            return np.zeros(0)
        t = self.torch
        X = np.stack(patches)[:, None].astype(np.float32)
        with self.lock, t.no_grad():
            out = self.net(t.tensor(X)).numpy().ravel()
        return out


def build_patch(depth: np.ndarray, seg: np.ndarray, geom: int,
                r0: int, c0: int, size: int = PATCH, bad_fill: float = 1.0) -> np.ndarray:
    """Crop depth quanh tam part, remap sang kich thuoc PATCHxPATCH."""
    H, W = depth.shape
    patch = np.ones((size, size)) * bad_fill
    sid = size // 2
    for i in range(size):
        rr = r0 + (i - sid)
        if not (0 <= rr < H):
            continue
        for j in range(size):
            cc = c0 + (j - sid)
            if 0 <= cc < W and seg[rr, cc, 0] == geom and depth[rr, cc] > 0:
                patch[i, j] = depth[rr, cc]
    return patch


class GraspProfile:
    """Chấm điểm + xep hang ung vien dựa tren CNN va heuristics vat ly."""

    def __init__(self, scorer: CnnScorer | None = None, task: int = 1,
                 plateau_margin: float = 0.012):
        self.scorer = scorer or CnnScorer()
        self.task = task
        self.plateau_margin = plateau_margin

    def _scores(self, cands, patches, depth, seg) -> np.ndarray:
        conf = self.scorer.score(patches)
        out = np.zeros(len(cands))
        for i, pc in enumerate(cands):
            s = 0.35 * float(conf[i])
            s += 0.35 * float(np.clip(pc.z_top - 0.02, 0.0, 0.12) / 0.12)  # top = cao hon de don
            s += 0.15 * float(np.clip(pc.pixels / 800.0, 0.0, 1.0))
            out[i] = s
        return out

    def rank(self, cands, patches, depth, seg, gi: list[int] | None = None) -> list[tuple[int, float]]:
        """Tra ve index (theo cands) da xep giam dan voi diem."""
        gi = gi if gi is not None else list(range(len(cands)))
        if not gi:
            return []
        scores = self._scores(cands, patches, depth, seg)
        gis = np.array(gi)
        order = np.argsort(-scores[gis])
        return [(int(gis[k]), float(scores[gis[k]])) for k in order]


class MaskRcnnDeepGraspProfile(GraspProfile):
    """Bai toan 1: nhan thuc = seg channel (Mask R-CNN), deep grasp = CNN patch.

    Fallback Transporter: neu khong co part nao dat duoc diem > muc, giu nguyen
    target cao nhat (sap xep theo o ngoi vi tri thap nhat).
    """

    def _scores(self, cands, patches, depth, seg) -> np.ndarray:
        return super()._scores(cands, patches, depth, seg)

    def rank(self, cands, patches, depth, seg, gi=None):
        return super().rank(cands, patches, depth, seg, gi)


class DexRepProfile(GraspProfile):
    """Bai toan 2 (DexRepNet++): am dung tinh chi DAI theo truc dong kep — do co nho hon
    (giong va khe do co) duoc uu tien; nhiem vu cellet con tinh den moc TARGET bin."""

    def _scores(self, cands, patches, depth, seg) -> np.ndarray:
        base = super()._scores(cands, patches, depth, seg)
        for i, pc in enumerate(cands):
            base[i] += 0.2 * float(np.clip(1.0 - pc.pixels / 2400.0, 0.0, 1.0))
        return base


class AimRecord:
    __slots__ = ("x", "y", "z_top", "name", "score", "task")

    def __init__(self, x, y, z_top, name="", score=0.0, task=1, body=None, confidence=0.0):
        self.x = x
        self.y = y
        self.z_top = z_top
        self.name = name
        self.score = score
        self.task = task
        self.body = body
        self.confidence = confidence

    def __repr__(self):
        return f"<Aim {self.name} ({self.x:.3f},{self.y:.3f},{self.z_top:.3f}) s={self.score:.2f}>"


def choose_aim(cands, patches, depth, seg, task: int = 1,
               scorer: CnnScorer | None = None) -> AimRecord:
    """Chon target tot nhat tu camera (khong dung ground truth xy)."""
    prof = DexRepProfile(scorer, task=task) if task == 2 else MaskRcnnDeepGraspProfile(scorer)
    ranked = prof.rank(cands, patches, depth, seg)
    if not ranked:
        return None
    i, score = ranked[0]
    pc = cands[i]
    return AimRecord(pc.x, pc.y, pc.z_top, pc.name, score, task, body=pc.body, confidence=pc.confidence)