"""Thi giac: snapshot RGB-D-Seg tu camera tren co tay, unproject va trich xuat ung vien gắp."""
from __future__ import annotations

import numpy as np
import mujoco

from . import config as C
from .icp import yaw_from_cloud


class PartCandidate:
    __slots__ = ("name", "body", "x", "y", "z_top", "pixels", "confidence",
                 "pcx", "pcy", "half", "yaw")

    def __init__(self, name: str, body: int, x: float, y: float, z_top: float,
                 pixels: int, confidence: float = 0.5, pcx: int = -1, pcy: int = -1,
                 half=None, yaw: float = 0.0):
        self.name = name
        self.body = body
        self.x = x
        self.y = y
        self.z_top = z_top
        self.pixels = pixels
        self.confidence = confidence
        self.pcx = int(pcx)
        self.pcy = int(pcy)
        self.half = half if half is not None else (0.02, 0.015, 0.01)
        self.yaw = float(yaw)

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z_top])

    def __repr__(self) -> str:
        return (f"<PartCandidate {self.name} x={self.x:.3f} y={self.y:.3f} "
                f"z_top={self.z_top:.3f} px={self.pixels} conf={self.confidence:.2f}>")


class Camera:
    """Ung chi se nao camera 'wrist_cam' cua mo hinh va lam unproject pixel -> the gioi.

    Dang nhap: 1 pixel depth D = khoang cach tu diem den mat phang anh (doc theo truc look).
    world = pos + D*look + D*(right*u*tan_x + up*v*tan_y)
    """
    def __init__(self, m: mujoco.MjModel, camera: str = C.CAMERA, w: int = 640, h: int = 480):
        self.m = m
        self.camera = camera
        self.cid = m.camera(camera).id
        self.w, self.h = int(w), int(h)
        self.tan_y = float(np.tan(np.radians(m.cam_fovy[self.cid] / 2.0)))
        self.tan_x = self.tan_y * (self.w / self.h)
        self._rows, self._cols = np.meshgrid(np.arange(self.h), np.arange(self.w), indexing='ij')
        self._ndc_x = 2.0 * (self._cols + 0.5) / self.w - 1.0
        self._ndc_y = 1.0 - 2.0 * (self._rows + 0.5) / self.h
        self.renderer = mujoco.Renderer(m, self.h, self.w)
        self.part_ids = list(range(m.nbody))
        self.part_geom = [g for g in range(m.ngeom)
                          if m.body(m.geom_bodyid[g]).name.startswith(C.PART_BODIES_PREFIX)]
        self.part_of_body = {int(m.geom_bodyid[g]): g for g in self.part_geom}
        self._R = None

    def pose(self, d: mujoco.MjData):
        """Tra ve (pos, right, up, look) trong he toa do the gioi."""
        X = np.asarray(d.cam_xmat[self.cid]).reshape(3, 3)
        return (np.asarray(d.cam_xpos[self.cid]).copy(), X[:, 0], X[:, 1], -X[:, 2])

    def snapshot(self, d: mujoco.MjData):
        """Chup RGB, depth (m), segmentation (H,W,2: [geom_id, last_segid])."""
        r = self.renderer
        r.disable_segmentation_rendering()
        r.enable_depth_rendering()
        r.update_scene(d, camera=self.camera)
        rgb = np.array(r.render(), dtype=np.uint8)
        depth = np.array(r.render()).astype(np.float32)
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=self.camera)
        seg = np.array(r.render()).astype(np.int32)
        r.disable_segmentation_rendering()
        return rgb, depth, seg

    def unproject(self, depth: np.ndarray, d: mujoco.MjData) -> np.ndarray:
        """depth (H,W) -> (H,W,3) toa do the gioi moi pixel."""
        pos, right, up, look = self.pose(d)
        D = depth[..., None]
        lat = (right[None, None, :] * (self._ndc_x * self.tan_x)[..., None]
               + up[None, None, :] * (self._ndc_y * self.tan_y)[..., None])
        good = D[..., 0] > 0
        pts = pos + D * (look[None, None, :] + lat)
        pts[~good] = np.nan
        return pts

    def part_cloud(self, d: mujoco.MjData, seg: np.ndarray, depth: np.ndarray,
                   body: int) -> np.ndarray:
        """Diem the gioi cua cac pixel thuoc 1 part (bo NaN)."""
        geom = self.part_of_body.get(int(body))
        if geom is None:
            return np.zeros((0, 3))
        r, c = np.where(seg[..., 0] == geom)
        pts = self.unproject(depth, d)
        return pts[r, c]

    def part_z_top(self, m: mujoco.MjModel, d: mujoco.MjData, body: int) -> float:
        """Mat dinh (theo truc the gioi z) cua box part, do dung kich thuoc geometry."""
        geom = self.part_of_body.get(int(body))
        half = m.geom(geom).size
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.xquat[body])
        R = R.reshape(3, 3)
        top_local = half * np.array([0.0, 0.0, 1.0])
        return float(d.xpos[body][2] + R[2] @ top_local)

    def candidates(self, d: mujoco.MjData, min_pixels: int = 30) -> list[PartCandidate]:
        """Trich xuat ung vien gắp tu camera (theo segmentation + trung vi cloud).

        Vi tri (x,y) = median cua cloud thay duoc (chiu occlusion tot). z_top = mat tren
        box theo geometry (khoi doan theo chieu sau). yaw = PCA tren cloud ngang,
        half = kich thuoc box trong mo hinh (catalog CAD)."""
        rgb, depth, seg = self.snapshot(d)
        out = []
        body_by_geom = {g: int(self.m.geom_bodyid[g]) for g in self.part_geom}
        seen_geom = set(np.unique(seg[..., 0]))
        for body in self.part_ids:
            if body not in body_by_geom.values():
                continue
            geom = self.part_of_body.get(body)
            if geom not in seen_geom:
                continue
            geom = self.part_of_body.get(body)
            if geom is None:
                continue
            mask = seg[..., 0] == geom
            if int(mask.sum()) < min_pixels:
                continue
            r, c = np.where(mask)
            pts = self.unproject(depth, d)
            pts = pts[r, c]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < 10:
                continue
            x = float(np.median(pts[:, 0]))
            y = float(np.median(pts[:, 1]))
            rmid, cmid = int(np.median(r)), int(np.median(c))
            half = tuple(float(v) for v in self.m.geom(geom).size)
            yaw = float(yaw_from_cloud(pts))
            out.append(PartCandidate(
                name=self.m.body(body).name, body=body,
                x=x, y=y,
                z_top=self.part_z_top(self.m, d, body),
                pixels=int(mask.sum()), pcx=cmid, pcy=rmid,
                half=half, yaw=yaw))
        return out

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            pass