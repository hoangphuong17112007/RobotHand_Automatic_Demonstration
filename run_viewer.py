# -*- coding: utf-8 -*-
"""Khao dong du lieu: mo Viewer 3D va TU DONG chay toan bo pipeline IPC + AI.

Cau truc an toan thread: mo phong, mission va camera deu chay tren MAIN THREAD
(qua IpcNode.advance() - mot buoc 1 tick) va duoc bao ve boi `vh.lock()` — cung
mutex ma viewer passive su dung khi copy mjData sang scene rieng de render. Nho do
mj_step khong bao gio trung voi `mj_copyDataVisual` cua viewer (`stack is in use`).
`vh.sync()` day frame moi nhat. AI chay song song trong thread rieng qua Bridge.
Dong viewer la ket thuc demo va ghi ket qua ra out/.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from automation import config as C
from automation.bridge import Bridge
from automation.ai_node import AiNode
from automation.ipc_node import IpcNode


def _draw_drop_marker(scene, runner) -> None:
    """Danh dau vi tri tha du kien cua chu ky ke tiep tren khay vang.

    Ve truc tiep vao user_scn cua viewer (khong sua model): mot dia mo the
    hien vung tha va mot cau nho o tam the hien chinh xac diem tha. Cap nhat
    moi frame -> sau khi ket thuc chu ky, marker nhay sang vi tri moi thong
    thoang (da tranh cac linh kien da tha truoc do).
    """
    if scene is None:
        return
    scene.ngeom = 0
    if runner is None:
        return
    xy = runner.mission.planned_drop_xy(runner.data)
    z = C.BIN_FLOOR_Z + 0.004
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([0.035, 0.0012, 0.0]),
        np.array([xy[0], xy[1], z]),
        mat,
        np.array([0.0, 0.85, 1.0, 0.30], dtype=np.float32),
    )
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.010, 0.0, 0.0]),
        np.array([xy[0], xy[1], z + 0.003]),
        mat,
        np.array([0.0, 0.85, 1.0, 0.95], dtype=np.float32),
    )
    scene.ngeom += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--task", type=int, choices=(1, 2), default=1)
    ap.add_argument("--infer-sleep", type=float, default=0.02)
    ap.add_argument("--snapshot-every", type=int, default=0)
    args = ap.parse_args()

    bridge = Bridge(history_size=4)
    ipc = IpcNode(bridge, cycles=args.cycles, task=args.task,
                  snapshot_every=args.snapshot_every)
    ai = AiNode(bridge, task=args.task, infer_sleep=args.infer_sleep)
    ai.start()

    # Runner + settle chay ngay tren main thread truoc khi mo viewer.
    ipc.prepare()
    print(f"[viewer] tay Robot tu dong gắp - task={args.task}, cycles={args.cycles}. "
          f"Dong cua so de ket thuc.", flush=True)
    try:
        with mujoco.viewer.launch_passive(ipc.runner.model, ipc.runner.data,
                                          show_left_ui=False) as vh:
            while vh.is_running():
                with vh.lock():
                    # Burst nhieu buoc sim moi frame de mo phong chay nhanh hon realtime,
                    # nhung van de viewer cap nhat lien tuc (khong giu lock qua lau).
                    burst = 0
                    status = "step"
                    while burst < 40 and not ipc.idle():
                        status = ipc.advance()
                        burst += 1
                        if status == "aim":
                            # Dang cho AI tinh: gia dung moi frame de AI kip xu ly.
                            break
                    # Danh dau vi tri tha cua chu ky ke tiep tren khay vang.
                    _draw_drop_marker(vh.user_scn, ipc.runner)
                vh.sync()
                time.sleep(0.001)
                if status == "done_all":
                    # Da chay xong so chu ky, giu viewer hien trang thai cuoi
                    # cho nguoi xem; dong cua so de ket thuc.
                    time.sleep(0.02)
    except Exception as e:
        print(f"[viewer] loi viewer: {e}", flush=True)
    finally:
        ipc.finish()
        ai.stop()
        ai.join(timeout=3.0)

    summary = ipc.summary_report()
    results = summary.get("results", [])
    if ipc.runner is not None:
        ipc.runner.save_csv(C.OUT_DIR / "telemetry" / "aipick.csv", tag=f"task{args.task}")
        ipc.runner.save_results(C.OUT_DIR / "results" / "aipick.csv")
    ok = sum(1 for r in results if r.get("outcome") == "ok")
    loose = sum(1 for r in results if r.get("outcome") == "loose")
    print(f"\n=== Demo ket thuc: {len(results)} chu ky | ok={ok} loose={loose} "
          f"| sim={summary.get('sim_time', 0):.1f}s | AI snapshots={len(ai.log)} ===")


if __name__ == "__main__":
    main()