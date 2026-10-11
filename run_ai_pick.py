"""Chay pipeline day du: IPC (mo phong + dieu khien + mission) + AI (thi giac + CNN) qua Bridge."""
from __future__ import annotations

import argparse
import time

from automation import config as C
from automation.bridge import Bridge
from automation.ai_node import AiNode
from automation.ipc_node import IpcNode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--task", type=int, choices=(1, 2), default=1)
    ap.add_argument("--infer-sleep", type=float, default=0.02,
                    help="gia thoi gian chay inference (s) moi snapshot")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="bit snapshot moi N buoc sim (0: chi dau ky)")
    args = ap.parse_args()

    bridge = Bridge(history_size=4)
    ipc = IpcNode(bridge, cycles=args.cycles, task=args.task,
                  snapshot_every=args.snapshot_every)
    ai = AiNode(bridge, task=args.task, infer_sleep=args.infer_sleep)
    t0 = time.perf_counter()
    ipc.start()
    ai.start()
    ipc.join()
    ai.stop()
    ai.join(timeout=5.0)

    summary = ipc.summary_report()
    results = summary.get("results", [])
    wall = summary.get("wall_time", time.perf_counter()) - t0
    if ipc.runner is not None:
        ipc.runner.save_csv(C.OUT_DIR / "telemetry" / "aipick.csv", tag=f"task{args.task}")
        ipc.runner.save_results(C.OUT_DIR / "results" / "aipick.csv")

    print("\n=== Ket qua pipeline IPC + AI ===")
    print(f"task={args.task} | cycles={len(results)} | sim_time={summary.get('sim_time', 0):.1f}s "
          f"| wall={wall:.1f}s | AI snapshots={len(ai.log)}")
    ok = sum(1 for r in results if r.get("outcome") == "ok")
    lost = sum(1 for r in results if r.get("outcome") == "lost")
    loose = sum(1 for r in results if r.get("outcome") == "loose")
    print(f"ok={ok} lost={lost} loose={loose}")
    print(f"{'cycle':>6} {'part':>9} {'outcome':>8} {'t(s)':>8} {'part_y':>8} {'ai':>4}")
    for r in results:
        print(f"{r.get('cycle', ''):>6} {r.get('part', ''):>9} {r.get('outcome', ''):>8} "
              f"{r.get('sim_dt', 0):>8.2f} {r.get('part_y', 0):>8.3f} "
              f"{str(r.get('ai', '')):>4}")
    for a in ai.log[:10]:
        print(f"  [AI] snap_ts={a['snap_ts']:.2f} score_ms={a['wall_ms']:.1f} "
              f"cand={a['n_cands']} pick={a['name']} s={a['score']:.2f} "
              f"act={a.get('action', 'grasp')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())