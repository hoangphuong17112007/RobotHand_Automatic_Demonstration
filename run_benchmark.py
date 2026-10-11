"""Chay N chu ky gat vung luc xon va xuat telemetry + ket qua + dashboard.

Chay: python run_benchmark.py [--cycles 8]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import sys

from automation import config as C
from automation.runner import Runner

MODEL_PATH = Path(__file__).with_name("main.xml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=8)
    args = ap.parse_args()

    runner = Runner(MODEL_PATH)
    runner.settle(2500)
    t0 = time.perf_counter()
    out = runner.run_cycles(n=args.cycles, reset_first=False, settle_steps=0)
    wall = time.perf_counter() - t0

    tel_csv = runner.save_csv(C.OUT_DIR / "telemetry" / "pick.csv", tag="pick")
    res_csv = runner.save_results(C.OUT_DIR / "results" / "pick.csv")
    print("\n=== KET QUA CHU KY ===")
    for r in out["results"]:
        print(f"{r['cycle']:>2} {r['part']:8s} {r['outcome']:8s} "
              f"sim={r['sim_dt']:6.2f}s wall={r['wall']:6.2f}s "
              f"drop=({r.get('drop_x', 0):.3f},{r.get('drop_y', 0):.3f}) "
              f"-> ({r['part_x']:.3f},{r['part_y']:.3f},{r['part_z']:.3f})")
    ok = sum(1 for r in out["results"] if r["outcome"] == "ok")
    print(f"\nThanh cong {ok}/{len(out['results'])}; sim {out['sim_time']:.1f}s; "
          f"wall {wall:.1f}s ({out['sim_time']/wall:.2f}x realtime)")
    warn = max((s["warning"] for s in runner.sample), default=0)
    print(f"warnings max = {warn}")
    print(f"telemetry -> {tel_csv}\nresults   -> {res_csv}")

    from automation.analytics import make_dashboard
    dash = make_dashboard(C.OUT_DIR)
    print(f"dashboard -> {dash}")


if __name__ == "__main__":
    main()