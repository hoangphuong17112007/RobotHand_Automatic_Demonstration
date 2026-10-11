"""Test kich ban gắp-linh-kien day du: gắp 1 linh kien tu thung nguon -> thung dich.

Chay: python test_pickplace.py
"""

import sys
import time
from pathlib import Path

import mujoco

from automation import config as C
from automation.runner import Runner

MODEL_PATH = Path(__file__).with_name("main.xml")


def main() -> None:
    runner = Runner(MODEL_PATH)
    runner.settle(2500)
    model = runner.model
    gid = model.actuator(C.GRIPPER_ACTUATOR).id

    print("Khoi tao xong. Bat dau chu ky gắp linh kien...")
    t0 = time.perf_counter()
    out = runner.run_cycles(n=1, reset_first=False, settle_steps=0)
    wall = time.perf_counter() - t0

    print("\n=== TELEMETRY (moi 10 buoc) ===")
    for s in runner.sample:
        print(f"t={s['t']:7.3f}s  phase={s['phase']}  "
              f"tip=({s['tip_x']:+.3f},{s['tip_y']:+.3f},{s['tip_z']:+.3f})  "
              f"F=({s['f_x']:+.1f},{s['f_y']:+.1f},{s['f_z']:+.1f})N  "
              f"touch={s['touch']:.3f} opening={s['opening']:.1f}mm")

    print("\n=== KET QUA CHU KY ===")
    for r in out["results"]:
        print(r)

    csv_path = runner.save_csv(C.OUT_DIR / "telemetry" / "g2_pickplace.csv", tag="g2")
    print(f"\nDbo telemetry -> {csv_path}  ({len(runner.sample)} mau)")
    print(f"tong thoi gian mo phong: {out['sim_time']:.2f}s  "
          f"wall: {wall:.2f}s  ({out['sim_time']/wall:.2f}x realtime)")
    print(f"warnings: max={max((s['warning'] for s in runner.sample), default=0)}")


if __name__ == "__main__":
    main()