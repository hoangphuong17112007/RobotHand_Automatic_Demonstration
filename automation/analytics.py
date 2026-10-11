"""Phan tich telemetry va ket qua: Force Profile, cycle time, ti le thanh cong."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE_NAMES = {0: "HOVER", 1: "DESCEND", 2: "GRIP", 3: "LIFT",
               4: "TRANSFER", 5: "DROP", 6: "RELEASE", 7: "RETURN"}
PHASE_COLORS = {0: "#9aa0a6", 1: "#1a73e8", 2: "#e37400", 3: "#188038",
                4: "#9334e6", 5: "#f9ab00", 6: "#d93025", 7: "#5f6368"}
SAFETY_FORCE = 25.0


def load(out_dir: Path | str, tel_name: str = "pick.csv",
         res_name: str = "pick.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    tel = pd.read_csv(out_dir / "telemetry" / tel_name, encoding="utf-8")
    res = pd.read_csv(out_dir / "results" / res_name, encoding="utf-8")
    return tel, res


def kpi_summary(res: pd.DataFrame) -> pd.DataFrame:
    ok = res["outcome"] == "ok"
    rows = {
        "so lan thu (cycles)": [int(len(res))],
        "thanh cong (ok)": [int(ok.sum())],
        "ko gat duoc (loose)": [int((res["outcome"] == "loose").sum())],
        "rot duong (lost)": [int((res["outcome"] == "lost").sum())],
        "ti le thanh cong (%)": [(100.0 * ok.mean()) if len(res) else 0.0],
        "cycle time TB (s)": [float(res.loc[ok, "sim_dt"].mean()) if ok.any() else np.nan],
        "cycle time min (s)": [float(res.loc[ok, "sim_dt"].min()) if ok.any() else np.nan],
        "cycle time max (s)": [float(res.loc[ok, "sim_dt"].max()) if ok.any() else np.nan],
        "tong sim time (s)": [float(res["sim_dt"].sum())],
    }
    return pd.DataFrame(rows, index=["giatri"]).T


def _apply_phase_color(ax: plt.Axes, tel: pd.DataFrame,
                       ylo: float, yhi: float, alpha: float = 0.15) -> None:
    for ph in tel["phase"].unique():
        sub = tel[tel["phase"] == ph]
        if sub.empty:
            continue
        ax.axvspan(sub["t"].min(), sub["t"].max(),
                   color=PHASE_COLORS.get(int(ph), "#888"),
                   alpha=alpha)
    prev = None
    for ph in tel["phase"].unique():
        sub = tel[tel["phase"] == ph]
        if sub.empty:
            continue
        t0 = sub["t"].min()
        ax.text(t0, yhi - 0.04 * (yhi - ylo), PHASE_NAMES.get(int(ph), str(ph)),
                fontsize=7, color=PHASE_COLORS.get(int(ph), "#333"))
    _ = prev


def plot_force_profile(tel: pd.DataFrame, save_to: Path,
                       title: str = "Force Profile - F/T sensor tai co tay") -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    t = tel["t"].values
    for k, comp, name, col in ((0, "f_x", "Fx (N)", "#d93025"),
                               (1, "f_y", "Fy (N)", "#1a73e8"),
                               (2, "f_z", "Fz (N)", "#188038")):
        ax = axes[k]
        ax.plot(t, tel[comp], color=col, lw=0.7)
        _apply_phase_color(ax, tel, float(np.minimum(tel[comp].min(), 0)),
                           float(tel[comp].max()))
        ax.axhline(SAFETY_FORCE, color="#888", ls="--", lw=1,
                   label="nguong an toan F/T")
        ax.axhline(-SAFETY_FORCE, color="#888", ls="--", lw=1)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel("thoi gian mo phong (s)")
    fig.tight_layout()
    fig.savefig(save_to, dpi=130)
    plt.close(fig)


def plot_cycle_times(res: pd.DataFrame, save_to: Path) -> None:
    ok = res[res["outcome"] == "ok"]
    colors = ["#188038" if a == "ok" else "#d93025" if a == "lost"
              else "#f9ab00" for a in res["outcome"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(res["cycle"].astype(str), res["sim_dt"], color=colors)
    for x, (r, a) in enumerate(zip(res["sim_dt"], res["outcome"])):
        ax.text(x, r + 0.2, f"{r:.1f}s", ha="center", fontsize=8)
    ax.set_ylabel("thoi gian chu ky (s sim)")
    ax.set_xlabel("chu ky")
    ax.set_title("Cycle Time - thoi gian moi thao tac gat")
    ax.grid(alpha=0.3, axis="y")
    if not ok.empty:
        ax.axhline(ok["sim_dt"].mean(), color="#188038", ls="--", lw=1.2,
                   label=f"TB chu ky thanh cong = {ok['sim_dt'].mean():.2f}s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_to, dpi=130)
    plt.close(fig)


def plot_tip_trajectory(tel: pd.DataFrame, save_to: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    ax = axes[0]
    for ph in tel["phase"].unique():
        sub = tel[tel["phase"] == ph]
        ax.plot(sub["tip_x"], sub["tip_y"], ".", ms=2.5,
                color=PHASE_COLORS.get(int(ph), "#888"), label=PHASE_NAMES.get(int(ph), str(ph)))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.legend(fontsize=7, ncol=4)
    ax.set_title("Quy dao tip (view tu tren)")
    ax = axes[1]
    ax.plot(tel["t"], tel["tip_z"], lw=0.8, color="#1a73e8")
    _apply_phase_color(ax, tel, float(tel["tip_z"].min()), float(tel["tip_z"].max()))
    ax.set_xlabel("thoi gian (s)")
    ax.set_ylabel("tip z (m)")
    ax.set_title("Do cao tip theo thoi gian")
    fig.tight_layout()
    fig.savefig(save_to, dpi=130)
    plt.close(fig)


def make_dashboard(out_dir: Path | str, tel: pd.DataFrame | None = None,
                   res: pd.DataFrame | None = None,
                   tel_name: str = "pick.csv",
                   res_name: str = "pick.csv",
                   suffix: str = "") -> Path:
    out_dir = Path(out_dir)
    dash = out_dir / "dashboard"
    dash.mkdir(parents=True, exist_ok=True)
    if tel is None or res is None:
        tel, res = load(out_dir, tel_name, res_name)
    su = f"_{suffix}" if suffix else ""
    plot_force_profile(tel, dash / f"force_profile{su}.png")
    plot_cycle_times(res, dash / f"cycle_times{su}.png")
    plot_tip_trajectory(tel, dash / f"trajectory{su}.png")
    kpi = kpi_summary(res)
    kpi.to_csv(dash / f"kpi{su}.csv", encoding="utf-8")
    return dash


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    d = make_dashboard(args.out)
    print("Dashboard: ", d)
    print(load(args.out))