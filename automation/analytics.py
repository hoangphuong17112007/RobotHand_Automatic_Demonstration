"""Phân tích telemetry và kết quả: phổ lực (Force Profile), thời gian chu kỳ, tỉ lệ thành công."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Tên hiển thị (tiếng Việt có dấu) của từng pha trong chu kỳ gắp - đặt.
PHASE_NAMES = {0: "Lơ lửng (HOVER)", 1: "Hạ xuống (DESCEND)",
               2: "Kẹp (GRIP)", 3: "Nâng (LIFT)",
               4: "Vận chuyển (TRANSFER)", 5: "Thả (DROP)",
               6: "Nhả (RELEASE)", 7: "Quay về (RETURN)"}
PHASE_COLORS = {0: "#9aa0a6", 1: "#1a73e8", 2: "#e37400", 3: "#188038",
                4: "#9334e6", 5: "#f9ab00", 6: "#d93025", 7: "#5f6368"}

# Tên hiển thị (tiếng Việt có dấu) của từng kết quả chu kỳ.
OUTCOME_NAMES = {
    "ok": "Thành công (ok)",
    "off_mark": "Đặt lệch dấu (off_mark)",
    "loose": "Không gắp được (loose)",
    "regrasp": "Phải kẹp lại (regrasp)",
    "lost": "Rơi mất (lost)",
    "no_target": "Không có mục tiêu (no_target)",
    "timeout": "Quá thời gian (timeout)",
}
# Chỉ "ok" là chu kỳ thành công; mọi kết quả khác là chu kỳ thất bại.
SUCCESS_COLOR = "#188038"
FAILURE_COLOR = "#d93025"

SAFETY_FORCE = 25.0


def _is_success(outcome: str) -> bool:
    return outcome == "ok"


def load(out_dir: Path | str, tel_name: str = "pick.csv",
         res_name: str = "pick.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    tel = pd.read_csv(out_dir / "telemetry" / tel_name, encoding="utf-8")
    res = pd.read_csv(out_dir / "results" / res_name, encoding="utf-8")
    return tel, res


def kpi_summary(res: pd.DataFrame) -> pd.DataFrame:
    ok = res["outcome"] == "ok"
    rows = {
        "Số lần thử (chu kỳ)": [int(len(res))],
        "Thành công (ok)": [int(ok.sum())],
        "Không gắp được (loose)": [int((res["outcome"] == "loose").sum())],
        "Đặt lệch dấu (off_mark)": [int((res["outcome"] == "off_mark").sum())],
        "Rơi mất (lost)": [int((res["outcome"] == "lost").sum())],
        "Tỉ lệ thành công (%)": [(100.0 * ok.mean()) if len(res) else 0.0],
        "Thời gian chu kỳ TB (s)": [float(res.loc[ok, "sim_dt"].mean()) if ok.any() else np.nan],
        "Thời gian chu kỳ min (s)": [float(res.loc[ok, "sim_dt"].min()) if ok.any() else np.nan],
        "Thời gian chu kỳ max (s)": [float(res.loc[ok, "sim_dt"].max()) if ok.any() else np.nan],
        "Tổng thời gian mô phỏng (s)": [float(res["sim_dt"].sum())],
    }
    return pd.DataFrame(rows, index=["Giá trị"]).T


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
                       title: str = "Phổ lực F/T tại cổ tay khi gắp linh kiện") -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    t = tel["t"].values
    for k, comp, name, col in ((0, "f_x", "Fx (N)", "#d93025"),
                               (1, "f_y", "Fy (N)", "#1a73e8"),
                               (2, "f_z", "Fz (N)", "#188038")):
        ax = axes[k]
        ax.plot(t, tel[comp], color=col, lw=0.7, label=f"Lực {name}")
        _apply_phase_color(ax, tel, float(np.minimum(tel[comp].min(), 0)),
                           float(tel[comp].max()))
        ax.axhline(SAFETY_FORCE, color="#888", ls="--", lw=1,
                   label=f"Ngưỡng an toàn F/T ({SAFETY_FORCE:.0f} N)")
        ax.axhline(-SAFETY_FORCE, color="#888", ls="--", lw=1)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Thời gian mô phỏng (s)")
    fig.tight_layout()
    fig.savefig(save_to, dpi=130)
    plt.close(fig)


def plot_cycle_times(res: pd.DataFrame, save_to: Path,
                     title: str = "Thời gian chu kỳ gắp linh kiện theo từng chu kỳ") -> None:
    ok = res[res["outcome"] == "ok"]
    n_ok = int(ok.shape[0])
    n_fail = int(len(res) - n_ok)
    colors = [SUCCESS_COLOR if _is_success(a) else FAILURE_COLOR
              for a in res["outcome"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(res["cycle"].astype(str), res["sim_dt"], color=colors)
    for x, (r, a) in enumerate(zip(res["sim_dt"], res["outcome"])):
        ax.text(x, r + 0.2, f"{r:.1f}s", ha="center", fontsize=8)
    ax.set_ylabel("Thời gian chu kỳ (s mô phỏng)")
    ax.set_xlabel("Chu kỳ (số thứ tự)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    handles = [
        mpatches.Patch(color=SUCCESS_COLOR,
                       label=f"Chu kỳ thành công ({n_ok})"),
        mpatches.Patch(color=FAILURE_COLOR,
                       label=f"Chu kỳ thất bại ({n_fail})"),
    ]
    if not ok.empty:
        ln = ax.axhline(ok["sim_dt"].mean(), color=SUCCESS_COLOR, ls="--", lw=1.2,
                        label=f"TB chu kỳ thành công = {ok['sim_dt'].mean():.2f}s")
        handles.append(ln)
    ax.legend(handles=handles)
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
    ax.legend(fontsize=7, ncol=4, title="Pha")
    ax.set_title("Quỹ đạo đầu kẹp - nhìn từ trên")
    ax = axes[1]
    ax.plot(tel["t"], tel["tip_z"], lw=0.8, color="#1a73e8")
    _apply_phase_color(ax, tel, float(tel["tip_z"].min()), float(tel["tip_z"].max()))
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Độ cao đầu kẹp z (m)")
    ax.set_title("Độ cao đầu kẹp theo thời gian")
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
