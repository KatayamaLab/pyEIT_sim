# coding: utf-8
"""
GREIT vs JAC 比較: Day 5, 1.0% ノイズ
"""

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.jac as jac
import pyeit.eit.greit as greit
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.eit.interp2d import sim2pts
from logging_utils import setup_logging, finalize_logging
import csv
from datetime import datetime

# ============================================================================
# フォント設定
# ============================================================================

plt.rcParams["font.family"] = "Nimbus Roman"
plt.rcParams["font.size"] = 18  # 基本フォントサイズを1.5倍 (12 -> 18)
plt.rcParams["axes.labelsize"] = 21  # 軸ラベル (14 -> 21)
plt.rcParams["axes.titlesize"] = 24  # サブプロットタイトル (16 -> 24)
plt.rcParams["xtick.labelsize"] = 16.5  # x軸目盛り (11 -> 16.5)
plt.rcParams["ytick.labelsize"] = 16.5  # y軸目盛り (11 -> 16.5)
plt.rcParams["legend.fontsize"] = 18  # 凡例 (12 -> 18)
plt.rcParams["figure.titlesize"] = 24  # 図全体のタイトル (16 -> 24)

# ============================================================================
# 設定
# ============================================================================

# 抵抗値データ (Ohm)
resistance_data = {
    1: 5200,  # Day 1 (基準)
    2: 5000,
    3: 4700,
    4: 4500,
    5: 4200,
    6: 3500,
    7: 2500,
    8: 1800,
    9: 1100,
    10: 800,
}

# 導電率コントラストの計算
R_day1 = resistance_data[1]
contrasts = {day: R_day1 / R for day, R in resistance_data.items()}

# パラメータ
n_el = 16
anomaly_center = (0.4, 0.0)
anomaly_radius = 0.15
background_perm = 1.0
target_day = 5
noise_level = 0.01  # 1.0%

# ============================================================================
# ロギング設定
# ============================================================================

logger = setup_logging("log/compare_greit_jac_day5")

# メッシュ生成
mesh_obj = mesh.create(n_el, h0=0.1)
pts = mesh_obj.node
tri = mesh_obj.element
x, y = pts[:, 0], pts[:, 1]

# プロトコル設定
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

# Forward solver
fwd = EITForward(mesh_obj, protocol_obj)

# 基準電圧 (Day 1)
v0_baseline = fwd.solve_eit(perm=background_perm)

# ============================================================================
# Day 5 の測定データ生成
# ============================================================================

contrast = contrasts[target_day]
print(
    f"Day {target_day}: Contrast = {contrast:.2f}x (R={resistance_data[target_day]} Ohm)"
)
print(f"Noise level: {noise_level * 100:.1f}%\n")

# 損傷メッシュの作成
anomaly = PyEITAnomaly_Circle(
    center=list(anomaly_center), r=anomaly_radius, perm=background_perm * contrast
)
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=background_perm)

# Forward計算
v1 = fwd.solve_eit(perm=mesh_new.perm)

# ノイズ付加
np.random.seed(42)  # 再現性のため
noise = noise_level * np.random.normal(0, 1, len(v1))
v1_noisy = v1 + noise * np.abs(v1)

# ============================================================================
# JAC再構成
# ============================================================================

eit_jac = jac.JAC(mesh_obj, protocol_obj)
eit_jac.setup(
    p=0.5, lamb=0.01, method="kotre", perm=background_perm, jac_normalized=True
)

ds_jac = eit_jac.solve(v1_noisy, v0_baseline, normalize=True)
ds_jac_n = sim2pts(pts, tri, np.real(ds_jac))

# ============================================================================
# GREIT再構成
# ============================================================================

eit_greit = greit.GREIT(mesh_obj, protocol_obj)
eit_greit.setup(p=0.50, lamb=0.01, perm=background_perm, jac_normalized=True)

ds_greit = eit_greit.solve(v1_noisy, v0_baseline, normalize=True)
x_greit, y_greit, ds_greit_img = eit_greit.mask_value(ds_greit, mask_value=np.nan)
ds_greit_real = np.real(ds_greit_img)

# ============================================================================
# プロット: 2つ並べて比較
# ============================================================================

# GREITのカラースケール範囲に統一
vmin_common = np.nanmin(ds_greit_real)
vmax_common = np.nanmax(ds_greit_real)
print(f"\nColor scale (GREIT-based): vmin={vmin_common:.4f}, vmax={vmax_common:.4f}\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"GREIT vs JAC Comparison (Day {target_day}, Noise: {noise_level * 100:.1f}%)",
    fontsize=24,
    y=0.98,
)

# ----- JAC -----
ax_jac = axes[0]
im_jac = ax_jac.tripcolor(
    x, y, tri, ds_jac_n, shading="flat", cmap="jet", vmin=vmin_common, vmax=vmax_common
)
ax_jac.set_aspect("equal")
ax_jac.set_title(f"JAC Method\n(Contrast: {contrast:.2f}x)", fontsize=19.5)

# カラーバー
cbar_jac = plt.colorbar(im_jac, ax=ax_jac, fraction=0.046, pad=0.04)
cbar_jac.set_label("Conductivity Change", fontsize=18)

# 真の損傷位置をマーク
circle_jac = plt.Circle(
    anomaly_center,
    anomaly_radius,
    color="lime",
    fill=False,
    linewidth=2.5,
    linestyle="--",
    label="True anomaly",
)
ax_jac.add_patch(circle_jac)
ax_jac.plot(
    anomaly_center[0], anomaly_center[1], "g+", markersize=18, markeredgewidth=3
)
ax_jac.legend(loc="upper right", fontsize=13.5)

# ----- GREIT -----
ax_greit = axes[1]
im_greit = ax_greit.imshow(
    ds_greit_real,
    interpolation="none",
    cmap="jet",
    origin="lower",
    extent=[-1, 1, -1, 1],
    vmin=vmin_common,
    vmax=vmax_common,
)
ax_greit.set_aspect("equal")
ax_greit.set_title(f"GREIT Method\n(Contrast: {contrast:.2f}x)", fontsize=19.5)

# カラーバー
cbar_greit = plt.colorbar(im_greit, ax=ax_greit, fraction=0.046, pad=0.04)
cbar_greit.set_label("Conductivity Change", fontsize=18)

# 真の損傷位置をマーク
circle_greit = plt.Circle(
    anomaly_center,
    anomaly_radius,
    color="lime",
    fill=False,
    linewidth=2.5,
    linestyle="--",
    label="True anomaly",
)
ax_greit.add_patch(circle_greit)
ax_greit.plot(
    anomaly_center[0], anomaly_center[1], "g+", markersize=18, markeredgewidth=3
)
ax_greit.legend(loc="upper right", fontsize=13.5)

plt.tight_layout()
plt.savefig(
    "img/comparison_greit_vs_jac_day5_noise1.0.png", dpi=150, bbox_inches="tight"
)
print(f"Saved: img/comparison_greit_vs_jac_day5_noise1.0.png")

# ============================================================================
# 統計情報
# ============================================================================

print("\n" + "=" * 60)
print("統計情報")
print("=" * 60)
print(f"JAC:")
print(f"  Min value: {np.min(ds_jac_n):.4f}")
print(f"  Max value: {np.max(ds_jac_n):.4f}")
print(f"  Mean value: {np.mean(ds_jac_n):.4f}")
print(f"\nGREIT:")
print(f"  Min value: {np.nanmin(ds_greit_real):.4f}")
print(f"  Max value: {np.nanmax(ds_greit_real):.4f}")
print(f"  Mean value: {np.nanmean(ds_greit_real):.4f}")
print("=" * 60)

# ============================================================================
# CSV出力
# ============================================================================

print("\n" + "=" * 60)
print("=== Saving Statistics to CSV ===")
print("=" * 60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"csv/compare_greit_jac_day5_stats_{timestamp}.csv"

with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["Method", "Min_Value", "Max_Value", "Mean_Value", "Contrast", "Noise_Level"]
    )
    writer.writerow(
        [
            "JAC",
            f"{np.min(ds_jac_n):.6f}",
            f"{np.max(ds_jac_n):.6f}",
            f"{np.mean(ds_jac_n):.6f}",
            f"{contrast:.4f}",
            f"{noise_level * 100:.1f}%",
        ]
    )
    writer.writerow(
        [
            "GREIT",
            f"{np.nanmin(ds_greit_real):.6f}",
            f"{np.nanmax(ds_greit_real):.6f}",
            f"{np.nanmean(ds_greit_real):.6f}",
            f"{contrast:.4f}",
            f"{noise_level * 100:.1f}%",
        ]
    )

print(f"CSV saved: {csv_filename}")
print("=" * 60)

# ============================================================================
# ロギング終了
# ============================================================================

finalize_logging(logger)

plt.show()
