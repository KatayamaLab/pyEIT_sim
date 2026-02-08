# coding: utf-8
"""
GREIT vs JAC 比較: Day 4, 5, 6 の比較 (1.0% ノイズ)
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
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelsize"] = 21  # 軸ラベル (14 -> 21)
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titlesize"] = 24  # サブプロットタイトル (16 -> 24)
plt.rcParams["axes.titleweight"] = "bold"
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
target_days = [4, 5, 6]  # 比較するDay
noise_level = 0.01  # 1.0%

# ============================================================================
# ロギング設定
# ============================================================================

logger = setup_logging("log/compare_greit_jac_multi_days")

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
# 各Dayの処理
# ============================================================================

results = {}

for target_day in target_days:
    contrast = contrasts[target_day]
    print(
        f"Day {target_day}: Contrast = {contrast:.2f}x (R={resistance_data[target_day]} Ohm)"
    )

    # 損傷メッシュの作成
    anomaly = PyEITAnomaly_Circle(
        center=list(anomaly_center), r=anomaly_radius, perm=background_perm * contrast
    )
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=background_perm)

    # Forward計算
    v1 = fwd.solve_eit(perm=mesh_new.perm)

    # ノイズ付加
    np.random.seed(42 + target_day)  # 再現性のため
    noise = noise_level * np.random.normal(0, 1, len(v1))
    v1_noisy = v1 + noise * np.abs(v1)

    # JAC再構成
    eit_jac = jac.JAC(mesh_obj, protocol_obj)
    eit_jac.setup(
        p=0.5, lamb=0.01, method="kotre", perm=background_perm, jac_normalized=True
    )
    ds_jac = eit_jac.solve(v1_noisy, v0_baseline, normalize=True)
    ds_jac_n = sim2pts(pts, tri, np.real(ds_jac))

    # GREIT再構成
    eit_greit = greit.GREIT(mesh_obj, protocol_obj)
    eit_greit.setup(p=0.50, lamb=0.01, perm=background_perm, jac_normalized=True)
    ds_greit = eit_greit.solve(v1_noisy, v0_baseline, normalize=True)
    x_greit, y_greit, ds_greit_img = eit_greit.mask_value(ds_greit, mask_value=np.nan)
    ds_greit_real = np.real(ds_greit_img)

    # 結果を保存
    results[target_day] = {
        "contrast": contrast,
        "ds_jac_n": ds_jac_n,
        "ds_greit_real": ds_greit_real,
        "x_greit": x_greit,
        "y_greit": y_greit,
    }

print(f"\nNoise level: {noise_level * 100:.1f}%\n")

# ============================================================================
# プロット: 3行2列 (Day 3, 5, 8 を縦に並べる)
# ============================================================================

# 全体のカラースケール範囲を計算
all_greit_values = [results[day]["ds_greit_real"] for day in target_days]
vmin_common = min(np.nanmin(vals) for vals in all_greit_values)
vmax_common = max(np.nanmax(vals) for vals in all_greit_values)
print(
    f"\nOverall color scale (GREIT-based): vmin={vmin_common:.4f}, vmax={vmax_common:.4f}"
)

# Day 5 (真ん中) のカラースケール範囲に統一
ds_greit_day5 = results[5]["ds_greit_real"]
vmin_day5 = np.nanmin(ds_greit_day5)
vmax_day5 = np.nanmax(ds_greit_day5)
print(f"Day 5 color scale (using this): vmin={vmin_day5:.4f}, vmax={vmax_day5:.4f}\n")

fig, axes = plt.subplots(3, 2, figsize=(14, 18))
fig.suptitle(
    f"GREIT vs JAC Comparison (Days 4, 5, 6 | Noise: {noise_level * 100:.1f}%)",
    fontsize=24,
    y=0.995,
)

for idx, target_day in enumerate(target_days):
    result = results[target_day]
    contrast = result["contrast"]
    ds_jac_n = result["ds_jac_n"]
    ds_greit_real = result["ds_greit_real"]

    # ----- JAC (左列) -----
    ax_jac = axes[idx, 0]
    im_jac = ax_jac.tripcolor(
        x,
        y,
        tri,
        ds_jac_n,
        shading="flat",
        cmap="jet",
        vmin=vmin_day5,
        vmax=vmax_day5,
    )
    ax_jac.set_aspect("equal")
    ax_jac.set_title(
        f"JAC - Day {target_day}\n(Contrast: {contrast:.2f}x)", fontsize=19.5
    )

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

    # ----- GREIT (右列) -----
    ax_greit = axes[idx, 1]
    im_greit = ax_greit.imshow(
        ds_greit_real,
        interpolation="none",
        cmap="jet",
        origin="lower",
        extent=[-1, 1, -1, 1],
        vmin=vmin_day5,
        vmax=vmax_day5,
    )
    ax_greit.set_aspect("equal")
    ax_greit.set_title(
        f"GREIT - Day {target_day}\n(Contrast: {contrast:.2f}x)", fontsize=19.5
    )

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
    "img/comparison_greit_vs_jac_days456_noise1.0.png", dpi=150, bbox_inches="tight"
)
print("Saved: img/comparison_greit_vs_jac_days456_noise1.0.png")

# ============================================================================
# 統計情報
# ============================================================================

print("\n" + "=" * 60)
print("統計情報")
print("=" * 60)

for target_day in target_days:
    result = results[target_day]
    print(f"\n--- Day {target_day} ---")
    print("JAC:")
    print(f"  Min value: {np.min(result['ds_jac_n']):.4f}")
    print(f"  Max value: {np.max(result['ds_jac_n']):.4f}")
    print(f"  Mean value: {np.mean(result['ds_jac_n']):.4f}")
    print("\nGREIT:")
    print(f"  Min value: {np.nanmin(result['ds_greit_real']):.4f}")
    print(f"  Max value: {np.nanmax(result['ds_greit_real']):.4f}")
    print(f"  Mean value: {np.nanmean(result['ds_greit_real']):.4f}")

print("=" * 60)

# ============================================================================
# CSV出力
# ============================================================================

print("\n" + "=" * 60)
print("=== Saving Statistics to CSV ===")
print("=" * 60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"csv/compare_greit_jac_days456_stats_{timestamp}.csv"

with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Day",
            "Method",
            "Min_Value",
            "Max_Value",
            "Mean_Value",
            "Contrast",
            "Noise_Level",
        ]
    )

    for target_day in target_days:
        result = results[target_day]
        contrast = result["contrast"]

        writer.writerow(
            [
                f"Day{target_day}",
                "JAC",
                f"{np.min(result['ds_jac_n']):.6f}",
                f"{np.max(result['ds_jac_n']):.6f}",
                f"{np.mean(result['ds_jac_n']):.6f}",
                f"{contrast:.4f}",
                f"{noise_level * 100:.1f}%",
            ]
        )
        writer.writerow(
            [
                f"Day{target_day}",
                "GREIT",
                f"{np.nanmin(result['ds_greit_real']):.6f}",
                f"{np.nanmax(result['ds_greit_real']):.6f}",
                f"{np.nanmean(result['ds_greit_real']):.6f}",
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
