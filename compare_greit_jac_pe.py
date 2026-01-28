# coding: utf-8
"""
GREITとJACの位置特定精度比較 (Noise 1.0%)

Monte Carloシミュレーションによる統計的比較
- GREIT vs JAC の Position Error 比較
- エラーバー付きプロット
- 統計的有意差の評価
"""

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.greit as greit
import pyeit.eit.jac as jac
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from matplotlib.ticker import NullFormatter

# ============================================================================
# フォント設定
# ============================================================================

plt.rcParams["font.family"] = "Nimbus Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 21
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["xtick.labelsize"] = 16.5
plt.rcParams["ytick.labelsize"] = 16.5
plt.rcParams["legend.fontsize"] = 19.5
plt.rcParams["figure.titlesize"] = 24

# ============================================================================
# 1. 共通設定
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

# シミュレーション設定
NOISE_LEVEL = 0.01  # 1.0%
N_TRIALS = 10000

print("=" * 70)
print("=== GREIT vs JAC Comparison (Noise 1.0%) ===")
print("=" * 70)
print(f"Monte Carlo trials: {N_TRIALS}")
print(f"Noise level: {NOISE_LEVEL * 100:.1f}%")
print()

# ============================================================================
# 2. メッシュとプロトコルのセットアップ
# ============================================================================

mesh_obj = mesh.create(n_el, h0=0.1)
pts = mesh_obj.node
tri = mesh_obj.element

protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
fwd = EITForward(mesh_obj, protocol_obj)

# 基準電圧
v0_baseline = fwd.solve_eit(perm=background_perm)

# ============================================================================
# 3. GREIT Solver設定
# ============================================================================

print("Setting up GREIT solver...")
eit_greit = greit.GREIT(mesh_obj, protocol_obj)
eit_greit.setup(p=0.50, lamb=0.01, perm=background_perm, jac_normalized=True)

# ============================================================================
# 4. JAC Solver設定
# ============================================================================

print("Setting up JAC solver...")
eit_jac = jac.JAC(mesh_obj, protocol_obj)
eit_jac.setup(
    p=0.5, lamb=0.01, method="kotre", perm=background_perm, jac_normalized=True
)

# ============================================================================
# 5. Monte Carloシミュレーション
# ============================================================================

days_list = sorted(contrasts.keys())
contrasts_list = [contrasts[day] for day in days_list]

# 結果保存用
results_greit = {"mean": [], "std": [], "all_trials": []}
results_jac = {"mean": [], "std": [], "all_trials": []}

print("\nRunning Monte Carlo simulations...")
print("-" * 70)

for day in days_list:
    contrast = contrasts[day]

    print(f"Day {day:2d} ({contrast:4.2f}x)...", end=" ", flush=True)

    # 損傷メッシュの作成
    anomaly = PyEITAnomaly_Circle(
        center=list(anomaly_center),
        r=anomaly_radius,
        perm=background_perm * contrast,
    )
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=background_perm)

    # Forward計算
    v1 = fwd.solve_eit(perm=mesh_new.perm)

    # メッシュ要素の重心 (JAC用)
    element_centers = np.mean(pts[tri], axis=1)[:, :2]

    # 各試行
    pe_trials_greit = []
    pe_trials_jac = []

    for trial in range(N_TRIALS):
        # ノイズ付加
        noise = NOISE_LEVEL * np.random.normal(0, 1, len(v1))
        v1_noisy = v1 + noise * np.abs(v1)

        # --- GREIT 再構成 ---
        ds_greit = eit_greit.solve(v1_noisy, v0_baseline, normalize=True)
        x, y, ds_greit_grid = eit_greit.mask_value(ds_greit, mask_value=np.nan)
        ds_greit_real = np.real(ds_greit_grid)

        # GREIT: 最大値位置を特定
        ds_flat = ds_greit_real.flatten()
        valid_indices = ~np.isnan(ds_flat)

        if np.any(valid_indices):
            max_idx_flat = np.nanargmax(ds_flat)
            max_idx = np.unravel_index(max_idx_flat, ds_greit_real.shape)

            ny, nx = ds_greit_real.shape
            x_coord = -1 + (max_idx[1] + 0.5) * (2.0 / nx)
            y_coord = -1 + (max_idx[0] + 0.5) * (2.0 / ny)
            reconstructed_center_greit = np.array([x_coord, y_coord])

            pe_greit = np.linalg.norm(
                reconstructed_center_greit - np.array(anomaly_center)
            )
            pe_trials_greit.append(pe_greit)

        # --- JAC 再構成 ---
        ds_jac = eit_jac.solve(v1_noisy, v0_baseline, normalize=True)
        ds_jac_real = np.real(ds_jac)
        max_idx_jac = np.argmax(ds_jac_real)

        reconstructed_center_jac = element_centers[max_idx_jac]
        pe_jac = np.linalg.norm(reconstructed_center_jac - np.array(anomaly_center))
        pe_trials_jac.append(pe_jac)

    # 統計量を計算
    if pe_trials_greit:
        mean_greit = np.mean(pe_trials_greit)
        std_greit = np.std(pe_trials_greit)
    else:
        mean_greit = np.nan
        std_greit = np.nan

    mean_jac = np.mean(pe_trials_jac)
    std_jac = np.std(pe_trials_jac)

    results_greit["mean"].append(mean_greit)
    results_greit["std"].append(std_greit)
    results_greit["all_trials"].append(pe_trials_greit)

    results_jac["mean"].append(mean_jac)
    results_jac["std"].append(std_jac)
    results_jac["all_trials"].append(pe_trials_jac)

    print(f"GREIT: {mean_greit:.4f}±{std_greit:.4f}, JAC: {mean_jac:.4f}±{std_jac:.4f}")

# ============================================================================
# 6. プロット1: 比較グラフ (エラーバー付き)
# ============================================================================

print("\nGenerating comparison plots...")

fig, ax = plt.subplots(figsize=(12, 7))

# GREIT
ax.errorbar(
    contrasts_list,
    results_greit["mean"],
    yerr=results_greit["std"],
    marker="o",
    linewidth=2.5,
    markersize=9,
    color="darkblue",
    capsize=5,
    capthick=2,
    label="GREIT",
    alpha=0.8,
)

# JAC
ax.errorbar(
    contrasts_list,
    results_jac["mean"],
    yerr=results_jac["std"],
    marker="s",
    linewidth=2.5,
    markersize=9,
    color="darkgreen",
    capsize=5,
    capthick=2,
    label="JAC",
    alpha=0.8,
)

ax.set_xlabel("Conductivity Contrast (fold)", fontsize=21)
ax.set_ylabel("Position Error (distance)", fontsize=21)
ax.set_title(
    f"GREIT vs JAC: Position Accuracy Comparison (Noise {NOISE_LEVEL * 100:.1f}%, N={N_TRIALS})",
    fontsize=22.5,
)
ax.set_xscale("log")
ax.grid(True, alpha=0.3, which="both")
ax.axhline(
    y=anomaly_radius,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Anomaly radius (r={anomaly_radius})",
    alpha=0.7,
)
ax.legend(fontsize=19.5, loc="best")

# 上軸にDay番号を表示
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xscale("log")
ax2.xaxis.set_major_formatter(NullFormatter())
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.set_xticks(contrasts_list)
ax2.set_xticklabels([f"D{d}" for d in days_list])
ax2.set_xlabel("Day", fontsize=21)

plt.tight_layout()
plt.savefig("comparison_greit_jac_pe_noise1.0.png", dpi=150, bbox_inches="tight")
print("  -> comparison_greit_jac_pe_noise1.0.png saved")

# ============================================================================
# 7. プロット2: バイオリンプロット (分布比較)
# ============================================================================

fig2, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# GREIT
ax = axes[0]
non_empty_trials_greit = [
    trials for trials in results_greit["all_trials"] if len(trials) > 0
]
non_empty_contrasts_greit = [
    c
    for c, trials in zip(contrasts_list, results_greit["all_trials"])
    if len(trials) > 0
]

if non_empty_trials_greit:
    ax.violinplot(
        non_empty_trials_greit,
        positions=non_empty_contrasts_greit,
        widths=0.15,
        showmeans=True,
        showmedians=True,
    )

ax.set_xlabel("Conductivity Contrast (fold)", fontsize=19.5)
ax.set_ylabel("Position Error (distance)", fontsize=19.5)
ax.set_title(f"GREIT (Noise {NOISE_LEVEL * 100:.1f}%)", fontsize=21, fontweight="bold")
ax.set_xscale("log")
ax.grid(True, alpha=0.3, axis="y")
ax.axhline(y=anomaly_radius, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

# JAC
ax = axes[1]
ax.violinplot(
    results_jac["all_trials"],
    positions=contrasts_list,
    widths=0.15,
    showmeans=True,
    showmedians=True,
)

ax.set_xlabel("Conductivity Contrast (fold)", fontsize=19.5)
ax.set_title(f"JAC (Noise {NOISE_LEVEL * 100:.1f}%)", fontsize=21, fontweight="bold")
ax.set_xscale("log")
ax.grid(True, alpha=0.3, axis="y")
ax.axhline(y=anomaly_radius, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

fig2.suptitle("GREIT vs JAC: Position Error Distribution", fontsize=24, y=0.98)
plt.tight_layout()
plt.savefig(
    "comparison_greit_jac_distribution_noise1.0.png", dpi=150, bbox_inches="tight"
)
print("  -> comparison_greit_jac_distribution_noise1.0.png saved")

# ============================================================================
# 8. プロット3: 直接比較 (差分)
# ============================================================================

fig3, ax = plt.subplots(figsize=(12, 6))

# PE差分 (JAC - GREIT)
# NaN値の処理
mean_greit_array = np.array(results_greit["mean"])
mean_jac_array = np.array(results_jac["mean"])

# NaN以外のインデックス
valid_mask = ~np.isnan(mean_greit_array)
valid_contrasts = [c for c, v in zip(contrasts_list, valid_mask) if v]
valid_days = [d for d, v in zip(days_list, valid_mask) if v]

pe_diff = mean_jac_array[valid_mask] - mean_greit_array[valid_mask]

ax.plot(
    valid_contrasts,
    pe_diff,
    marker="D",
    linewidth=2.5,
    markersize=10,
    color="purple",
    label="PE difference (JAC - GREIT)",
)

ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.5)
ax.fill_between(
    valid_contrasts,
    0,
    pe_diff,
    where=(pe_diff >= 0),
    alpha=0.3,
    color="red",
    label="JAC worse",
)
ax.fill_between(
    valid_contrasts,
    0,
    pe_diff,
    where=(pe_diff < 0),
    alpha=0.3,
    color="blue",
    label="GREIT worse",
)

ax.set_xlabel("Conductivity Contrast (fold)", fontsize=21)
ax.set_ylabel("Position Error Difference (JAC - GREIT)", fontsize=21)
ax.set_title(
    f"GREIT vs JAC: Performance Difference (Noise {NOISE_LEVEL * 100:.1f}%)",
    fontsize=22.5,
)
ax.set_xscale("log")
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=18, loc="best")

# 上軸
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xscale("log")
ax2.xaxis.set_major_formatter(NullFormatter())
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.set_xticks(valid_contrasts)
ax2.set_xticklabels([f"D{d}" for d in valid_days])
ax2.set_xlabel("Day", fontsize=21)

plt.tight_layout()
plt.savefig(
    "comparison_greit_jac_difference_noise1.0.png", dpi=150, bbox_inches="tight"
)
print("  -> comparison_greit_jac_difference_noise1.0.png saved")

# ============================================================================
# 9. 統計サマリー
# ============================================================================

print("\n" + "=" * 70)
print("=== Comparison Summary ===")
print("=" * 70)

for i, day in enumerate(days_list):
    contrast = contrasts_list[i]

    mean_g = results_greit["mean"][i]
    std_g = results_greit["std"][i]
    mean_j = results_jac["mean"][i]
    std_j = results_jac["std"][i]

    if not np.isnan(mean_g):
        diff = mean_j - mean_g
        better = "GREIT" if diff > 0 else "JAC"

        print(f"\nDay {day:2d} ({contrast:.2f}x):")
        print(f"  GREIT: {mean_g:.4f} ± {std_g:.4f}")
        print(f"  JAC:   {mean_j:.4f} ± {std_j:.4f}")
        print(f"  Difference: {diff:+.4f} (Better: {better})")
    else:
        print(f"\nDay {day:2d} ({contrast:.2f}x):")
        print(f"  GREIT: NaN (no valid reconstructions)")
        print(f"  JAC:   {mean_j:.4f} ± {std_j:.4f}")

# 全体統計
valid_indices = [i for i, m in enumerate(results_greit["mean"]) if not np.isnan(m)]
if valid_indices:
    valid_diffs = [
        results_jac["mean"][i] - results_greit["mean"][i] for i in valid_indices
    ]

    print("\n" + "-" * 70)
    print("Overall Statistics (valid days only):")
    print(f"  Average difference (JAC - GREIT): {np.mean(valid_diffs):.4f}")
    print(f"  GREIT wins: {sum(1 for d in valid_diffs if d > 0)} days")
    print(f"  JAC wins:   {sum(1 for d in valid_diffs if d < 0)} days")

print("=" * 70)

plt.show()
