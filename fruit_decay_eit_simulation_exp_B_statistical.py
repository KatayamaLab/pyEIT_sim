# coding: utf-8
"""
果実腐敗進行に伴うEIT検出感度シミュレーション

実験B: 位置特定精度の統計的評価 (Monte Carlo Simulation)
- N回の試行による平均と標準偏差の算出
- エラーバー付きプロット
- 分布の可視化 (バイオリンプロット)
"""

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.jac as jac
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle

# ============================================================================
# 1. 入力データと前処理
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

# 導電率コントラストの計算: Contrast = R_day1 / R_i
R_day1 = resistance_data[1]
contrasts = {day: R_day1 / R for day, R in resistance_data.items()}

print("=== Conductivity Contrast (Day 1 baseline) ===")
for day, contrast in contrasts.items():
    print(f"Day {day:2d}: {contrast:.2f}x (R={resistance_data[day]:4d} Ohm)")
print()

# ============================================================================
# 2. シミュレーション環境のセットアップ
# ============================================================================

# パラメータ
n_el = 16  # 電極数
anomaly_center = (0.4, 0.0)  # 損傷位置
anomaly_radius = 0.15  # 損傷サイズ
background_perm = 1.0  # 背景導電率

# メッシュ生成
mesh_obj = mesh.create(n_el, h0=0.1)
pts = mesh_obj.node
tri = mesh_obj.element
x, y = pts[:, 0], pts[:, 1]

# プロトコル設定 (隣接法)
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

# Forward solver
fwd = EITForward(mesh_obj, protocol_obj)

# 基準電圧 (Day 1: 健康な果実)
v0_baseline = fwd.solve_eit(perm=background_perm)

# JAC solver設定
eit = jac.JAC(mesh_obj, protocol_obj)
eit.setup(p=0.5, lamb=0.01, method="kotre", perm=background_perm, jac_normalized=True)

# ============================================================================
# 3. 実験B: 位置特定精度の統計的評価 (Monte Carlo Simulation)
# ============================================================================

print("=== Experiment B: Statistical Position Accuracy Evaluation ===")

# モンテカルロシミュレーション設定
noise_levels_B = (0.001, 0.005, 0.01)  # 0.1%, 0.5%, 1.0%
N_TRIALS = 10000  # 試行回数

days_list = sorted(contrasts.keys())
contrasts_list = [contrasts[day] for day in days_list]

# 結果を保存する辞書 (平均, 標準偏差, 全データ)
position_errors_dict = {
    noise: {"mean": [], "std": [], "all_trials": []} for noise in noise_levels_B
}

for noise_level in noise_levels_B:
    print(f"\n--- Noise Level: {noise_level * 100:.1f}% (N={N_TRIALS} trials) ---")

    for day in days_list:
        contrast = contrasts[day]

        # 損傷メッシュの作成
        anomaly = PyEITAnomaly_Circle(
            center=list(anomaly_center),
            r=anomaly_radius,
            perm=background_perm * contrast,
        )
        mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=background_perm)

        # Forward計算 (ノイズなし基準電圧)
        v1 = fwd.solve_eit(perm=mesh_new.perm)

        # N回の試行
        position_errors_trials = []

        for trial in range(N_TRIALS):
            # ノイズ付加 (毎回異なる乱数)
            noise = noise_level * np.random.normal(0, 1, len(v1))
            v1_noisy = v1 + noise * np.abs(v1)

            # 再構成
            ds = eit.solve(v1_noisy, v0_baseline, normalize=True)

            # 再構成画像の最大値位置を特定
            ds_real = np.real(ds)
            max_idx = np.argmax(ds_real)

            # メッシュ要素の重心を計算
            element_centers = np.mean(pts[tri], axis=1)[:, :2]
            reconstructed_center = element_centers[max_idx]

            # Position Error計算
            true_center = np.array(anomaly_center)
            position_error = np.linalg.norm(reconstructed_center - true_center)
            position_errors_trials.append(position_error)

        # 統計量を計算
        mean_pe = np.mean(position_errors_trials)
        std_pe = np.std(position_errors_trials)

        position_errors_dict[noise_level]["mean"].append(mean_pe)
        position_errors_dict[noise_level]["std"].append(std_pe)
        position_errors_dict[noise_level]["all_trials"].append(position_errors_trials)

        print(
            f"Day {day:2d} ({contrast:4.2f}x): "
            f"PE = {mean_pe:.4f} ± {std_pe:.4f} (mean ± std)"
        )

# ============================================================================
# 4. プロット1: 平均値 + エラーバー
# ============================================================================

print("\nGenerating statistical plots...")

fig, ax = plt.subplots(figsize=(12, 7))

colors = ["navy", "darkgreen", "darkred"]
markers = ["o", "s", "^"]

for idx, noise_level in enumerate(noise_levels_B):
    mean_errors = position_errors_dict[noise_level]["mean"]
    std_errors = position_errors_dict[noise_level]["std"]

    ax.errorbar(
        contrasts_list,
        mean_errors,
        yerr=std_errors,
        marker=markers[idx],
        linewidth=2,
        markersize=8,
        color=colors[idx],
        capsize=5,
        capthick=2,
        label=f"Noise: {noise_level * 100:.1f}%",
    )

ax.set_xlabel("Conductivity Contrast (fold)", fontsize=14)
ax.set_ylabel("Position Error (distance)", fontsize=14)
ax.set_title(
    f"Experiment B: Position Accuracy vs Contrast (N={N_TRIALS} trials, mean ± std)",
    fontsize=15,
)
ax.set_xscale('log')  # 対数スケールに変更
ax.grid(True, alpha=0.3, which='both')  # major/minor両方のグリッドを表示
ax.axhline(
    y=anomaly_radius,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Anomaly radius (r={anomaly_radius})",
)
ax.legend(fontsize=12, loc="best")

# 二次軸でDay番号を表示
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xscale('log')  # 二次軸も対数スケールに
ax2.set_xticks(contrasts_list)
ax2.set_xticklabels([f"D{d}" for d in days_list])
ax2.set_xlabel("Day", fontsize=12)

plt.tight_layout()
plt.savefig("experiment_B_statistical_mean_std.png", dpi=150, bbox_inches="tight")
print("  -> experiment_B_statistical_mean_std.png saved")

# ============================================================================
# 5. プロット2: バイオリンプロット (分布の可視化)
# ============================================================================

fig2, axes = plt.subplots(1, len(noise_levels_B), figsize=(18, 6), sharey=True)

for idx, noise_level in enumerate(noise_levels_B):
    ax = axes[idx]
    all_trials = position_errors_dict[noise_level]["all_trials"]

    # バイオリンプロット
    parts = ax.violinplot(
        all_trials,
        positions=contrasts_list,
        widths=0.15,
        showmeans=True,
        showmedians=True,
    )

    ax.set_xlabel("Conductivity Contrast (fold)", fontsize=12)
    ax.set_title(f"Noise: {noise_level * 100:.1f}% (N={N_TRIALS})", fontsize=13)
    ax.set_xscale('log')  # 対数スケールに変更
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(
        y=anomaly_radius,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )

    if idx == 0:
        ax.set_ylabel("Position Error (distance)", fontsize=12)

fig2.suptitle("Experiment B: Position Error Distribution", fontsize=15, y=0.98)
plt.tight_layout()
plt.savefig("experiment_B_statistical_distribution.png", dpi=150, bbox_inches="tight")
print("  -> experiment_B_statistical_distribution.png saved")

# ============================================================================
# 6. Results Summary
# ============================================================================

print("\n" + "=" * 70)
print("=== Statistical Simulation Results Summary ===")
print("=" * 70)
print("Configuration:")
print(f"  - Number of electrodes: {n_el}")
print(f"  - Anomaly position: {anomaly_center}")
print(f"  - Anomaly radius: {anomaly_radius}")
print(f"  - Background conductivity: {background_perm}")
print(f"  - Monte Carlo trials: {N_TRIALS}")
print()
print("Experiment B Results (Statistical Evaluation):")
for noise_level in noise_levels_B:
    mean_errors = position_errors_dict[noise_level]["mean"]
    std_errors = position_errors_dict[noise_level]["std"]

    print(f"\n  Noise Level: {noise_level * 100:.1f}%")
    print("    - Best case (lowest mean PE):")
    min_idx = np.argmin(mean_errors)
    print(
        f"      Day {days_list[min_idx]} ({contrasts_list[min_idx]:.2f}x): "
        f"{mean_errors[min_idx]:.4f} ± {std_errors[min_idx]:.4f}"
    )

    print("    - Worst case (highest mean PE):")
    max_idx = np.argmax(mean_errors)
    print(
        f"      Day {days_list[max_idx]} ({contrasts_list[max_idx]:.2f}x): "
        f"{mean_errors[max_idx]:.4f} ± {std_errors[max_idx]:.4f}"
    )

    # 95%信頼区間がanomalyより小さい条件
    print("    - Reliable detection (mean + 2*std < radius):")
    reliable = [
        (d, c, m, s)
        for d, c, m, s in zip(days_list, contrasts_list, mean_errors, std_errors)
        if m + 2 * s < anomaly_radius
    ]
    if reliable:
        for d, c, m, s in reliable:
            print(f"      Day {d} ({c:.2f}x): {m:.4f} ± {s:.4f}")
    else:
        print("      None")

print("=" * 70)

plt.show()
