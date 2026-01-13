# coding: utf-8
"""
果実腐敗進行に伴うEIT検出感度シミュレーション

実験A: ノイズ耐性評価 (Visual Grid)
実験B: 位置特定精度の定量評価 (Position Error)
"""

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.jac as jac
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.eit.interp2d import sim2pts

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
anomaly_center = [0.4, 0.0]  # 損傷位置
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
# 3. 実験A: ノイズ耐性評価 (Visual Grid)
# ============================================================================

print("=== Experiment A: Noise Tolerance Evaluation ===")

# サンプリング設定
selected_days = [1, 3, 5, 8, 10]
noise_levels = [0.0, 0.001, 0.005, 0.01]  # 0%, 0.1%, 0.5%, 1.0%

fig, axes = plt.subplots(len(selected_days), len(noise_levels), figsize=(16, 20))
fig.suptitle("Experiment A: Decay Progress vs Noise Level", fontsize=16, y=0.995)

for i, day in enumerate(selected_days):
    contrast = contrasts[day]

    # 損傷メッシュの作成
    anomaly = PyEITAnomaly_Circle(
        center=anomaly_center, r=anomaly_radius, perm=background_perm * contrast
    )
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=background_perm)

    # Forward計算
    v1 = fwd.solve_eit(perm=mesh_new.perm)

    for j, noise_level in enumerate(noise_levels):
        # ノイズ付加
        if noise_level > 0:
            noise = noise_level * np.random.normal(0, 1, len(v1))
            v1_noisy = v1 + noise * np.abs(v1)  # 相対ノイズ
        else:
            v1_noisy = v1

        # 再構成
        ds = eit.solve(v1_noisy, v0_baseline, normalize=True)
        ds_n = sim2pts(pts, tri, np.real(ds))

        # プロット
        ax = axes[i, j]
        im = ax.tripcolor(x, y, tri, ds_n, shading="flat", cmap="RdBu_r")
        ax.set_aspect("equal")

        # タイトル設定
        if i == 0:
            ax.set_title(f"Noise: {noise_level * 100:.1f}%", fontsize=12)
        if j == 0:
            ax.set_ylabel(f"Day {day}\n({contrast:.2f}x)", fontsize=11)

        # カラーバー
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 真の損傷位置をマーク
        circle = plt.Circle(
            anomaly_center,
            anomaly_radius,
            color="green",
            fill=False,
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(circle)
        ax.plot(
            anomaly_center[0], anomaly_center[1], "g+", markersize=15, markeredgewidth=2
        )

plt.tight_layout()
plt.savefig("experiment_A_noise_tolerance.png", dpi=150, bbox_inches="tight")
print("Experiment A completed: experiment_A_noise_tolerance.png saved\n")

# ============================================================================
# 4. 実験B: 位置特定精度の定量評価 (Position Error)
# ============================================================================

print("=== Experiment B: Position Accuracy Evaluation ===")

# 固定ノイズレベル
fixed_noise_level = 0.001  # 0.1%

position_errors = []
days_list = sorted(contrasts.keys())
contrasts_list = [contrasts[day] for day in days_list]

for day in days_list:
    contrast = contrasts[day]

    # 損傷メッシュの作成
    anomaly = PyEITAnomaly_Circle(
        center=anomaly_center, r=anomaly_radius, perm=background_perm * contrast
    )
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=background_perm)

    # Forward計算
    v1 = fwd.solve_eit(perm=mesh_new.perm)

    # ノイズ付加
    noise = fixed_noise_level * np.random.normal(0, 1, len(v1))
    v1_noisy = v1 + noise * np.abs(v1)

    # 再構成
    ds = eit.solve(v1_noisy, v0_baseline, normalize=True)

    # 再構成画像の最大値位置を特定
    ds_real = np.real(ds)
    max_idx = np.argmax(ds_real)

    # メッシュ要素の重心を計算 (x, y座標のみ)
    element_centers = np.mean(pts[tri], axis=1)[:, :2]
    reconstructed_center = element_centers[max_idx]

    # Position Error計算
    true_center = np.array(anomaly_center)
    position_error = np.linalg.norm(reconstructed_center - true_center)
    position_errors.append(position_error)

    print(
        f"Day {day:2d} ({contrast:4.2f}x): PE = {position_error:.4f}, "
        f"Estimated pos = [{reconstructed_center[0]:.3f}, {reconstructed_center[1]:.3f}]"
    )

# プロット
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(contrasts_list, position_errors, "o-", linewidth=2, markersize=8, color="navy")
ax.set_xlabel("Conductivity Contrast (fold)", fontsize=14)
ax.set_ylabel("Position Error (distance)", fontsize=14)
ax.set_title(
    f"Experiment B: Position Accuracy vs Contrast\n(Noise Level: {fixed_noise_level * 100:.1f}%)",
    fontsize=15,
)
ax.grid(True, alpha=0.3)
ax.axhline(
    y=anomaly_radius,
    color="red",
    linestyle="--",
    label=f"Anomaly radius (r={anomaly_radius})",
)
ax.legend(fontsize=12)

# 二次軸でDay番号を表示
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(contrasts_list)
ax2.set_xticklabels([f"D{d}" for d in days_list])
ax2.set_xlabel("Day", fontsize=12)

plt.tight_layout()
plt.savefig("experiment_B_position_error.png", dpi=150, bbox_inches="tight")
print("\nExperiment B completed: experiment_B_position_error.png saved")

# ============================================================================
# 5. Results Summary
# ============================================================================

print("\n" + "=" * 70)
print("=== Simulation Results Summary ===")
print("=" * 70)
print(f"Configuration:")
print(f"  - Number of electrodes: {n_el}")
print(f"  - Anomaly position: {anomaly_center}")
print(f"  - Anomaly radius: {anomaly_radius}")
print(f"  - Background conductivity: {background_perm}")
print()
print(f"Experiment A Results:")
print(f"  - Test days: {len(selected_days)} (Days: {selected_days})")
print(
    f"  - Noise levels: {len(noise_levels)} ({[f'{n * 100:.1f}%' for n in noise_levels]})"
)
print(f"  - Total images: {len(selected_days) * len(noise_levels)}")
print()
print(f"Experiment B Results:")
print(
    f"  - Min PE: {min(position_errors):.4f} (Day {days_list[np.argmin(position_errors)]}, {contrasts_list[np.argmin(position_errors)]:.2f}x)"
)
print(
    f"  - Max PE: {max(position_errors):.4f} (Day {days_list[np.argmax(position_errors)]}, {contrasts_list[np.argmax(position_errors)]:.2f}x)"
)
print(f"  - Contrast threshold where PE < anomaly radius: ", end="")
threshold_contrasts = [
    c for c, pe in zip(contrasts_list, position_errors) if pe < anomaly_radius
]
if threshold_contrasts:
    print(f"{min(threshold_contrasts):.2f}x or higher")
else:
    print("None (PE > radius in all cases)")
print("=" * 70)

plt.show()
