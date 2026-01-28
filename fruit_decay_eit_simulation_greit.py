# coding: utf-8
"""
果実腐敗進行に伴うEIT検出感度シミュレーション (GREIT版)

実験A: ノイズ耐性評価 (Visual Grid)
実験B: 位置特定精度の定量評価 (Position Error)
"""

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.greit as greit
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from logging_utils import setup_logging, finalize_logging, CSVWriter

# ============================================================================
# フォント設定
# ============================================================================

plt.rcParams["font.family"] = "Nimbus Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 21
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["xtick.labelsize"] = 16.5
plt.rcParams["ytick.labelsize"] = 16.5
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["figure.titlesize"] = 24

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

# ============================================================================
# ロギング設定
# ============================================================================

logger = setup_logging("log/fruit_decay_eit_simulation_greit")

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

# プロトコル設定 (隣接法)
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

# Forward solver
fwd = EITForward(mesh_obj, protocol_obj)

# 基準電圧 (Day 1: 健康な果実)
v0_baseline = fwd.solve_eit(perm=background_perm)

# GREIT solver設定
eit = greit.GREIT(mesh_obj, protocol_obj)
eit.setup(p=0.50, lamb=0.01, perm=background_perm, jac_normalized=True)

# ============================================================================
# 3. 実験A: ノイズ耐性評価 (Visual Grid)
# ============================================================================

print("=== Experiment A: Noise Tolerance Evaluation ===")

# サンプリング設定
selected_days = (1, 3, 5, 8, 10)
noise_levels = (0.0, 0.001, 0.005, 0.01)  # 0%, 0.1%, 0.5%, 1.0%

# カラースケールの範囲を事前計算して統一
print("Calculating global color scale...")
all_reconstructions = []

for day in selected_days:
    contrast = contrasts[day]
    anomaly = PyEITAnomaly_Circle(
        center=list(anomaly_center), r=anomaly_radius, perm=background_perm * contrast
    )
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=background_perm)
    v1 = fwd.solve_eit(perm=mesh_new.perm)

    for noise_level in noise_levels:
        if noise_level > 0:
            noise = noise_level * np.random.normal(0, 1, len(v1))
            v1_noisy = v1 + noise * np.abs(v1)
        else:
            v1_noisy = v1

        ds = eit.solve(v1_noisy, v0_baseline, normalize=True)
        x, y, ds_greit = eit.mask_value(ds, mask_value=np.nan)
        all_reconstructions.append(np.real(ds_greit))

# 全画像での最小値・最大値を取得
vmin_global = np.nanmin([np.nanmin(recon) for recon in all_reconstructions])
vmax_global = np.nanmax([np.nanmax(recon) for recon in all_reconstructions])
print(f"Global color scale: vmin={vmin_global:.4f}, vmax={vmax_global:.4f}")

# 行ごと(Dayごと)の最小値・最大値を計算
vmin_per_day = []
vmax_per_day = []
for i in range(len(selected_days)):
    day_reconstructions = all_reconstructions[
        i * len(noise_levels) : (i + 1) * len(noise_levels)
    ]
    vmin_per_day.append(np.nanmin([np.nanmin(recon) for recon in day_reconstructions]))
    vmax_per_day.append(np.nanmax([np.nanmax(recon) for recon in day_reconstructions]))
    print(
        f"Day {selected_days[i]:2d} color scale: vmin={vmin_per_day[i]:.4f}, vmax={vmax_per_day[i]:.4f}"
    )
print()

# ======================================================================
# バージョン1: 行ごと統一スケール版 (Dayごとの比較用)
# ======================================================================
print("Generating row-wise unified scale version...")
fig1, axes1 = plt.subplots(len(selected_days), len(noise_levels), figsize=(16, 20))
fig1.suptitle(
    "Experiment A (GREIT): Row-wise Unified Scale (Per-Day Comparison)",
    fontsize=24,
    y=0.995,
)

recon_idx = 0
for i, day in enumerate(selected_days):
    contrast = contrasts[day]

    for j, noise_level in enumerate(noise_levels):
        ds_greit = all_reconstructions[recon_idx]
        recon_idx += 1

        # プロット(行ごとに統一されたカラースケール)
        ax = axes1[i, j]
        im = ax.imshow(
            ds_greit,
            interpolation="none",
            cmap="jet",
            vmin=vmin_per_day[i],
            vmax=vmax_per_day[i],
            origin="lower",
            extent=[-1, 1, -1, 1],
        )
        ax.set_aspect("equal")

        # タイトル設定
        if i == 0:
            ax.set_title(f"Noise: {noise_level * 100:.1f}%", fontsize=18)
        if j == 0:
            ax.set_ylabel(f"Day {day}\n({contrast:.2f}x)", fontsize=16.5)

        # カラーバー
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 真の損傷位置をマーク
        circle = plt.Circle(
            anomaly_center,
            anomaly_radius,
            color="lime",
            fill=False,
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(circle)
        ax.plot(
            anomaly_center[0], anomaly_center[1], "g+", markersize=15, markeredgewidth=2
        )

plt.tight_layout()
plt.savefig("img/experiment_A_greit_unified_scale.png", dpi=150, bbox_inches="tight")
print("  -> img/experiment_A_greit_unified_scale.png saved (row-wise scale)")

# ======================================================================
# バージョン2: 個別スケール版 (詳細観察用)
# ======================================================================
print("Generating individual scale version...")
fig2, axes2 = plt.subplots(len(selected_days), len(noise_levels), figsize=(16, 20))
fig2.suptitle(
    "Experiment A (GREIT): Individual Color Scale (Detail View)", fontsize=24, y=0.995
)

recon_idx = 0
for i, day in enumerate(selected_days):
    contrast = contrasts[day]

    for j, noise_level in enumerate(noise_levels):
        ds_greit = all_reconstructions[recon_idx]
        recon_idx += 1

        # プロット(個別カラースケール)
        ax = axes2[i, j]
        im = ax.imshow(
            ds_greit,
            interpolation="none",
            cmap="jet",
            origin="lower",
            extent=[-1, 1, -1, 1],
        )
        ax.set_aspect("equal")

        # タイトル設定
        if i == 0:
            ax.set_title(f"Noise: {noise_level * 100:.1f}%", fontsize=18)
        if j == 0:
            ax.set_ylabel(f"Day {day}\n({contrast:.2f}x)", fontsize=16.5)

        # カラーバー
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 真の損傷位置をマーク
        circle = plt.Circle(
            anomaly_center,
            anomaly_radius,
            color="lime",
            fill=False,
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(circle)
        ax.plot(
            anomaly_center[0], anomaly_center[1], "g+", markersize=15, markeredgewidth=2
        )

plt.tight_layout()
plt.savefig("img/experiment_A_greit_individual_scale.png", dpi=150, bbox_inches="tight")
print("  -> img/experiment_A_greit_individual_scale.png saved")
print("Experiment A completed: 2 versions saved\n")

# ============================================================================
# 4. 実験B: 位置特定精度の定量評価 (Position Error)
# ============================================================================

print("=== Experiment B: Position Accuracy Evaluation ===")

# 複数ノイズレベルでの評価
noise_levels_B = (0.001, 0.005, 0.01)  # 0.1%, 0.5%, 1.0%

days_list = sorted(contrasts.keys())
contrasts_list = [contrasts[day] for day in days_list]

# 結果を保存する辞書
position_errors_dict = {noise: [] for noise in noise_levels_B}

for noise_level in noise_levels_B:
    print(f"\n--- Noise Level: {noise_level * 100:.1f}% ---")

    for day in days_list:
        contrast = contrasts[day]

        # 損傷メッシュの作成
        anomaly = PyEITAnomaly_Circle(
            center=list(anomaly_center),
            r=anomaly_radius,
            perm=background_perm * contrast,
        )
        mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=background_perm)

        # Forward計算
        v1 = fwd.solve_eit(perm=mesh_new.perm)

        # ノイズ付加
        noise = noise_level * np.random.normal(0, 1, len(v1))
        v1_noisy = v1 + noise * np.abs(v1)

        # 再構成
        ds = eit.solve(v1_noisy, v0_baseline, normalize=True)
        x, y, ds_greit = eit.mask_value(ds, mask_value=np.nan)
        ds_real = np.real(ds_greit)

        # 再構成画像の最大値位置を特定
        # NaN値を無視して最大値のインデックスを取得
        ds_flat = ds_real.flatten()
        valid_indices = ~np.isnan(ds_flat)
        if np.any(valid_indices):
            max_idx_flat = np.nanargmax(ds_flat)
            max_idx = np.unravel_index(max_idx_flat, ds_real.shape)

            # GREITのグリッド座標系で位置を計算
            # ds_greitのshapeから座標を逆算
            ny, nx = ds_real.shape
            # extent=[-1, 1, -1, 1]に対応
            x_coord = -1 + (max_idx[1] + 0.5) * (2.0 / nx)
            y_coord = -1 + (max_idx[0] + 0.5) * (2.0 / ny)
            reconstructed_center = np.array([x_coord, y_coord])

            # Position Error計算
            true_center = np.array(anomaly_center)
            position_error = np.linalg.norm(reconstructed_center - true_center)
            position_errors_dict[noise_level].append(position_error)

            print(
                f"Day {day:2d} ({contrast:4.2f}x): PE = {position_error:.4f}, "
                f"Estimated pos = [{reconstructed_center[0]:.3f}, {reconstructed_center[1]:.3f}]"
            )
        else:
            # すべてNaNの場合
            position_errors_dict[noise_level].append(np.nan)
            print(f"Day {day:2d} ({contrast:4.2f}x): PE = NaN (all values are NaN)")

# プロット(複数ノイズレベル)
fig, ax = plt.subplots(figsize=(12, 7))

colors = ["navy", "darkgreen", "darkred"]
markers = ["o", "s", "^"]

for idx, noise_level in enumerate(noise_levels_B):
    position_errors = position_errors_dict[noise_level]
    ax.plot(
        contrasts_list,
        position_errors,
        marker=markers[idx],
        linewidth=2,
        markersize=8,
        color=colors[idx],
        label=f"Noise: {noise_level * 100:.1f}%",
    )

ax.set_xlabel("Conductivity Contrast (fold)", fontsize=21)
ax.set_ylabel("Position Error (distance)", fontsize=21)
ax.set_title(
    "Experiment B (GREIT): Position Accuracy vs Contrast (Multiple Noise Levels)",
    fontsize=22.5,
)
ax.set_xscale("log")  # 対数スケールに変更

# X軸の範囲を設定
ax.set_xlim(left=0.95)

# X軸のフォーマッタを設定(科学的記法を無効化)
formatter = ScalarFormatter()
formatter.set_scientific(False)
formatter.set_useOffset(False)
ax.xaxis.set_major_formatter(formatter)
# 補助目盛りも表示
ax.xaxis.set_minor_locator(LogLocator(subs="all"))
minor_formatter = ScalarFormatter()
minor_formatter.set_scientific(False)
minor_formatter.set_useOffset(False)
ax.xaxis.set_minor_formatter(minor_formatter)

ax.grid(True, alpha=0.3, which="both")  # major/minor両方のグリッドを表示
ax.axhline(
    y=anomaly_radius,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Anomaly radius (r={anomaly_radius})",
)
ax.legend(fontsize=18, loc="best")

# 上軸にDay番号を表示
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xscale("log")  # 二次軸も対数スケールに

# 自動生成される対数スケールの数値ラベルを消去する
ax2.xaxis.set_major_formatter(NullFormatter())
ax2.xaxis.set_minor_formatter(NullFormatter())

ax2.set_xticks(contrasts_list)
ax2.set_xticklabels([f"D{d}" for d in days_list])
ax2.set_xlabel("Day", fontsize=21)

plt.tight_layout()
plt.savefig("img/experiment_B_greit_position_error.png", dpi=150, bbox_inches="tight")
print("\nExperiment B completed: img/experiment_B_greit_position_error.png saved")

# ============================================================================
# 5. Results Summary
# ============================================================================

print("\n" + "=" * 70)
print("=== Simulation Results Summary (GREIT) ===")
print("=" * 70)
print("Configuration:")
print(f"  - Reconstruction method: GREIT")
print(f"  - Number of electrodes: {n_el}")
print(f"  - Anomaly position: {anomaly_center}")
print(f"  - Anomaly radius: {anomaly_radius}")
print(f"  - Background conductivity: {background_perm}")
print()
print("Experiment A Results:")
print(f"  - Test days: {len(selected_days)} (Days: {selected_days})")
print(
    f"  - Noise levels: {len(noise_levels)} ({[f'{n * 100:.1f}%' for n in noise_levels]})"
)
print(f"  - Total images: {len(selected_days) * len(noise_levels)}")
print("  - Output files: 2 versions (row-wise unified scale + individual scale)")
print()
print("Experiment B Results:")
for noise_level in noise_levels_B:
    position_errors = position_errors_dict[noise_level]
    # NaN値を除外
    valid_errors = [pe for pe in position_errors if not np.isnan(pe)]
    if valid_errors:
        min_pe_idx = np.argmin(position_errors)
        max_pe_idx = np.argmax(position_errors)
        print(f"\n  Noise Level: {noise_level * 100:.1f}%")
        print(
            f"    - Min PE: {min(valid_errors):.4f} (Day {days_list[min_pe_idx]}, {contrasts_list[min_pe_idx]:.2f}x)"
        )
        print(
            f"    - Max PE: {max(valid_errors):.4f} (Day {days_list[max_pe_idx]}, {contrasts_list[max_pe_idx]:.2f}x)"
        )
        print("    - Contrast threshold where PE < anomaly radius: ", end="")
        threshold_contrasts = [
            c
            for c, pe in zip(contrasts_list, position_errors)
            if not np.isnan(pe) and pe < anomaly_radius
        ]
        if threshold_contrasts:
            print(f"{min(threshold_contrasts):.2f}x or higher")
        else:
            print("None (PE > radius in all cases)")
    else:
        print(f"\n  Noise Level: {noise_level * 100:.1f}%")
        print("    - No valid results (all NaN)")

# ============================================================================
# CSV出力
# ============================================================================

print("\n" + "=" * 70)
print("=== Saving Results to CSV ===")
print("=" * 70)

csv_writer = CSVWriter("csv/fruit_decay_eit_simulation_greit")

# 位置エラーデータをCSVに保存
csv_writer.save_position_errors(
    days=days_list,
    contrasts=contrasts_list,
    results=position_errors_dict,
    resistance_data=resistance_data,
)

print("=" * 70)

# ============================================================================
# ロギング終了
# ============================================================================

finalize_logging(logger)

print("=" * 70)

plt.show()
