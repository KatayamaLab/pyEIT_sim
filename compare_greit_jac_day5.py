# coding: utf-8
"""
GREIT vs JAC 比較: 可変Day比較 (1.0% ノイズ)
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
import argparse

# ============================================================================
# コマンドライン引数の解析
# ============================================================================


def parse_arguments():
    parser = argparse.ArgumentParser(description="GREIT vs JAC 比較ツール")
    parser.add_argument(
        "--days",
        type=int,
        nargs="+",
        default=[4, 5, 6],
        help="比較するDay番号 (例: --days 3 5 8)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="ノイズレベル (デフォルト: 0.01 = 1.0%%)",
    )
    parser.add_argument(
        "--scale-mode",
        type=str,
        choices=["unified", "individual"],
        default="unified",
        help="カラースケールモード: unified=各行でJACとGREITを共通化(--scale-day指定時は全図統一), individual=各プロット個別",
    )
    parser.add_argument(
        "--scale-day",
        type=int,
        default=None,
        help="unified モード時に全図で統一する基準Day番号 (未指定時は各行でそのDayのJAC/GREITを共通化)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="乱数シード (デフォルト: 42、再現性のため固定)",
    )
    parser.add_argument(
        "--no-fixed-seed",
        action="store_true",
        help="乱数シードを固定しない（実行ごとに異なるノイズ）",
    )
    return parser.parse_args()


args = parse_arguments()

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
target_days = sorted(args.days)  # コマンドライン引数から取得
noise_level = args.noise  # コマンドライン引数から取得
scale_mode = args.scale_mode

# Dayの妥当性チェック
for day in target_days:
    if day not in resistance_data:
        raise ValueError(f"Day {day} is not defined in resistance_data")

print(f"比較対象: Days {target_days}")
print(f"ノイズレベル: {noise_level * 100:.1f}%")
print(f"スケールモード: {scale_mode}")
if args.no_fixed_seed:
    print(f"乱数シード: 固定しない（実行ごとに異なるノイズ）")
else:
    print(f"乱数シード: {args.random_seed}（固定）")
print()

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
    if args.no_fixed_seed:
        # シードを固定しない（毎回異なるノイズ）
        pass  # np.random.seedを呼ばない
    else:
        # シードを固定（再現性のため）
        np.random.seed(args.random_seed + target_day)
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
# カラースケール範囲の決定
# ============================================================================

if scale_mode == "unified":
    if args.scale_day is not None:
        # Day指定あり: 全ての図で指定したDayのスケールを使う
        reference_day = args.scale_day
        if reference_day not in target_days:
            raise ValueError(
                f"指定されたscale_day={reference_day}が比較対象に含まれていません"
            )

        ds_greit_ref = results[reference_day]["ds_greit_real"]
        ds_jac_ref = results[reference_day]["ds_jac_n"]

        # 両方を含む統一スケール
        vmin_unified = min(np.nanmin(ds_greit_ref), np.min(ds_jac_ref))
        vmax_unified = max(np.nanmax(ds_greit_ref), np.max(ds_jac_ref))

        print(
            f"Unified scale mode: Using Day {reference_day} as reference for ALL plots"
        )
        print(f"Unified scale: vmin={vmin_unified:.4f}, vmax={vmax_unified:.4f}\n")
    else:
        # Day指定なし: 各行でそのDayのJACとGREITが同じスケール（行ごとに異なる）
        print(f"Unified scale mode: Each row uses same scale for its JAC and GREIT\n")
else:
    print(f"Individual scale mode: Each subplot uses its own scale\n")

# ============================================================================
# プロット: 動的な行数
# ============================================================================

num_rows = len(target_days)
fig, axes = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows))

scale_mode_str = f"Scale: {scale_mode.capitalize()}"
if scale_mode == "unified":
    if args.scale_day is not None:
        scale_mode_str += f" (All from Day {reference_day})"
    else:
        scale_mode_str += f" (Row-wise)"

fig.suptitle(
    f"GREIT vs JAC Comparison (Days {', '.join(map(str, target_days))} | Noise: {noise_level * 100:.1f}% | {scale_mode_str})",
    fontsize=24,
    y=0.995,
)

# 1行のみの場合はaxesを2次元配列に変換
if num_rows == 1:
    axes = axes.reshape(1, -1)

for idx, target_day in enumerate(target_days):
    result = results[target_day]
    contrast = result["contrast"]
    ds_jac_n = result["ds_jac_n"]
    ds_greit_real = result["ds_greit_real"]

    # カラースケール範囲を決定
    if scale_mode == "unified":
        if args.scale_day is not None:
            # Day指定あり: 全図で同じスケール
            vmin_plot = vmin_unified
            vmax_plot = vmax_unified
        else:
            # Day指定なし: この行でJACとGREITが同じスケール
            vmin_plot = min(np.min(ds_jac_n), np.nanmin(ds_greit_real))
            vmax_plot = max(np.max(ds_jac_n), np.nanmax(ds_greit_real))
    else:  # individual
        # 各プロットで個別のスケール
        vmin_jac = np.min(ds_jac_n)
        vmax_jac = np.max(ds_jac_n)
        vmin_greit = np.nanmin(ds_greit_real)
        vmax_greit = np.nanmax(ds_greit_real)

    # ----- JAC (左列) -----
    ax_jac = axes[idx, 0]

    if scale_mode == "individual":
        im_jac = ax_jac.tripcolor(
            x,
            y,
            tri,
            ds_jac_n,
            shading="flat",
            cmap="jet",
            vmin=vmin_jac,
            vmax=vmax_jac,
        )
    else:  # unified
        im_jac = ax_jac.tripcolor(
            x,
            y,
            tri,
            ds_jac_n,
            shading="flat",
            cmap="jet",
            vmin=vmin_plot,
            vmax=vmax_plot,
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

    if scale_mode == "individual":
        im_greit = ax_greit.imshow(
            ds_greit_real,
            interpolation="none",
            cmap="jet",
            origin="lower",
            extent=[-1, 1, -1, 1],
            vmin=vmin_greit,
            vmax=vmax_greit,
        )
    else:  # unified
        im_greit = ax_greit.imshow(
            ds_greit_real,
            interpolation="none",
            cmap="jet",
            origin="lower",
            extent=[-1, 1, -1, 1],
            vmin=vmin_plot,
            vmax=vmax_plot,
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

# ファイル名を動的に生成
days_str = "_".join(map(str, target_days))
scale_suffix = f"_{scale_mode}"
if scale_mode == "unified" and args.scale_day is not None:
    scale_suffix += f"_day{args.scale_day}"
output_filename = f"img/comparison_greit_vs_jac_days{days_str}_noise{noise_level * 100:.1f}{scale_suffix}.png"
plt.savefig(output_filename, dpi=150, bbox_inches="tight")
print(f"Saved: {output_filename}")

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
csv_filename = f"csv/compare_greit_jac_days{days_str}_stats_{timestamp}.csv"

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
            "Scale_Mode",
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
                scale_mode,
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
                scale_mode,
            ]
        )

print(f"CSV saved: {csv_filename}")
print("=" * 60)

# ============================================================================
# ロギング終了
# ============================================================================

finalize_logging(logger)

plt.show()
