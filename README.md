## Day5プログラムの使い方

このプログラムでは、GREITとJAC法を比較することができます。以下に実行方法を示します：

```bash
python compare_greit_jac_day5.py --days 4 5 6 --noise 0.01
```

### オプション
- `--days`： 比較するDay番号を指定します（例: `--days 3 5 8`）。
- `--noise`： ノイズレベルを指定します（デフォルト: `0.01` (1%)）。
- `--scale-mode`： カラースケールモードを選択できます（`unified` または `individual`）。