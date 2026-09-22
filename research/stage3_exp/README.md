# stage3_exp

Stage3 Feature Toggle Experiment の分割版。

元ファイル:

```text
research/stage3_toggle_experiment.py
```

巨大化したため機能別に分割。

---

# エントリポイント

## 通常実行

```bash
python stage3_toggle_experiment.py
```

実行フロー:

```text
stage3_toggle_experiment.py
          │
          ▼
stage3_exp.experiment_runner
```

---

## Seed並列実行

```bash
python parallel_seed_sweep.py
```

実行フロー:

```text
parallel_seed_sweep.py
          │
          ▼
stage3_exp.experiment_context
          │
          ▼
各モジュールを再エクスポート
```

experiment_context.py は
parallel_seed_sweep 用の互換レイヤー。

---

# ファイル構成

```text
stage3_exp/

├─ __init__.py
├─ config.py
├─ runtime.py
│
├─ feature_catalog.py
├─ targets.py
├─ data_builder.py
│
├─ model_factory.py
├─ losses.py
├─ trainer.py
│
├─ inference.py
├─ diagnostics.py
│
├─ cache.py
│
├─ experiment_context.py
└─ experiment_runner.py
```

---

# 依存関係

```text
config
 │
 ├─ runtime
 │
 └─ feature_catalog
       │
       ├─ targets
       │
       ├─ data_builder
       │
       ├─ model_factory
       │
       └─ cache
               │
               ▼
            trainer
               │
               ▼
           inference
               │
               ▼
          diagnostics
               │
               ▼
       experiment_runner
```

原則:

- config は最上位
- feature_catalog は特徴量定義のみ
- experiment_runner が全体統括

循環参照は禁止

---

# 各モジュール

## config.py

実験設定。

主な内容:

- FEATURE_TOGGLES
- MODEL_ARCH
- LABEL_MODE
- BATCH_SIZE
- バリア設定
- 学習設定

新しい実験設定はここへ追加。

---

## runtime.py

共通ユーティリティ。

主な内容:

```python
DEVICE
set_seed()
log()
elapsed()
```

---

## feature_catalog.py

特徴量管理。

主な内容:

```python
FEATURE_GROUPS
FEATURE_CATALOG

BASELINE_STOCK_COLS
BASELINE_MACRO_COLS

TEST_STOCK_COLS
TEST_MACRO_COLS

STOCK_COMPUTE_FNS
MACRO_COMPUTE_FNS
```

特徴量追加時はまずここを更新する。

---

## targets.py

ラベル生成関連。

主な関数:

```python
compute_simulated_targets()
compute_ret5_fixed_horizon_targets()
simulate_ret_pct_d1_close()
```

---

## data_builder.py

学習データ作成。

主な関数:

```python
prepare_ticker_data()
build_dataset()
prepare_backtest_pool()
```

最も重要な前処理モジュール。

---

## model_factory.py

モデル生成。

主な内容:

```python
ARCH_CLASSES
build_model()
```

アーキテクチャ追加時はここを更新。

---

## losses.py

ランキング損失。

主なクラス:

```python
NearPairRankingLossFast
NearPairRankingLossRegimeAware
```

---

## trainer.py

学習処理。

主な関数:

```python
train_model()
```

DataLoader
Optimizer
Training Loop

を管理。

---

## inference.py

推論処理。

主な関数:

```python
predict1()
run_backtest_inference()
run_backtest_inference_ensemble()
```

---

## diagnostics.py

評価・分析。

主な関数:

```python
evaluate_topn_curve()
monthly_topn_pf_matrix()
evaluate_daily_topn_overlap()
```

---

## cache.py

ベースラインキャッシュ管理。

主な関数:

```python
baseline_cache_fingerprint()
```

---

## experiment_runner.py

実験の統括。

役割:

```text
データ準備
 ↓
学習
 ↓
推論
 ↓
評価
```

元の

```python
if __name__ == "__main__":
```

で行っていた処理を保持。

---

## experiment_context.py

parallel_seed_sweep 専用。

```python
import stage3_exp.experiment_context as m
```

を可能にするための再エクスポート層。

実装ロジックは追加しない。

---

# トラブルシューティング

## KeyError: stock_ret_1d

確認:

```python
FEATURE_TOGGLES
FEATURE_CATALOG
```

のキーが一致しているか。

---

## ModuleNotFoundError: modules

確認:

```text
F:\stockModel
```

が sys.path に追加されているか。

---

## ModuleNotFoundError: training

確認:

```text
F:\stockModel
```

が worker プロセスにも追加されているか。

---

# 編集ルール

|変更内容|編集先|
|---------|---------|
|特徴量追加|feature_catalog.py|
|特徴量計算追加|feature_catalog.py|
|ラベル変更|targets.py|
|モデル変更|model_factory.py|
|損失変更|losses.py|
|学習変更|trainer.py|
|推論変更|inference.py|
|評価変更|diagnostics.py|
|設定変更|config.py|

experiment_runner.py はできるだけ薄く保つ。
