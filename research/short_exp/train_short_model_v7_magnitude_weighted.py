"""空売りモデルv7: リターンの大きさで重み付けしたペアワイズ損失(2026-09-24追加、
ユーザー提案)。特徴量(SHORT_STOCK_COLS_V2)・ラベル(risk_adjusted_blend)・EV重み(2.0:1.0)は
v2から一切変えず、ペアワイズ損失の各ペアへの重みだけを|ret_pct差|ベースに変える
(losses_short_magnitude.py、外れ値はmax_weight=3.0でクリップ)。

実行方法:
    python research/short_exp/train_short_model_v7_magnitude_weighted.py
"""
import sys
import os

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

from train_short_model_v2 import SHORT_STOCK_COLS_V2

SHORT_MODEL_CACHE_DIR_V7 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cache", "short_model_cache_v7_magnitude"
)


def main():
    import torch
    import stage3_exp.experiment_context as m
    from short_exp.data_builder_short import build_dataset_short
    from short_exp.trainer_short_magnitude import train_model_short_magnitude

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄")
    print(f"[*] 特徴量: stock={SHORT_STOCK_COLS_V2}")
    print("[*] 損失: |ret_pct差|重み付けペアワイズ(max_weight=3.0)、EV重み2.0:1.0(v2と同一)")

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df, force_rebuild=False)
    margin_pool = m.get_margin_pool_df(tickers, macro_df, force_rebuild=False)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df, force_rebuild=False)
    macro_pool_df = m.augment_macro_pool_df(
        m.get_full_macro_pool_df(tickers, force_rebuild=False)
    )

    _base_per_ticker = m.build_per_ticker(
        pool, margin_pool, sector_pool, macro_pool_df, SHORT_STOCK_COLS_V2
    )
    _base_valid_dates = m.valid_cross_section_dates(
        _base_per_ticker, SHORT_STOCK_COLS_V2, m.MIN_CROSS_SECTION
    )
    shared_split_date = m.compute_global_split_date(_base_valid_dates, None)
    print(f"[*] split date: {shared_split_date.date()}")

    tr0, va0 = build_dataset_short(
        m.BASELINE_MACRO_COLS, SHORT_STOCK_COLS_V2, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, global_split_date_override=shared_split_date,
    )
    print(f"[+] train={len(tr0[2])} val={len(va0[2])}")

    os.makedirs(SHORT_MODEL_CACHE_DIR_V7, exist_ok=True)
    for seed in m.SEEDS:
        ckpt_path = os.path.join(SHORT_MODEL_CACHE_DIR_V7, f"state_{seed}.pt")
        if os.path.exists(ckpt_path):
            print(f"[*] seed={seed}は既にキャッシュ済み(途中終了からの再開用): {ckpt_path}")
            continue
        model = train_model_short_magnitude(
            seed, tr0, va0, SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS,
            label=f"short_v7_magnitude seed={seed}", max_weight=3.0,
        )
        torch.save(model.state_dict(), ckpt_path)
        del model
        torch.cuda.empty_cache()
    print(f"[+] チェックポイントを保存しました: {SHORT_MODEL_CACHE_DIR_V7}")


if __name__ == "__main__":
    main()
