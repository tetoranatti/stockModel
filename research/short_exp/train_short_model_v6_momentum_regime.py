"""空売りモデルv6: 市場モメンタムレジーム別ペアワイズ損失(2026-09-24追加、ユーザー提案)。
特徴量(SHORT_STOCK_COLS_V2)・ラベル(risk_adjusted_blend)・EV重み(2.0:1.0、v5検証で
最良と確認済み)はv2から一切変えず、ペア比較を同一モメンタムレジーム(regime_momentum.py、
20日NK225リターンの拡大窓3分位)内に限定する制約だけを追加する。共有config.pyの
USE_REGIME_AWARE_LOSSはFalseのまま変更しない(trainer_short_asym.pyのuse_regime_aware
引数でこのスクリプトだけ局所的にTrueにする)。

動機: [[walk_forward_validation_2026-09-24]]で判明した「市場の上昇モメンタムが強い
局面ほど空売りPFが悪化する」という空売り特有の地合い依存性を、レジームを跨いだペア比較
から生じるノイズとして緩和できないか検証する。

実行方法:
    python research/short_exp/train_short_model_v6_momentum_regime.py
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

SHORT_MODEL_CACHE_DIR_V6 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cache", "short_model_cache_v6_momentum_regime"
)


def main():
    import torch
    import stage3_exp.experiment_context as m
    from short_exp.data_builder_short import build_dataset_short
    from short_exp.trainer_short_asym import train_model_short_asym
    from short_exp.regime_momentum import compute_expanding_momentum_regime_labels

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄")
    print(f"[*] 特徴量: stock={SHORT_STOCK_COLS_V2}")
    print("[*] レジーム別ペアワイズ損失: モメンタムレジーム(LOW/MID/HIGH)、EV重み2.0:1.0(v2と同一)")

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df, force_rebuild=False)
    margin_pool = m.get_margin_pool_df(tickers, macro_df, force_rebuild=False)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df, force_rebuild=False)
    macro_pool_df = m.augment_macro_pool_df(
        m.get_full_macro_pool_df(tickers, force_rebuild=False)
    )

    momentum_regime_labels = compute_expanding_momentum_regime_labels(macro_pool_df)
    print(f"[*] モメンタムレジーム分布(全期間): "
          f"{momentum_regime_labels.value_counts().to_dict()}")

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
        macro_pool_df, regime_filter=None, regime_labels_for_loss=momentum_regime_labels,
        global_split_date_override=shared_split_date,
    )
    print(f"[+] train={len(tr0[2])} val={len(va0[2])}")

    os.makedirs(SHORT_MODEL_CACHE_DIR_V6, exist_ok=True)
    for seed in m.SEEDS:
        ckpt_path = os.path.join(SHORT_MODEL_CACHE_DIR_V6, f"state_{seed}.pt")
        if os.path.exists(ckpt_path):
            print(f"[*] seed={seed}は既にキャッシュ済み(途中終了からの再開用): {ckpt_path}")
            continue
        model = train_model_short_asym(
            seed, tr0, va0, SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS,
            label=f"short_v6_momentum_regime seed={seed}",
            ev_win_weight=2.0, ev_stop_weight=1.0, use_regime_aware=True,
        )
        torch.save(model.state_dict(), ckpt_path)
        del model
        torch.cuda.empty_cache()
    print(f"[+] チェックポイントを保存しました: {SHORT_MODEL_CACHE_DIR_V6}")


if __name__ == "__main__":
    main()
