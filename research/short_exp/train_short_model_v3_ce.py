"""空売りモデルv3学習: 決済理由ラベル(exit_class)+交差エントロピー損失
(2026-09-24追加、ユーザー提案「TP/SL到達順序型ラベル」)。v2(9+5特徴量、
risk_adjusted_blendラベル+NearPairRankingLossRegimeAwareのペアワイズ順位学習)から
特徴量集合は変えず、ラベル・損失関数だけを差し替えて比較する
(train_short_model_v2.py参照、SHORT_STOCK_COLS_V2をそのまま流用)。

build_dataset_short(use_exit_class_labels=True)がconfig.pyのLABEL_MODEを無視して
targets_short.compute_simulated_short_targetsのexit_class(0=Stop/1=Hold/2=Win)を
教師ラベルにする。trainer_ce.train_model_ceが標準的な交差エントロピー(クラス不均衡補正
weight付き)でサンプル単位学習する——並列学習infra(parallel_seed_sweep.py/
greedy_feature_search.py)はNearPairRankingLoss前提のため使わず、trainer_alpha.pyと
同じく5seedを1プロセス内で逐次学習する(train_alpha_short_model.py参照、途中終了からの
再開用にseedごとキャッシュ済みならスキップする)。

実行方法:
    python research/short_exp/train_short_model_v3_ce.py
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

SHORT_MODEL_CACHE_DIR_V3 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cache", "short_model_cache_v3_ce"
)


def main():
    import torch
    import stage3_exp.experiment_context as m
    from short_exp.data_builder_short import build_dataset_short
    from short_exp.trainer_ce import train_model_ce

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄")
    print(f"[*] 特徴量: stock={SHORT_STOCK_COLS_V2}")
    print(f"[*] 特徴量: macro={m.BASELINE_MACRO_COLS}")
    print("[*] ラベル: exit_class(0=Stop/1=Hold/2=Win) 損失: CrossEntropy(クラス重み付き)")

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
        use_exit_class_labels=True,
    )
    print(f"[+] train={len(tr0[2])} val={len(va0[2])}")

    os.makedirs(SHORT_MODEL_CACHE_DIR_V3, exist_ok=True)
    for seed in m.SEEDS:
        ckpt_path = os.path.join(SHORT_MODEL_CACHE_DIR_V3, f"state_{seed}.pt")
        if os.path.exists(ckpt_path):
            print(f"[*] seed={seed}は既にキャッシュ済み(途中終了からの再開用): {ckpt_path}")
            continue
        model = train_model_ce(
            seed, tr0, va0, SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS,
            label=f"short_v3_ce seed={seed}",
        )
        torch.save(model.state_dict(), ckpt_path)
        del model
        torch.cuda.empty_cache()
    print(f"[+] チェックポイントを保存しました: {SHORT_MODEL_CACHE_DIR_V3}")


if __name__ == "__main__":
    main()
