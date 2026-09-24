"""空売りモデルv5: EV_WIN_WEIGHT:EV_STOP_WEIGHT非対称性の空売り専用再検証
(2026-09-24追加、ユーザー提案)。特徴量(SHORT_STOCK_COLS_V2)・ラベル(risk_adjusted_blend、
v2で最良と確認済み)は一切変えず、損失内のスコア重み(p_win:p_stop)だけを変えて
NearPairRankingLossShortAsym(losses_short_asym.py)+trainer_short_asym.pyで学習する。
共有stage3_exp/config.pyのEV_WIN_WEIGHT/EV_STOP_WEIGHT(ロング本番にも使われる2.0:1.0)は
変更しない。

バックテスト側のscore列も学習時と同じ重みで再計算する必要がある——
run_backtest_inference_ensembleが返すp_win/p_stop列(modules/model_inference.pyの
predict_probabilities_ensemble_batchはEV重みをハードコード2:1しているため、そのままの
score列は使えない)から、この実験用の重みで score = win_weight*p_win - stop_weight*p_stop
を再計算してevaluate_daily_topn/evaluate_topn_curveに渡す(backtest_short_model_v5_*.pyで実施)。

trainer_short_asymはparallel_seed_sweep.pyのworkerディスパッチ(train_short_model_v2.py等が
使うtrain_configs_parallel)に接続していない(あちらはm.train_model固定呼び出し)ため、
trainer_alpha.pyと同じく5seedを1プロセス内で逐次学習する。

実行方法:
    python research/short_exp/train_short_model_v5_asym_weight.py --win-weight 1.0 --stop-weight 1.0
    python research/short_exp/train_short_model_v5_asym_weight.py --win-weight 1.0 --stop-weight 2.0
"""
import sys
import os
import argparse

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

from train_short_model_v2 import SHORT_STOCK_COLS_V2


def cache_dir_for(win_weight, stop_weight):
    tag = f"w{win_weight:g}_s{stop_weight:g}".replace(".", "p")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "cache", f"short_model_cache_v5_{tag}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--win-weight", type=float, required=True)
    parser.add_argument("--stop-weight", type=float, required=True)
    args = parser.parse_args()

    import torch
    import stage3_exp.experiment_context as m
    from short_exp.data_builder_short import build_dataset_short
    from short_exp.trainer_short_asym import train_model_short_asym

    cache_dir = cache_dir_for(args.win_weight, args.stop_weight)

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄")
    print(f"[*] 特徴量: stock={SHORT_STOCK_COLS_V2}")
    print(f"[*] EV重み: win={args.win_weight} stop={args.stop_weight} (既定は2.0:1.0)")

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

    os.makedirs(cache_dir, exist_ok=True)
    for seed in m.SEEDS:
        ckpt_path = os.path.join(cache_dir, f"state_{seed}.pt")
        if os.path.exists(ckpt_path):
            print(f"[*] seed={seed}は既にキャッシュ済み(途中終了からの再開用): {ckpt_path}")
            continue
        model = train_model_short_asym(
            seed, tr0, va0, SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS,
            label=f"short_v5(w{args.win_weight}:s{args.stop_weight}) seed={seed}",
            ev_win_weight=args.win_weight, ev_stop_weight=args.stop_weight,
        )
        torch.save(model.state_dict(), ckpt_path)
        del model
        torch.cuda.empty_cache()
    print(f"[+] チェックポイントを保存しました: {cache_dir}")


if __name__ == "__main__":
    main()
