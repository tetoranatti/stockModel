"""空売りモデルv5(EV_WIN_WEIGHT:EV_STOP_WEIGHT非対称性)のバックテスト(2026-09-24追加)。
run_backtest_inference_ensembleが返すp_win/p_stop列(生確率、重み無し)から、学習時と
同じ重みでscore列を再計算してからevaluate_daily_topn/evaluate_topn_curveに渡す
——run_backtest_inference_ensemble自体のscore列はmodules/model_inference.pyが
ハードコードする2:1のままなので、そのままでは使えない点に注意。

実行方法:
    python research/short_exp/backtest_short_model_v5_asym_weight.py --win-weight 1.0 --stop-weight 1.0
    python research/short_exp/backtest_short_model_v5_asym_weight.py --win-weight 1.0 --stop-weight 2.0
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
from train_short_model_v5_asym_weight import cache_dir_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--win-weight", type=float, required=True)
    parser.add_argument("--stop-weight", type=float, required=True)
    args = parser.parse_args()

    import torch
    import stage3_exp.experiment_context as m
    from short_exp.data_builder_short import prepare_backtest_pool_short

    cache_dir = cache_dir_for(args.win_weight, args.stop_weight)

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

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
    print(f"[*] EV重み: win={args.win_weight} stop={args.stop_weight}")

    close_price_lookup = {}
    pool_test = prepare_backtest_pool_short(
        m.BASELINE_MACRO_COLS, SHORT_STOCK_COLS_V2, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, global_split_date_override=shared_split_date,
        close_price_out=close_price_lookup,
    )
    print(f"[+] backtest_pool(空売りv5)={len(pool_test)}")

    models = []
    for seed in m.SEEDS:
        ckpt = os.path.join(cache_dir, f"state_{seed}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"{ckpt}が見つかりません。先にtrain_short_model_v5_asym_weight.pyを"
                f"同じ--win-weight/--stop-weightで実行してください。"
            )
        model = m.build_model(SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS)
        model.load_state_dict(torch.load(ckpt, map_location=m.DEVICE))
        model.eval()
        models.append(model)

    d_ens = m.run_backtest_inference_ensemble(models, pool_test)
    # run_backtest_inference_ensembleのscore列はmodules/model_inference.pyがハードコードする
    # 2:1のまま(学習時の重みを反映しない)ため、この実験専用の重みで再計算する。
    d_ens["score"] = args.win_weight * d_ens["p_win"] - args.stop_weight * d_ens["p_stop"]

    r = m.evaluate_daily_topn(
        d_ens, f"空売りモデルv5(win={args.win_weight}:stop={args.stop_weight}、アンサンブル日次Top-5)",
        n_per_day=m.DAILY_TOPN, min_reliable_n=m.MIN_RELIABLE_N,
        max_concurrent=m.MAX_CONCURRENT_POSITIONS, close_price_lookup=close_price_lookup,
    )

    print("\n" + "=" * 90)
    print(f"【TOPN別PF分析(空売りモデルv5 win={args.win_weight}:stop={args.stop_weight}、制約無し)】")
    print("=" * 90)
    curve_df = m.evaluate_topn_curve(
        d_ens, topn_list=[1, 3, 5, 10, 20, 30], close_price_lookup=close_price_lookup,
    )
    print(curve_df.to_string(index=False))

    del models
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
