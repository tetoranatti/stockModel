"""空売りランキングモデル v2 のバックテスト(2026-09-24追加)。train_short_model_v2.pyが
保存したチェックポイント(9 stock + 5 macro、dist_from_high20除外)を読み込むだけで、
再学習せずに何度でも評価方法を変えて試せる。v1(10+5、ロングから流用しただけ)との比較用に
両方の制約無し/クラスタ制約TopN分析を出す。

実行方法:
    python research/short_exp/backtest_short_model_v2.py
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

from train_short_model_v2 import SHORT_STOCK_COLS_V2, SHORT_MODEL_CACHE_DIR


def main():
    import torch
    import stage3_exp.experiment_context as m
    from short_exp.data_builder_short import prepare_backtest_pool_short

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

    close_price_lookup = {}
    pool_test = prepare_backtest_pool_short(
        m.BASELINE_MACRO_COLS, SHORT_STOCK_COLS_V2, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, global_split_date_override=shared_split_date,
        close_price_out=close_price_lookup,
    )
    print(f"[+] backtest_pool(空売りv2)={len(pool_test)}")

    models = []
    for seed in m.SEEDS:
        ckpt = os.path.join(SHORT_MODEL_CACHE_DIR, f"state_{seed}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"{ckpt}が見つかりません。先にtrain_short_model_v2.pyを実行してください。"
            )
        model = m.build_model(SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS)
        model.load_state_dict(torch.load(ckpt, map_location=m.DEVICE))
        model.eval()
        models.append(model)

    d_ens = m.run_backtest_inference_ensemble(models, pool_test)

    r = m.evaluate_daily_topn(
        d_ens, "空売りモデルv2(アンサンブル日次Top-5)", n_per_day=m.DAILY_TOPN,
        min_reliable_n=m.MIN_RELIABLE_N, max_concurrent=m.MAX_CONCURRENT_POSITIONS,
        close_price_lookup=close_price_lookup,
    )

    print("\n" + "=" * 90)
    print("【TOPN別PF分析(空売りモデルv2、制約無し)】")
    print("=" * 90)
    curve_df = m.evaluate_topn_curve(
        d_ens, topn_list=[1, 3, 5, 10, 20, 30], close_price_lookup=close_price_lookup,
    )
    print(curve_df.to_string(index=False))

    if m.TICKER_TO_CLUSTER:
        print("\n" + "=" * 90)
        print(f"【TOPN別PF分析(空売りモデルv2、クラスタ制約 max_per_cluster={m.MAX_PER_CLUSTER})】")
        print("=" * 90)
        curve_cluster_df = m.evaluate_topn_curve(
            d_ens, topn_list=[1, 3, 5, 10, 20, 30], max_per_cluster=m.MAX_PER_CLUSTER,
            close_price_lookup=close_price_lookup,
        )
        print(curve_cluster_df.to_string(index=False))

    del models
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
