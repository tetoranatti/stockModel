"""空売りランキングモデルのバックテスト(2026-09-24追加)。train_short_model.pyが保存した
チェックポイントを読み込むだけで、再学習せずに何度でも評価方法を変えて試せる。

スコアはロング版と同じev(2*p_win-p_stop)——ここでのp_win/p_stopは「空売りが利確方向に
動いたか/損切り方向に動いたか」の確率(ラベルの符号を空売り損益にした時点で意味が反転
している)。日次スコア上位N件を選ぶ既存のevaluate_daily_topn/evaluate_topn_curveを
そのまま再利用でき、ret_pctが既に空売り損益で符号付けされているためPF計算等の枠組みは
一切変更不要。

実行方法:
    python research/short_exp/backtest_short_model.py
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

SHORT_MODEL_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cache", "short_model_cache"
)


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
        pool, margin_pool, sector_pool, macro_pool_df, m.BASELINE_STOCK_COLS
    )
    _base_valid_dates = m.valid_cross_section_dates(
        _base_per_ticker, m.BASELINE_STOCK_COLS, m.MIN_CROSS_SECTION
    )
    shared_split_date = m.compute_global_split_date(_base_valid_dates, None)
    print(f"[*] split date: {shared_split_date.date()}")

    close_price_lookup = {}
    pool_test = prepare_backtest_pool_short(
        m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, global_split_date_override=shared_split_date,
        close_price_out=close_price_lookup,
    )
    print(f"[+] backtest_pool(空売り)={len(pool_test)}")

    models = []
    for seed in m.SEEDS:
        ckpt = os.path.join(SHORT_MODEL_CACHE_DIR, f"state_{seed}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"{ckpt}が見つかりません。先にtrain_short_model.pyを実行してください。"
            )
        model = m.build_model(m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
        model.load_state_dict(torch.load(ckpt, map_location=m.DEVICE))
        model.eval()
        models.append(model)

    d_ens = m.run_backtest_inference_ensemble(models, pool_test)

    r = m.evaluate_daily_topn(
        d_ens, "空売りモデル(アンサンブル日次Top-5)", n_per_day=m.DAILY_TOPN,
        min_reliable_n=m.MIN_RELIABLE_N, max_concurrent=m.MAX_CONCURRENT_POSITIONS,
        close_price_lookup=close_price_lookup,
    )

    print("\n" + "=" * 90)
    print("【TOPN別PF分析(空売りモデル、制約無し)】")
    print("=" * 90)
    curve_df = m.evaluate_topn_curve(
        d_ens, topn_list=[1, 3, 5, 10, 20, 30], close_price_lookup=close_price_lookup,
    )
    print(curve_df.to_string(index=False))

    if m.TICKER_TO_CLUSTER:
        print("\n" + "=" * 90)
        print(f"【TOPN別PF分析(空売りモデル、クラスタ制約 max_per_cluster={m.MAX_PER_CLUSTER})】")
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
