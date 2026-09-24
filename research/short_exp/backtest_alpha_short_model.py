"""Alpha回帰・weighted Huber版の空売りモデルのバックテスト(2026-09-24追加)。
train_alpha_short_model.pyが保存したチェックポイントを読み込むだけで、再学習せずに
何度でも評価方法を変えて試せる。

バックテストの約定プール(w_s/w_m/実際のret_pct)はdata_builder_short.
prepare_backtest_pool_short(バリア・トレーリングストップ込みの空売り損益)をそのまま
再利用する——学習ターゲット(alpha_5d、固定期間)と実際の約定シミュレーション
(バリア付き、より現実的)を意図的に分けているため。スコアはinference_alpha.
run_backtest_inference_ensemble_alphaが返すscore=-predicted_alpha(予測alphaが小さい
=マイナスが大きいほど高スコア)で、既存のevaluate_topn_curve/evaluate_daily_topnを
無変更でそのまま使える。

実行方法:
    python research/short_exp/backtest_alpha_short_model.py
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

ALPHA_MODEL_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cache", "short_alpha_model_cache"
)


def main():
    import torch
    import stage3_exp.experiment_context as m
    from short_exp.data_builder_short import prepare_backtest_pool_short
    from short_exp.model_alpha import build_alpha_model
    from short_exp.inference_alpha import run_backtest_inference_ensemble_alpha

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
    print(f"[+] backtest_pool(空売り、バリア付き約定)={len(pool_test)}")

    models = []
    for seed in m.SEEDS:
        ckpt = os.path.join(ALPHA_MODEL_CACHE_DIR, f"state_{seed}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"{ckpt}が見つかりません。先にtrain_alpha_short_model.pyを実行してください。"
            )
        model = build_alpha_model(m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
        model.load_state_dict(torch.load(ckpt, map_location=m.DEVICE))
        model.eval()
        models.append(model)

    d_ens = run_backtest_inference_ensemble_alpha(models, pool_test)
    print(f"[*] predicted alpha: mean={d_ens['pred_alpha'].mean():.4f} "
          f"std={d_ens['pred_alpha'].std():.4f} min={d_ens['pred_alpha'].min():.4f} "
          f"max={d_ens['pred_alpha'].max():.4f}")

    r = m.evaluate_daily_topn(
        d_ens, "Alpha回帰空売りモデル(アンサンブル日次Top-5)", n_per_day=m.DAILY_TOPN,
        min_reliable_n=m.MIN_RELIABLE_N, max_concurrent=m.MAX_CONCURRENT_POSITIONS,
        close_price_lookup=close_price_lookup,
    )

    print("\n" + "=" * 90)
    print("【TOPN別PF分析(Alpha回帰空売りモデル、制約無し)】")
    print("=" * 90)
    curve_df = m.evaluate_topn_curve(
        d_ens, topn_list=[1, 3, 5, 10, 20, 30], close_price_lookup=close_price_lookup,
    )
    print(curve_df.to_string(index=False))

    if m.TICKER_TO_CLUSTER:
        print("\n" + "=" * 90)
        print(f"【TOPN別PF分析(Alpha回帰空売りモデル、クラスタ制約 max_per_cluster={m.MAX_PER_CLUSTER})】")
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
