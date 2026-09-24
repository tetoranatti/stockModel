"""空売りモデルv2の貸株コスト込みバックテスト(2026-09-24、ユーザー提案「貸株コストの検証」)。
[[short_selling_model_2026-09-24]]で報告したPF 1.51等はコスト無し(価格差のみ)の数字——
このリポジトリにはロング/ショートいずれの側にも取引コスト(手数料・スリッページ・貸株料)を
差し引く仕組みがそもそも存在しない([[borrow_cost_backtest_2026-09-24]]参照)。かつ実際の
銘柄別貸株料・逆日歩データもキャッシュされていない(margin_history_long.csv由来の
信用倍率は需給プロキシであって円建てコストではない)ため、モデル化するなら想定値を
置くしかない。

ユーザーとの合意(2026-09-24): 複数レート感応度分析ではなく、フラット年率のみを採用。
一般信用の標準的な貸株料水準(年率1.1%程度)を保有日数(暦日、entry_date〜exit_dateの
実日数)で按分して各トレードのret_pctから差し引く。逆日歩(制度信用の品貸料スパイク)は
実データが無いためモデル化しない——本スクリプトの結果は「一般信用で借りられる前提の
下限コスト」であり、逆日歩が発生する銘柄がある場合はこれより悪化しうる点に注意。

再学習はしない: train_short_model_v2.pyが保存済みのチェックポイントをそのまま読み込み、
バックテスト評価の後処理としてret_pctを調整するだけ(モデルの学習目的関数はコスト無し
のまま——学習ラベルまでコスト込みにする場合は別途検討)。

実行方法:
    python research/short_exp/backtest_short_model_v2_borrow_cost.py
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

# 一般信用の標準的な貸株料水準(年率)。実測データが無いための想定値。
ANNUAL_BORROW_RATE = 0.011


def apply_borrow_cost(d_ens, annual_rate):
    """entry_date〜exit_dateの暦日数で按分した貸株コストをret_pctから差し引いたコピーを返す。"""
    d_cost = d_ens.copy()
    holding_days = (d_cost["exit_date"] - d_cost["entry_date"]).dt.days
    d_cost["ret_pct"] = d_cost["ret_pct"] - annual_rate * holding_days / 365.0
    return d_cost, holding_days


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
    d_cost, holding_days = apply_borrow_cost(d_ens, ANNUAL_BORROW_RATE)
    print(
        f"[*] 貸株コスト: 年率{ANNUAL_BORROW_RATE*100:.2f}% "
        f"(平均保有{holding_days.mean():.1f}暦日 -> 平均コスト{ANNUAL_BORROW_RATE*holding_days.mean()/365*100:.3f}%/トレード)"
    )

    print("\n" + "=" * 90)
    print("【日次Top-5運用評価: コスト無し vs 貸株コスト込み】")
    print("=" * 90)
    r_no_cost = m.evaluate_daily_topn(
        d_ens, "コスト無し", n_per_day=m.DAILY_TOPN,
        min_reliable_n=m.MIN_RELIABLE_N, max_concurrent=m.MAX_CONCURRENT_POSITIONS,
        close_price_lookup=close_price_lookup,
    )
    r_cost = m.evaluate_daily_topn(
        d_cost, f"貸株コスト込み(年率{ANNUAL_BORROW_RATE*100:.1f}%)", n_per_day=m.DAILY_TOPN,
        min_reliable_n=m.MIN_RELIABLE_N, max_concurrent=m.MAX_CONCURRENT_POSITIONS,
        close_price_lookup=close_price_lookup,
    )

    print("\n" + "=" * 90)
    print("【TOPN別PF分析: コスト無し】")
    print("=" * 90)
    curve_df = m.evaluate_topn_curve(
        d_ens, topn_list=[1, 3, 5, 10, 20, 30], close_price_lookup=close_price_lookup,
    )
    print(curve_df.to_string(index=False))

    print("\n" + "=" * 90)
    print(f"【TOPN別PF分析: 貸株コスト込み(年率{ANNUAL_BORROW_RATE*100:.1f}%)】")
    print("=" * 90)
    curve_cost_df = m.evaluate_topn_curve(
        d_cost, topn_list=[1, 3, 5, 10, 20, 30], close_price_lookup=close_price_lookup,
    )
    print(curve_cost_df.to_string(index=False))

    if m.TICKER_TO_CLUSTER:
        print("\n" + "=" * 90)
        print(
            f"【TOPN別PF分析: 貸株コスト込み、クラスタ制約 max_per_cluster={m.MAX_PER_CLUSTER}】"
        )
        print("=" * 90)
        curve_cluster_cost_df = m.evaluate_topn_curve(
            d_cost, topn_list=[1, 3, 5, 10, 20, 30], max_per_cluster=m.MAX_PER_CLUSTER,
            close_price_lookup=close_price_lookup,
        )
        print(curve_cluster_cost_df.to_string(index=False))

    del models
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
