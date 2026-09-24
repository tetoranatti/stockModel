"""val期間とバックテスト期間が同じ25%区間になっている問題(2026-09-23、ユーザー指摘)が
成績にどの程度影響するかを実測する。

現行方式(手法A): train=先頭75%、val=残り25%全部、backtest=残り25%全部(val とbacktest
が同じ区間 = checkpoint選択がbacktest区間の情報を(僅かに)使ってしまっている)。

手法B(リーク排除): train=先頭75%(Aと同じ)、val=75%〜87.5%点のみ、backtest=87.5%〜
100%点のみ(=training/checkpoint選択が一切見ない、真にブラインドな区間)。

同じ87.5%〜100%区間で手法Aのモデル(d_ens_A_subset、Aの評価結果をこの区間だけに絞る)と
手法Bのモデル(d_ens_B)を比較すれば、「backtest区間の情報がcheckpoint選択に漏れている
ことによる過大評価」の大きさを直接測れる(同じ評価区間・同じ特徴量・同じtrain区間で、
val_end_date_overrideの有無だけが違う)。

注意: 87.5%〜100%は全体の12.5%しかないため、手法Bのbacktest件数は通常の半分程度になり
ノイズが大きくなる(feedback_ml_validation_methodologyの「nを必ず確認」原則に従い、
出力にtrades件数を必ず表示する)。

実行方法:
    python research/stage3_exp/measure_val_leakage.py
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass


def main():
    import stage3_exp.experiment_context as m
    import parallel_seed_sweep as pss

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄 ({m.UNIVERSE_PATH})")

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df, force_rebuild=False)
    margin_pool = m.get_margin_pool_df(tickers, macro_df, force_rebuild=False)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df, force_rebuild=False)
    macro_pool_df = m.augment_macro_pool_df(
        m.get_full_macro_pool_df(tickers, force_rebuild=False)
    )
    loss_regime_labels = (
        m.compute_regime_labels_expanding(macro_pool_df) if m.USE_REGIME_AWARE_LOSS else None
    )

    _base_per_ticker = m.build_per_ticker(
        pool, margin_pool, sector_pool, macro_pool_df, m.BASELINE_STOCK_COLS
    )
    _base_valid_dates = m.valid_cross_section_dates(
        _base_per_ticker, m.BASELINE_STOCK_COLS, m.MIN_CROSS_SECTION
    )
    split_75 = m.compute_global_split_date(_base_valid_dates, None, percentile=0.75)
    split_875 = m.compute_global_split_date(_base_valid_dates, None, percentile=0.875)
    print(f"[*] split_75(train/val境界)={split_75.date()}  split_875(val/ブラインド境界)={split_875.date()}")

    # ---- 手法A(現行方式): val=backtest=[split_75, 100%) ----
    tr0_a, va0_a = m.build_dataset(
        m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, regime_labels_for_loss=loss_regime_labels,
        global_split_date_override=split_75,
    )
    close_price_lookup_a = {}
    pool_base_a = m.prepare_backtest_pool(
        m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, global_split_date_override=split_75,
        close_price_out=close_price_lookup_a,
    )

    # ---- 手法B(リーク排除): val=[split_75, split_875)、backtest=[split_875, 100%) ----
    tr0_b, va0_b = m.build_dataset(
        m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, regime_labels_for_loss=loss_regime_labels,
        global_split_date_override=split_75, val_end_date_override=split_875,
    )
    close_price_lookup_b = {}
    pool_base_b = m.prepare_backtest_pool(
        m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
        macro_pool_df, regime_filter=None, global_split_date_override=split_875,
        close_price_out=close_price_lookup_b,
    )
    print(
        f"[+] 手法A: train={len(tr0_a[2])} val={len(va0_a[2])} backtest_pool={len(pool_base_a)}\n"
        f"[+] 手法B: train={len(tr0_b[2])} val={len(va0_b[2])} backtest_pool={len(pool_base_b)}"
    )

    tmpdir = tempfile.mkdtemp(prefix="measure_val_leakage_")
    shared_data_path = os.path.join(tmpdir, "shared_data.pkl")
    pss.dump_shared_training_data(
        shared_data_path,
        {
            "tr_A": tr0_a, "va_A": va0_a,
            "s_cols_A": m.BASELINE_STOCK_COLS, "m_cols_A": m.BASELINE_MACRO_COLS,
            "tr_B": tr0_b, "va_B": va0_b,
            "s_cols_B": m.BASELINE_STOCK_COLS, "m_cols_B": m.BASELINE_MACRO_COLS,
        },
    )

    seeds = m.SEEDS
    out_paths = {}
    jobs = []
    for label in ("A", "B"):
        for s in seeds:
            out_path = os.path.join(tmpdir, f"state_{label}_{s}.pt")
            out_paths[(label, s)] = out_path
            jobs.append((label, s, shared_data_path, out_path))

    print(f"[*] A×{len(seeds)}seed + B×{len(seeds)}seed = {len(jobs)}ジョブを最大{m.PARALLEL_MAX_CONCURRENT}並列で学習します...")
    t0 = time.time()
    pss.run_jobs_parallel(jobs, os.path.abspath(pss.__file__), m.PARALLEL_MAX_CONCURRENT)
    print(f"[+] 全{len(jobs)}ジョブ学習完了: 実時間={time.time() - t0:.1f}秒")

    import torch
    models_a, models_b = [], []
    for s in seeds:
        model_a = m.build_model(m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
        model_a.load_state_dict(torch.load(out_paths[("A", s)], map_location=m.DEVICE))
        models_a.append(model_a)
        model_b = m.build_model(m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS)
        model_b.load_state_dict(torch.load(out_paths[("B", s)], map_location=m.DEVICE))
        models_b.append(model_b)

    d_ens_a = m.run_backtest_inference_ensemble(models_a, pool_base_a)
    d_ens_b = m.run_backtest_inference_ensemble(models_b, pool_base_b)

    # 手法Aの評価結果を、手法Bのbacktest区間([split_875, 100%))だけに絞る
    # (=同じ評価区間・同じtrain区間で、checkpoint選択がその区間を見たか否かだけが違う比較)
    d_ens_a_subset = d_ens_a[pd_to_datetime(d_ens_a["date"]) >= split_875].copy()

    print("\n" + "=" * 90)
    print(f"【比較区間 [{split_875.date()}, 終端) のTOPN別PF】")
    print("=" * 90)
    print(f"手法A(val=backtest全体からの絞り込み、n(A全体)={len(d_ens_a)}, "
          f"n(A絞り込み後)={len(d_ens_a_subset)}) vs 手法B(val/backtestを分離、n(B)={len(d_ens_b)})")

    for cname, kwargs in [
        (f"業種制約 max_per_sector={m.MAX_PER_SECTOR}", dict(max_per_sector=m.MAX_PER_SECTOR)),
        (f"クラスタ制約 max_per_cluster={m.MAX_PER_CLUSTER}", dict(max_per_cluster=m.MAX_PER_CLUSTER)),
    ]:
        print(f"\n--- {cname} ---")
        topn_a = m.evaluate_topn_curve(
            d_ens_a_subset, topn_list=[1, 3, 5, 10, 20, 30],
            close_price_lookup=close_price_lookup_a, **kwargs,
        )
        topn_b = m.evaluate_topn_curve(
            d_ens_b, topn_list=[1, 3, 5, 10, 20, 30],
            close_price_lookup=close_price_lookup_b, **kwargs,
        )
        merged = topn_a[["top_n", "trades", "pf"]].rename(
            columns={"trades": "trades_A", "pf": "pf_A"}
        ).merge(
            topn_b[["top_n", "trades", "pf"]].rename(
                columns={"trades": "trades_B", "pf": "pf_B"}
            ),
            on="top_n",
        )
        merged["pf_diff(A-B)"] = merged["pf_A"] - merged["pf_B"]
        print(merged.to_string(index=False))

    del models_a, models_b
    torch.cuda.empty_cache()

    try:
        import shutil
        shutil.rmtree(tmpdir)
    except Exception as e:
        print(f"[!] 一時ディレクトリを削除できませんでした: {tmpdir} ({e})")


def pd_to_datetime(s):
    import pandas as pd
    return pd.to_datetime(s)


if __name__ == "__main__":
    main()
