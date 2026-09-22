"""Main experiment orchestration extracted from the original script."""

import os
import time
import numpy as np
import pandas as pd
import torch

from modules.macro_features import load_macro_slim5
from modules.cross_sectional_features import valid_cross_section_dates
from _feature_cache_utils import (get_full_feature_pool_df, get_full_macro_pool_df,
    get_margin_pool_df, get_sector_relative_pool_df)
from _eval_utils import (compute_is_low_regime_filter, compute_is_high_regime_filter,
    compute_is_mid_regime_filter, compute_regime_labels, compute_regime_labels_expanding,
    regime_breakdown, evaluate, evaluate_daily_topn)
from training.train_model_v8_exp import UNIVERSE_PATH, MIN_CROSS_SECTION
from .config import *
from .runtime import DEVICE, log
from .feature_catalog import *
from .data_builder import *
from .cache import baseline_cache_fingerprint
from .model_factory import build_model
from .trainer import train_model
from .inference import run_backtest_inference, run_backtest_inference_ensemble
from .diagnostics import *

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_experiment():
    removed_from_baseline = (set(BASELINE_STOCK_COLS) | set(BASELINE_MACRO_COLS)) - set(
        TEST_STOCK_COLS + TEST_MACRO_COLS
    )
    added_candidates = set(TEST_STOCK_COLS + TEST_MACRO_COLS) - (
        set(BASELINE_STOCK_COLS) | set(BASELINE_MACRO_COLS)
    )
    print(
        f"[*] テスト側 株{len(TEST_STOCK_COLS)}個 + マクロ{len(TEST_MACRO_COLS)}個 = 計{len(TEST_FEATS)}個"
    )
    print(
        f"    既存から外した: {sorted(removed_from_baseline) if removed_from_baseline else '(なし)'}"
    )
    print(
        f"    新規追加した候補: {sorted(added_candidates) if added_candidates else '(なし)'}"
    )
    # archごとに実際に使われるパラメータだけを表示する(MODEL_ARCHに関わらず常に
    # dual_stream用トグルを表示すると、gru_macro_mlp_cross使用時に無関係な値を表示してしまう)。
    if MODEL_ARCH == "gru_macro_mlp_cross":
        print(
            f"[*] モデル設定: arch={MODEL_ARCH}, stock_hidden={STOCK_HIDDEN_DIM}, macro_hidden={MACRO_HIDDEN_DIM}, "
            f"num_heads={CROSS_ATTN_NUM_HEADS}, use_cross_ffn={CROSS_ATTN_USE_FFN}, "
            f"train_batch_size={TRAIN_BATCH_SIZE}, seeds={SEEDS}\n"
        )
    else:
        print(
            f"[*] モデル設定: arch={MODEL_ARCH}, hidden_dim={MODEL_HIDDEN_DIM}, num_heads={MODEL_NUM_HEADS}, "
            f"use_cross_ffn={USE_CROSS_FFN}, use_final_norm={USE_FINAL_NORM}, "
            f"num_self_attn_layers={NUM_SELF_ATTN_LAYERS}, train_batch_size={TRAIN_BATCH_SIZE}, seeds={SEEDS}\n"
        )

    with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    macro_df = load_macro_slim5(return_source="nk225")
    pool = get_full_feature_pool_df(tickers, macro_df)
    margin_pool = get_margin_pool_df(tickers, macro_df)
    sector_pool = get_sector_relative_pool_df(tickers, macro_df)
    macro_pool_df = augment_macro_pool_df(get_full_macro_pool_df(tickers))

    if REGIME_FILTER == "LOW":
        is_target = compute_is_low_regime_filter(macro_pool_df)
        log(
            f"[*] REGIME_FILTER=LOW: 全{len(is_target)}日中{int(is_target.sum())}日({is_target.mean()*100:.0f}%)"
            f"がLOWレジーム(拡大窓is_low、look-ahead無し)"
        )
        train_regime_filter = is_target
    elif REGIME_FILTER == "HIGH":
        is_target = compute_is_high_regime_filter(macro_pool_df)
        log(
            f"[*] REGIME_FILTER=HIGH: 全{len(is_target)}日中{int(is_target.sum())}日({is_target.mean()*100:.0f}%)"
            f"がHIGHレジーム(拡大窓is_high、look-ahead無し)"
        )
        train_regime_filter = is_target
    elif REGIME_FILTER == "MID":
        is_target = compute_is_mid_regime_filter(macro_pool_df)
        log(
            f"[*] REGIME_FILTER=MID: 全{len(is_target)}日中{int(is_target.sum())}日({is_target.mean()*100:.0f}%)"
            f"がMIDレジーム(拡大窓、look-ahead無し)"
        )
        train_regime_filter = is_target
    elif REGIME_FILTER is None:
        train_regime_filter = None
    else:
        raise ValueError(
            f"REGIME_FILTER='{REGIME_FILTER}'は未対応です(None / 'LOW' / 'MID' / 'HIGH'のみ)"
        )

    regime_labels = compute_regime_labels(macro_pool_df)
    if USE_REGIME_AWARE_LOSS:
        log(
            "[*] USE_REGIME_AWARE_LOSS=True: NearPairRankingLossのペアを同一regime同士に限定します"
            "(ペアタグ付けはlook-ahead無しの拡大窓判定を使用)"
        )

    # データセット・バックテストプールはシードに依存しないので、シードループの外で1回だけ構築する
    log("[*] データセット・バックテストプール構築(シード非依存、1回だけ)...")
    # regime_labels(全期間固定分位、報告用regime_breakdownのみで使う)とは別に、学習の
    # ペアタグ付けにはlook-ahead無しの拡大窓版を使う。
    loss_regime_labels = (
        compute_regime_labels_expanding(macro_pool_df)
        if USE_REGIME_AWARE_LOSS
        else None
    )

    # split dateはbase/testで完全に共通化する(build_dataset/prepare_backtest_poolが
    # それぞれ自分のvalid_datesから独立に計算すると、候補特徴量のNaNパターン次第で分割日が
    # 微妙に食い違いうるため)。ベースライン(既存14、常に固定)のvalid_datesを基準に
    # 1回だけ計算し、base/test両方に同じ値を渡す。
    _base_per_ticker = build_per_ticker(
        pool, margin_pool, sector_pool, macro_pool_df, BASELINE_STOCK_COLS
    )
    _base_valid_dates = valid_cross_section_dates(
        _base_per_ticker, BASELINE_STOCK_COLS, MIN_CROSS_SECTION
    )
    shared_split_date = compute_global_split_date(
        _base_valid_dates, train_regime_filter
    )
    log(f"[*] 共通split date(base/test): {shared_split_date.date()}")

    tr0, va0 = build_dataset(
        BASELINE_MACRO_COLS,
        BASELINE_STOCK_COLS,
        pool,
        margin_pool,
        sector_pool,
        macro_pool_df,
        regime_filter=train_regime_filter,
        regime_labels_for_loss=loss_regime_labels,
        global_split_date_override=shared_split_date,
    )
    tr1, va1 = build_dataset(
        TEST_MACRO_COLS,
        TEST_STOCK_COLS,
        pool,
        margin_pool,
        sector_pool,
        macro_pool_df,
        regime_filter=train_regime_filter,
        regime_labels_for_loss=loss_regime_labels,
        global_split_date_override=shared_split_date,
    )
    pool_base = prepare_backtest_pool(
        BASELINE_MACRO_COLS,
        BASELINE_STOCK_COLS,
        pool,
        margin_pool,
        sector_pool,
        macro_pool_df,
        regime_filter=train_regime_filter,
        global_split_date_override=shared_split_date,
    )
    pool_test = prepare_backtest_pool(
        TEST_MACRO_COLS,
        TEST_STOCK_COLS,
        pool,
        margin_pool,
        sector_pool,
        macro_pool_df,
        regime_filter=train_regime_filter,
        global_split_date_override=shared_split_date,
    )
    log(
        f"[+] train0={len(tr0[2])} val0={len(va0[2])} / train1={len(tr1[2])} val1={len(va1[2])}"
    )
    log(f"[+] backtest_pool base={len(pool_base)} test={len(pool_test)}")

    results_base, results_test = [], []
    models_base, models_test = [], []
    if PROFILE_FIRST_EPOCH:
        # プロファイリングは並列学習と相性が悪い(複数プロセスの出力が混ざる)ため、
        # PROFILE_FIRST_EPOCH=Trueのときだけ元の逐次ループにフォールバックする。
        for si, seed in enumerate(SEEDS):
            log(f"=== seed={seed} ===")
            model_base = train_model(
                seed,
                tr0,
                va0,
                BASELINE_STOCK_COLS,
                BASELINE_MACRO_COLS,
                "ベースライン(既存14固定)",
                profile_this_call=(si == 0),
            )
            d_base = run_backtest_inference(model_base, pool_base)
            model_test = train_model(
                seed,
                tr1,
                va1,
                TEST_STOCK_COLS,
                TEST_MACRO_COLS,
                f"テスト構成({len(TEST_FEATS)}個)",
            )
            d_test = run_backtest_inference(model_test, pool_test)

            r_base = evaluate(
                d_base,
                f"seed={seed} ベースライン score Top-K",
                min_reliable_n=MIN_RELIABLE_N,
                common_sample_k=COMMON_SAMPLE_K,
                max_concurrent=MAX_CONCURRENT_POSITIONS,
            )
            r_test = evaluate(
                d_test,
                f"seed={seed} テスト構成 score Top-K",
                min_reliable_n=MIN_RELIABLE_N,
                common_sample_k=COMMON_SAMPLE_K,
                max_concurrent=MAX_CONCURRENT_POSITIONS,
            )
            r_base["seed"] = seed
            r_test["seed"] = seed
            results_base.append(r_base)
            results_test.append(r_test)
            models_base.append(model_base)
            models_test.append(model_test)
            print()
    else:
        # base×seeds + test×seeds = 2*len(SEEDS)個の独立した学習ジョブを、1つのフラットな
        # jobリストとして並列学習する(parallel_seed_sweep.pyで検証済みの方式——3seed 2.44x、
        # 5seed 2.9x、逐次/並列で結果が完全一致、[[feedback_nn_training_performance]]参照)。
        import tempfile, shutil as _shutil, torch as _torch
        import parallel_seed_sweep as _pss

        # baselineモデルのキャッシュ。設定が前回と変わっていなければbaseline側の学習を
        # スキップしてキャッシュ済みstate_dictを再利用する(test側は毎回学習する)。
        _fp = baseline_cache_fingerprint(len(tr0[2]), len(va0[2]), shared_split_date)
        _cache_dir = os.path.join(
            RESEARCH_DIR,
            "cache",
            "baseline_model_cache",
            _fp,
        )
        os.makedirs(_cache_dir, exist_ok=True)
        _cached_seeds = {
            s
            for s in SEEDS
            if os.path.exists(os.path.join(_cache_dir, f"state_{s}.pt"))
        }
        if _cached_seeds:
            log(
                f"[*] baselineキャッシュ命中(fingerprint={_fp}): seed={sorted(_cached_seeds)} は学習をスキップします"
            )

        _tmpdir = tempfile.mkdtemp(prefix="stage3_main_parallel_")
        _shared_data_path = os.path.join(_tmpdir, "shared_data.pkl")

        _pss.dump_shared_training_data(
            _shared_data_path,
            {
                "tr_base": tr0,
                "va_base": va0,
                "s_cols_base": BASELINE_STOCK_COLS,
                "m_cols_base": BASELINE_MACRO_COLS,
                "tr_test": tr1,
                "va_test": va1,
                "s_cols_test": TEST_STOCK_COLS,
                "m_cols_test": TEST_MACRO_COLS,
            },
        )

        log(
            f"[+] worker共有データをmemmap化しました: "
            f"{os.path.join(_tmpdir, 'shared_arrays')}"
        )

        _jobs = []
        _out_paths = {}
        for seed in SEEDS:
            for label in ("base", "test"):
                if label == "base" and seed in _cached_seeds:
                    _out_paths[(label, seed)] = os.path.join(
                        _cache_dir, f"state_{seed}.pt"
                    )
                    continue
                out_path = os.path.join(_tmpdir, f"state_{label}_{seed}.pt")
                _jobs.append((label, seed, _shared_data_path, out_path))
                _out_paths[(label, seed)] = out_path

        log(
            f"[*] base×{len(SEEDS) - len(_cached_seeds)}seed(新規)+ test×{len(SEEDS)}seed = {len(_jobs)}ジョブを"
            f"並列学習します(最大{PARALLEL_MAX_CONCURRENT}並列)..."
        )
        _t0 = time.time()
        _pss.run_jobs_parallel(
            _jobs, os.path.abspath(_pss.__file__), PARALLEL_MAX_CONCURRENT
        )
        log(f"[+] 全{len(_jobs)}ジョブ学習完了: 実時間={time.time()-_t0:.1f}秒")

        # 新規学習したbaselineモデルをキャッシュに保存(次回以降の再利用のため)
        for seed in SEEDS:
            if seed not in _cached_seeds:
                _shutil.copy2(
                    _out_paths[("base", seed)],
                    os.path.join(_cache_dir, f"state_{seed}.pt"),
                )

        for seed in SEEDS:
            log(f"=== seed={seed}(推論・評価) ===")
            model_base = build_model(BASELINE_STOCK_COLS, BASELINE_MACRO_COLS)
            model_base.load_state_dict(
                _torch.load(_out_paths[("base", seed)], map_location=DEVICE)
            )
            model_base.eval()
            d_base = run_backtest_inference(model_base, pool_base)

            model_test = build_model(TEST_STOCK_COLS, TEST_MACRO_COLS)
            model_test.load_state_dict(
                _torch.load(_out_paths[("test", seed)], map_location=DEVICE)
            )
            model_test.eval()
            d_test = run_backtest_inference(model_test, pool_test)

            r_base = evaluate(
                d_base,
                f"seed={seed} ベースライン score Top-K",
                min_reliable_n=MIN_RELIABLE_N,
                common_sample_k=COMMON_SAMPLE_K,
                max_concurrent=MAX_CONCURRENT_POSITIONS,
            )
            r_test = evaluate(
                d_test,
                f"seed={seed} テスト構成 score Top-K",
                min_reliable_n=MIN_RELIABLE_N,
                common_sample_k=COMMON_SAMPLE_K,
                max_concurrent=MAX_CONCURRENT_POSITIONS,
            )
            r_base["seed"] = seed
            r_test["seed"] = seed
            results_base.append(r_base)
            results_test.append(r_test)
            models_base.append(model_base)
            models_test.append(model_test)
            print()

        try:
            import shutil

            shutil.rmtree(_tmpdir)
        except Exception as e:
            log(
                f"[!] 一時memmapディレクトリを削除できませんでした: "
                f"{_tmpdir} ({type(e).__name__}: {e})"
            )

    # アンサンブル評価: 単一seedはSTRONG BUYが特定regimeに偏る不安定性があるため、
    # 本番と同じsoftmax確率平均アンサンブル(predict_probabilities_ensemble_batch)での
    # 評価も追加する。SEEDS=1個でも単一seedと同じ結果になるだけなのでスキップしない。
    print("=" * 90)
    print(
        f"【アンサンブル評価({len(SEEDS)}seed平均、本番run_dynamic_regime_screening_v8.pyと同じロジック)】"
    )
    print("=" * 90)
    d_base_ens = run_backtest_inference_ensemble(models_base, pool_base)
    d_test_ens = run_backtest_inference_ensemble(models_test, pool_test)
    r_base_ens = evaluate(
        d_base_ens,
        f"ベースライン(アンサンブル{len(SEEDS)}seed) score Top-K",
        min_reliable_n=MIN_RELIABLE_N,
        common_sample_k=COMMON_SAMPLE_K,
        max_concurrent=MAX_CONCURRENT_POSITIONS,
    )
    r_test_ens = evaluate(
        d_test_ens,
        f"テスト構成(アンサンブル{len(SEEDS)}seed) score Top-K",
        min_reliable_n=MIN_RELIABLE_N,
        common_sample_k=COMMON_SAMPLE_K,
        max_concurrent=MAX_CONCURRENT_POSITIONS,
    )

    #TOPN別PF分析 月毎のPFバラツキがTOPNのランキング分け能力の問題か確認する
    print("\n" + "=" * 90)
    print("【TOPN別PF分析】")
    print("=" * 90)

    topn_df = evaluate_topn_curve(
        d_test_ens,
        topn_list=[1, 3, 5, 10, 20, 30]
    )

    print(topn_df.to_string(index=False))

    monthly_topn_df = monthly_topn_pf_matrix(
        d_test_ens,
        topn_list=[1, 3, 5, 10, 20, 30],
    )

    print("\n" + "=" * 90)
    print("【月別 × TopN PF】")
    print("=" * 90)
    print(monthly_topn_df.to_string())

    july_detail = daily_detail_for_month(
        d_test_ens,
        target_month="2026-07",
        top_n=5,
    )

    print(july_detail.to_string())

    july_top5_overlap = evaluate_daily_topn_overlap(
        d_test_ens,
        top_n=5,
        lookback_days=5,
        target_month="2026-07",
    )

    display_cols = [
        "date",
        "tickers",
        "previous_overlap_count",
        "previous_overlap_ratio",
        "lookback_overlap_count",
        "lookback_overlap_ratio",
        "avg_score",
        "avg_ret",
        "win_rate",
        "pf",
    ]

    print("\n" + "=" * 120)
    print("【2026年7月 日別Top5重複率】")
    print("=" * 120)

    print(
        july_top5_overlap[display_cols].to_string(
            index=False,
            formatters={
                "date": lambda x: x.strftime("%Y-%m-%d"),
                "previous_overlap_ratio": "{:.1%}".format,
                "lookback_overlap_ratio": "{:.1%}".format,
                "avg_score": "{:.4f}".format,
                "avg_ret": "{:.4f}".format,
                "win_rate": "{:.1%}".format,
                "pf": lambda x: (
                    "inf" if np.isinf(x) else f"{x:.3f}"
                ),
            },
        )
    )

    overlap_performance = (
        july_top5_overlap
        .assign(
            overlap_bucket=pd.cut(
                july_top5_overlap[
                    "lookback_overlap_ratio"
                ],
                bins=[-0.01, 0.20, 0.40, 0.60, 0.80, 1.00],
                labels=[
                    "0-20%",
                    "21-40%",
                    "41-60%",
                    "61-80%",
                    "81-100%",
                ],
            )
        )
        .groupby(
            "overlap_bucket",
            observed=True,
        )
        .agg(
            days=("date", "size"),
            avg_ret=("avg_ret", "mean"),
            positive_days=(
                "avg_ret",
                lambda x: int((x > 0).sum()),
            ),
            avg_score=("avg_score", "mean"),
        )
        .reset_index()
    )
    
    overlap_performance["positive_day_rate"] = (
        overlap_performance["positive_days"]
        / overlap_performance["days"]
    )
    
    print("\n【過去5取引日との重複率別成績】")
    print(
        overlap_performance.to_string(
            index=False,
            formatters={
                "avg_ret": "{:.4f}".format,
                "avg_score": "{:.4f}".format,
                "positive_day_rate": "{:.1%}".format,
            },
        )
    )

    # score(=EV_WIN_WEIGHT*p_win-EV_STOP_WEIGHT*p_stop、本番のev_scoreと同じ2:1重み)
    # 上位K件で選ぶ。決定論的tie-break(evaluate()と同じ基準: score降順→ticker昇順→date昇順)
    sub_base_ens = d_base_ens.sort_values(
        ["score", "ticker", "date"], ascending=[False, True, True]
    ).head(min(COMMON_SAMPLE_K, len(d_base_ens)))
    sub_test_ens = d_test_ens.sort_values(
        ["score", "ticker", "date"], ascending=[False, True, True]
    ).head(min(COMMON_SAMPLE_K, len(d_test_ens)))
    if len(sub_base_ens) > 0:
        regime_breakdown(
            sub_base_ens,
            "ベースライン(アンサンブル)",
            regime_labels,
            min_reliable_n=MIN_RELIABLE_N,
        )
    if len(sub_test_ens) > 0:
        regime_breakdown(
            sub_test_ens,
            "テスト構成(アンサンブル)",
            regime_labels,
            min_reliable_n=MIN_RELIABLE_N,
        )
    print(
        f"\n  [判定・アンサンブル] baseline PF={r_base_ens['pf']:.2f} vs テスト構成 PF={r_test_ens['pf']:.2f} -> "
        f"{'テスト構成の方が良い' if r_test_ens['pf'] > r_base_ens['pf'] else 'ベースラインの方が良い'}"
        f"(単一seed毎の判定より、こちらの方が実運用のアンサンブル推論に近い)"
    )
    print()

    # 日次Top-N運用評価。全期間score上位K件は特定の数日にシグナルが偏っていても検知できない
    # ため、実際の運用(毎日DAILY_TOPN件しか執行できない)に忠実な評価をアンサンブル予測に追加。
    print("=" * 90)
    print(
        f"【日次Top-N運用評価(1日あたり{DAILY_TOPN}件、アンサンブル{len(SEEDS)}seed)】"
    )
    print("=" * 90)
    r_base_daily = evaluate_daily_topn(
        d_base_ens,
        f"ベースライン(日次Top-{DAILY_TOPN})",
        n_per_day=DAILY_TOPN,
        min_reliable_n=MIN_RELIABLE_N,
        max_concurrent=MAX_CONCURRENT_POSITIONS,
    )
    r_test_daily = evaluate_daily_topn(
        d_test_ens,
        f"テスト構成(日次Top-{DAILY_TOPN})",
        n_per_day=DAILY_TOPN,
        min_reliable_n=MIN_RELIABLE_N,
        max_concurrent=MAX_CONCURRENT_POSITIONS,
    )
    print(
        f"\n  [判定・日次Top-N] baseline PF={r_base_daily['pf']:.2f} vs テスト構成 PF={r_test_daily['pf']:.2f} -> "
        f"{'テスト構成の方が良い' if r_test_daily['pf'] > r_base_daily['pf'] else 'ベースラインの方が良い'}"
        f"(全期間Top-Kより日々の執行に忠実。全期間Top-Kの判定と食い違う場合、全期間Top-Kの"
        f"結果が特定の数日に偏っていた可能性が高い)"
    )
    print()

    # base/testで特徴量セットが違うとdropnaで落ちる(ticker,date)が微妙に異なり得るため、
    # 両方に共通するサンプルだけに絞った比較も出す(「モデルの質の差」と「評価サンプル自体の
    # 差」が混ざるのを避けるため)。
    key_base = set(zip(d_base_ens["ticker"], d_base_ens["date"]))
    key_test = set(zip(d_test_ens["ticker"], d_test_ens["date"]))
    common_keys = key_base & key_test
    log(
        f"[*] base/testの(ticker,date)集合: base={len(key_base)} test={len(key_test)} common={len(common_keys)} "
        f"(baseのみ={len(key_base - key_test)}件, testのみ={len(key_test - key_base)}件)"
    )
    if len(common_keys) > 0:
        mask_base_common = [
            k in common_keys for k in zip(d_base_ens["ticker"], d_base_ens["date"])
        ]
        mask_test_common = [
            k in common_keys for k in zip(d_test_ens["ticker"], d_test_ens["date"])
        ]
        print("=" * 90)
        print(
            f"【base/test共通(ticker,date)サブセットでの比較(アンサンブル、共通n={len(common_keys)}件)】"
        )
        print("=" * 90)
        r_base_common = evaluate(
            d_base_ens[mask_base_common],
            "ベースライン(共通サブセット)",
            min_reliable_n=MIN_RELIABLE_N,
            common_sample_k=COMMON_SAMPLE_K,
            max_concurrent=MAX_CONCURRENT_POSITIONS,
        )
        r_test_common = evaluate(
            d_test_ens[mask_test_common],
            "テスト構成(共通サブセット)",
            min_reliable_n=MIN_RELIABLE_N,
            common_sample_k=COMMON_SAMPLE_K,
            max_concurrent=MAX_CONCURRENT_POSITIONS,
        )
        print(
            f"\n  [判定・共通サブセット] baseline PF={r_base_common['pf']:.2f} vs テスト構成 PF={r_test_common['pf']:.2f} -> "
            f"{'テスト構成の方が良い' if r_test_common['pf'] > r_base_common['pf'] else 'ベースラインの方が良い'}"
            f"(特徴量差によるdropnaパターンの違いを排除した、最も公平な比較)"
        )
        print()

    # 異なるモデル(seed)の生予測を単純にpd.concatして1つの大きなサンプルとして扱うのは
    # 統計的に妥当でないため行わない。regimeごとの内訳は上のアンサンブル評価
    # (複数モデルの確率平均、本番と同じ手法)の regime_breakdown を使う。

    df_base = pd.DataFrame(results_base)
    df_test = pd.DataFrame(results_test)

    print("=" * 90)
    print("【シード別サマリ(標準指標)】")
    print("=" * 90)
    cols_main = ["seed", "n", "win_rate", "pf", "avg_ret", "max_dd", "few_trades_flag"]
    print("  -- ベースライン(既存14固定) --")
    print(df_base[cols_main].to_string(index=False))
    print("  -- テスト構成 --")
    print(df_test[cols_main].to_string(index=False))

    print("\n" + "=" * 90)
    print("【シード別サマリ(top_k_pf・月別一貫性・銘柄集中度)】")
    print("=" * 90)
    cols_extra = [
        "seed",
        "top_k_pf",
        "top_k_n",
        "month_consistency",
        "top1_pct",
        "top5_pct",
        "hhi",
    ]
    print("  -- ベースライン --")
    print(df_base[cols_extra].to_string(index=False))
    print("  -- テスト構成 --")
    print(df_test[cols_extra].to_string(index=False))

    print("\n" + "=" * 90)
    print("【seedごとのPF差分(テスト構成 - ベースライン)】")
    print("=" * 90)
    diff_df = pd.DataFrame(
        {
            "seed": df_base["seed"],
            "pf_base": df_base["pf"],
            "pf_test": df_test["pf"],
            "pf_diff": df_test["pf"] - df_base["pf"],
            "top_k_pf_base": df_base["top_k_pf"],
            "top_k_pf_test": df_test["top_k_pf"],
            "top_k_pf_diff": df_test["top_k_pf"] - df_base["top_k_pf"],
        }
    )
    print(diff_df.to_string(index=False))
    n_seeds_test_better = (diff_df["pf_diff"] > 0).sum()
    print(
        f"\n  テスト構成がベースラインを上回ったseed数: {n_seeds_test_better}/{len(SEEDS)}"
        f"(約定ベースPF基準) | {(diff_df['top_k_pf_diff'] > 0).sum()}/{len(SEEDS)}(top_k_pf基準)"
    )

    print("\n" + "=" * 90)
    print("【mean ± std(組み合わせごとのシードばらつき比較用)】")
    print("=" * 90)
    finite_base_pf = df_base["pf"][np.isfinite(df_base["pf"])]
    finite_test_pf = df_test["pf"][np.isfinite(df_test["pf"])]
    print(
        f"  ベースライン PF: mean={finite_base_pf.mean():.3f} std={finite_base_pf.std():.3f} "
        f"(seed毎: {[round(x, 2) for x in df_base['pf']]})"
    )
    print(
        f"  テスト構成   PF: mean={finite_test_pf.mean():.3f} std={finite_test_pf.std():.3f} "
        f"(seed毎: {[round(x, 2) for x in df_test['pf']]})"
    )
    print(
        f"  ベースライン win_rate: mean={df_base['win_rate'].mean():.1f}% std={df_base['win_rate'].std():.1f}%"
    )
    print(
        f"  テスト構成   win_rate: mean={df_test['win_rate'].mean():.1f}% std={df_test['win_rate'].std():.1f}%"
    )
    print(
        f"  ベースライン MaxDD: mean={df_base['max_dd'].mean()*100:.1f}% std={df_base['max_dd'].std()*100:.1f}%"
    )
    print(
        f"  テスト構成   MaxDD: mean={df_test['max_dd'].mean()*100:.1f}% std={df_test['max_dd'].std()*100:.1f}%"
    )

    n_few_base = df_base["few_trades_flag"].sum()
    n_few_test = df_test["few_trades_flag"].sum()
    if n_few_base or n_few_test:
        print(
            f"\n  [!] 少数トレード警告(n<{MIN_RELIABLE_N}件): "
            f"ベースライン {n_few_base}/{len(SEEDS)}シード、テスト構成 {n_few_test}/{len(SEEDS)}シード"
            f" — 該当シードのPFは信頼性が低い可能性"
        )

    print("\n" + "=" * 90)
    print(
        f"【判定・単一seed平均】baseline PF mean={finite_base_pf.mean():.2f} vs テスト構成 PF mean={finite_test_pf.mean():.2f} -> "
        f"{'テスト構成の方が良い' if finite_test_pf.mean() > finite_base_pf.mean() else 'ベースラインの方が良い(このテスト構成は不採用推奨)'}"
    )
    print(
        f"    std比較: テスト構成のstdが{'大きい(不安定化)' if finite_test_pf.std() > finite_base_pf.std() else '小さい(安定化)'} "
        f"(baseline std={finite_base_pf.std():.3f} vs test std={finite_test_pf.std():.3f})"
    )
    print(
        f"【判定・アンサンブル(実運用に近い)】baseline PF={r_base_ens['pf']:.2f} vs テスト構成 PF={r_test_ens['pf']:.2f} -> "
        f"{'テスト構成の方が良い' if r_test_ens['pf'] > r_base_ens['pf'] else 'ベースラインの方が良い'}"
        f" (単一seed平均とアンサンブルの判定が食い違う場合は、単一seedのばらつき自体が"
        f"regime依存の可能性があるため、アンサンブル側を優先すること)"
    )
    print("=" * 90)
