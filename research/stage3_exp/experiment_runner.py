"""Main experiment orchestration extracted from the original script."""

import os
import time

import pandas as pd

from modules.macro_features import load_macro_slim5
from modules.cross_sectional_features import valid_cross_section_dates
from _feature_cache_utils import (
    get_full_feature_pool_df,
    get_full_macro_pool_df,
    get_margin_pool_df,
    get_sector_relative_pool_df,
)
from training.train_model_v8_exp import UNIVERSE_PATH, MIN_CROSS_SECTION
from .config import *
from .runtime import DEVICE, log
from .feature_catalog import *
from .data_builder import *
from .cache import baseline_cache_fingerprint
from .model_factory import build_model
from .trainer import train_model
from .inference import run_backtest_inference_ensemble
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

    if USE_REGIME_AWARE_LOSS:
        log(
            "[*] USE_REGIME_AWARE_LOSS=True: NearPairRankingLossのペアを同一regime同士に限定します"
            "(ペアタグ付けはlook-ahead無しの拡大窓判定を使用)"
        )

    # データセット・バックテストプールはシードに依存しないので、シードループの外で1回だけ構築する
    log("[*] データセット・バックテストプール構築(シード非依存、1回だけ)...")
    # 学習のペアタグ付けにはlook-ahead無しの拡大窓版を使う。
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
    close_price_lookup_base = {}
    pool_base = prepare_backtest_pool(
        BASELINE_MACRO_COLS,
        BASELINE_STOCK_COLS,
        pool,
        margin_pool,
        sector_pool,
        macro_pool_df,
        regime_filter=train_regime_filter,
        global_split_date_override=shared_split_date,
        close_price_out=close_price_lookup_base,
    )
    close_price_lookup_test = {}
    pool_test = prepare_backtest_pool(
        TEST_MACRO_COLS,
        TEST_STOCK_COLS,
        pool,
        margin_pool,
        sector_pool,
        macro_pool_df,
        regime_filter=train_regime_filter,
        global_split_date_override=shared_split_date,
        close_price_out=close_price_lookup_test,
    )
    log(
        f"[+] train0={len(tr0[2])} val0={len(va0[2])} / train1={len(tr1[2])} val1={len(va1[2])}"
    )
    log(f"[+] backtest_pool base={len(pool_base)} test={len(pool_test)}")

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
            model_test = train_model(
                seed,
                tr1,
                va1,
                TEST_STOCK_COLS,
                TEST_MACRO_COLS,
                f"テスト構成({len(TEST_FEATS)}個)",
            )
            models_base.append(model_base)
            models_test.append(model_test)
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
            log(f"=== seed={seed}(モデル読み込み) ===")
            model_base = build_model(BASELINE_STOCK_COLS, BASELINE_MACRO_COLS)
            model_base.load_state_dict(
                _torch.load(_out_paths[("base", seed)], map_location=DEVICE)
            )
            model_base.eval()

            model_test = build_model(TEST_STOCK_COLS, TEST_MACRO_COLS)
            model_test.load_state_dict(
                _torch.load(_out_paths[("test", seed)], map_location=DEVICE)
            )
            model_test.eval()

            models_base.append(model_base)
            models_test.append(model_test)

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
    # 評価を使う。SEEDS=1個でも単一seedと同じ結果になるだけなのでスキップしない。
    d_base_ens = run_backtest_inference_ensemble(models_base, pool_base)
    d_test_ens = run_backtest_inference_ensemble(models_test, pool_test)

    def _topn_compare_table(**kwargs):
        """TOPN別PF分析をbase/test両方で計算する(2026-09-23、ユーザー指摘: 実際に見ている
        のはこの3表だけなのにtest側しか出しておらずbase/testの比較ができていなかったため)。
        base_df/test_df/diff_df(pf差分のみ)を返す——表示側で縦二段に積んで出す
        (2026-09-23再改修、ユーザー指摘: base_test列が横並びだと行が長すぎて比較しづらい)。"""
        cols = [
            "top_n", "trades", "pf", "avg_ret", "cagr", "max_dd",
            "worst_month_pf", "median_month_pf", "pf_lt_05_months",
        ]
        base_df = evaluate_topn_curve(
            d_base_ens, topn_list=[1, 3, 5, 10, 20, 30],
            close_price_lookup=close_price_lookup_base, **kwargs,
        )[cols]
        test_df = evaluate_topn_curve(
            d_test_ens, topn_list=[1, 3, 5, 10, 20, 30],
            close_price_lookup=close_price_lookup_test, **kwargs,
        )[cols]
        diff_df = pd.DataFrame({
            "top_n": base_df["top_n"].values,
            "pf_base": base_df["pf"].values,
            "pf_test": test_df["pf"].values,
            "pf_diff": test_df["pf"].values - base_df["pf"].values,
        })
        return base_df, test_df, diff_df

    def _print_topn_compare(title, **kwargs):
        base_df, test_df, diff_df = _topn_compare_table(**kwargs)
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)
        print("  -- ベースライン --")
        print(base_df.to_string(index=False))
        print("  -- テスト構成 --")
        print(test_df.to_string(index=False))
        print("  -- 差分(pf_test - pf_base) --")
        print(diff_df.to_string(index=False))

    # TOPN別PF分析 月毎のPFバラツキがTOPNのランキング分け能力の問題か確認する
    _print_topn_compare("【TOPN別PF分析(base vs test)】", max_per_sector=MAX_PER_SECTOR)

    # 60日リターン相関クラスタ制約版(2026-09-23追加、業種制約版との比較用)
    if TICKER_TO_CLUSTER:
        _print_topn_compare(
            f"【TOPN別PF分析(クラスタ制約 max_per_cluster={MAX_PER_CLUSTER}, "
            f"N_CLUSTERS={N_CLUSTERS}) base vs test】",
            max_per_cluster=MAX_PER_CLUSTER,
        )

    monthly_topn_df_base = monthly_topn_pf_matrix(
        d_base_ens, topn_list=[1, 3, 5, 10, 20, 30],
    )
    monthly_topn_df_test = monthly_topn_pf_matrix(
        d_test_ens, topn_list=[1, 3, 5, 10, 20, 30],
    )

    print("\n" + "=" * 90)
    print("【月別 × TopN PF】")
    print("=" * 90)
    print("  -- ベースライン --")
    print(monthly_topn_df_base.to_string())
    print("  -- テスト構成 --")
    print(monthly_topn_df_test.to_string())

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
            close_price_lookup=close_price_lookup_base,
        )
        r_test_common = evaluate(
            d_test_ens[mask_test_common],
            "テスト構成(共通サブセット)",
            min_reliable_n=MIN_RELIABLE_N,
            common_sample_k=COMMON_SAMPLE_K,
            max_concurrent=MAX_CONCURRENT_POSITIONS,
            close_price_lookup=close_price_lookup_test,
        )
        print(
            f"\n  [判定・共通サブセット] baseline PF={r_base_common['pf']:.2f} vs テスト構成 PF={r_test_common['pf']:.2f} -> "
            f"{'テスト構成の方が良い' if r_test_common['pf'] > r_base_common['pf'] else 'ベースラインの方が良い'}"
            f"(特徴量差によるdropnaパターンの違いを排除した、最も公平な比較)"
        )
        print()

    print("=" * 90)
