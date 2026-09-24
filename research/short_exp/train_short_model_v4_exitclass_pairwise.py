"""空売りモデルv4学習: 決済理由ラベル(exit_class)+既存のペアワイズランキング損失
(2026-09-24追加、ユーザー提案「v3(CE)が悪化した原因が離散ラベルなのかCE損失なのか
切り分けたい」)。v3(exit_class+CrossEntropy)はv2比で明確に悪化(PF 1.51->0.75)したが、
「離散化自体が悪い」のか「CE損失自体が悪い」のかは未確定だった。本スクリプトはv3と同じ
exit_classラベル(0=Stop/1=Hold/2=Win)を使いつつ、損失関数だけをv2と同じ
NearPairRankingLossRegimeAware(ペアワイズ順位学習)に戻す——ラベルの離散化だけを独立変数
として切り出す。

NearPairRankingLossRegimeAwareはsign(ret_pct差)しか見ないため、離散ラベル(0/1/2)でも
無改造でそのまま動く(同クラス同士のペアはsign=0で自然に除外される)。そのため学習
インフラはtrain_short_model_v2.pyと完全に同じ(train_configs_parallel経由の5seed並列学習)
で、build_dataset_shortの呼び出しにuse_exit_class_labels=Trueを渡すだけで済む。

実行方法:
    python research/short_exp/train_short_model_v4_exitclass_pairwise.py
"""
import sys
import os
import time
import shutil

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

from train_short_model_v2 import SHORT_STOCK_COLS_V2

SHORT_MODEL_CACHE_DIR_V4 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cache", "short_model_cache_v4_exitclass_pairwise"
)
MAX_CONCURRENT = 10


def main():
    import stage3_exp.experiment_context as m
    import parallel_seed_sweep as pss
    import greedy_feature_search as gfs
    from short_exp.data_builder_short import build_dataset_short

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄")
    print(f"[*] 特徴量: stock={SHORT_STOCK_COLS_V2}")
    print(f"[*] 特徴量: macro={m.BASELINE_MACRO_COLS}")
    print("[*] ラベル: exit_class(0=Stop/1=Hold/2=Win) 損失: NearPairRankingLossRegimeAware(v2と同一)")

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

    configs = {"short_v4": (tr0, va0, SHORT_STOCK_COLS_V2, m.BASELINE_MACRO_COLS)}
    print(f"\n[*] {len(m.SEEDS)}seedを最大{MAX_CONCURRENT}並列で学習します...")
    t0 = time.time()
    out_paths, tmpdir, elapsed = gfs.train_configs_parallel(m, pss, configs, MAX_CONCURRENT)
    print(f"[+] 学習完了: 実時間={elapsed:.1f}秒")

    os.makedirs(SHORT_MODEL_CACHE_DIR_V4, exist_ok=True)
    for seed in m.SEEDS:
        shutil.copy2(
            out_paths[("short_v4", seed)],
            os.path.join(SHORT_MODEL_CACHE_DIR_V4, f"state_{seed}.pt"),
        )
    print(f"[+] チェックポイントを保存しました: {SHORT_MODEL_CACHE_DIR_V4}")

    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
