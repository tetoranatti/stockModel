"""5日後Alpha回帰・weighted Huber版の空売りモデル学習(2026-09-24追加、ユーザー提案)。
バリア/トレーリングストップの分類ラベル(targets_short.py)とは別系統——target=alpha_5d
(日経相対5日先リターン)、loss=Weighted Huber、推論はTopN以外の変更不要(scoreへ変換して
既存の評価枠組みをそのまま使う)。5seedを逐次学習し、チェックポイントを永続キャッシュへ
保存する(バックテストは再学習せずキャッシュ読み込みだけで繰り返せる)。

実行方法:
    python research/short_exp/train_alpha_short_model.py
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
    from short_exp.data_builder_alpha import build_alpha_dataset
    from short_exp.trainer_alpha import train_alpha_model

    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    print(f"[*] ユニバース: {len(tickers)}銘柄")
    print(f"[*] 特徴量: stock={m.BASELINE_STOCK_COLS}")
    print(f"[*] 特徴量: macro={m.BASELINE_MACRO_COLS}")

    macro_df = m.load_macro_slim5(return_source="nk225")
    pool = m.get_full_feature_pool_df(tickers, macro_df, force_rebuild=False)
    margin_pool = m.get_margin_pool_df(tickers, macro_df, force_rebuild=False)
    sector_pool = m.get_sector_relative_pool_df(tickers, macro_df, force_rebuild=False)
    macro_pool_df = m.augment_macro_pool_df(
        m.get_full_macro_pool_df(tickers, force_rebuild=False)
    )
    nk_close_full = macro_df["NK_Close"]

    _base_per_ticker = m.build_per_ticker(
        pool, margin_pool, sector_pool, macro_pool_df, m.BASELINE_STOCK_COLS
    )
    _base_valid_dates = m.valid_cross_section_dates(
        _base_per_ticker, m.BASELINE_STOCK_COLS, m.MIN_CROSS_SECTION
    )
    shared_split_date = m.compute_global_split_date(_base_valid_dates, None)
    print(f"[*] split date: {shared_split_date.date()}")

    tr0, va0 = build_alpha_dataset(
        m.BASELINE_MACRO_COLS, m.BASELINE_STOCK_COLS, pool, margin_pool, sector_pool,
        macro_pool_df, nk_close_full, regime_filter=None, global_split_date_override=shared_split_date,
    )
    tr_y, va_y = tr0[2], va0[2]
    print(f"[+] train={len(tr_y)} val={len(va_y)}")
    print(f"[*] train alpha_5d: mean={tr_y.mean():.4f} std={tr_y.std():.4f} "
          f"min={tr_y.min():.4f} max={tr_y.max():.4f}")
    print(f"[*] train weight: mean={tr0[3].mean():.3f} max={tr0[3].max():.3f}")

    os.makedirs(ALPHA_MODEL_CACHE_DIR, exist_ok=True)
    for seed in m.SEEDS:
        ckpt_path = os.path.join(ALPHA_MODEL_CACHE_DIR, f"state_{seed}.pt")
        if os.path.exists(ckpt_path):
            print(f"[*] seed={seed}は既にキャッシュ済み(途中終了からの再開用): {ckpt_path}")
            continue
        model = train_alpha_model(seed, tr0, va0, m.BASELINE_STOCK_COLS, m.BASELINE_MACRO_COLS,
                                   label=f"alpha5d seed={seed}")
        torch.save(model.state_dict(), ckpt_path)
        del model
        torch.cuda.empty_cache()
    print(f"[+] チェックポイントを保存しました: {ALPHA_MODEL_CACHE_DIR}")


if __name__ == "__main__":
    main()
