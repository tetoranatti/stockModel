"""空売りモデルの特徴量backward elimination(2026-09-24追加、ユーザー指摘「ロングを
単純に逆にしただけではダメだと分かっていたので特徴量を選び直す必要がある」)。

greedy_feature_backward.pyのrun_backward_elimination()をbuild_dataset_fn/
prepare_backtest_pool_fnパラメータ経由でそのまま再利用し、data_builder_short側の
関数を渡すだけ。現在の空売りモデルはロング版のBASELINE_STOCK_COLS/BASELINE_MACRO_COLS
(10+5)をそのまま流用しているだけなので、まずはそこから「空売りラベルの下で本当に
効いている特徴量」を後退法で絞り込む(ロング用に選ばれた特徴量が空売りにも同じように
効くとは限らない)。

キャッシュ名前空間はcache_dir_name未使用(backward版はbaseline_model_cacheを使わない
ため元から衝突しない)。ログはresult_log_pathで別ファイルに分離し、ロング側の
greedy_backward_log.parquetを上書きしない。

実行方法:
    python research/short_exp/backward_elim_short.py
    python research/short_exp/backward_elim_short.py --margin 0.01
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

RESULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cache", "short_backward_log.parquet"
)


def main():
    import greedy_feature_backward as gb
    from short_exp.data_builder_short import build_dataset_short, prepare_backtest_pool_short

    margin_arg, max_concurrent_arg = 0.0, 10
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--margin":
            margin_arg = float(args[i + 1]); i += 2
        elif args[i] == "--max-concurrent":
            max_concurrent_arg = int(args[i + 1]); i += 2
        else:
            i += 1

    gb.run_backward_elimination(
        margin=margin_arg, max_concurrent=max_concurrent_arg,
        build_dataset_fn=build_dataset_short, prepare_backtest_pool_fn=prepare_backtest_pool_short,
        result_log_path=RESULT_LOG_PATH,
    )


if __name__ == "__main__":
    main()
