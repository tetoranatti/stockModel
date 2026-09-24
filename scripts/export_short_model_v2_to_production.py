"""空売りモデルv2(research/short_exp/train_short_model_v2.py)の研究用チェックポイント
(raw state_dictのみ)を、本番modules/model_inference.py::load_trained_model()が読める
自己記述的な辞書形式に詰め直し、トップレベルにshort_model_v2_seed{42-46}.ptとして配置する
(2026-09-24追加、実運用へのショートモデル本番投入)。

研究側はUSE_SECTOR_EMBEDDING=True(research/stage3_exp/config.py)で学習しているため、
num_sectors/sector_embed_dimをここで明示的に埋め込む(本番のload_trained_model()は
これらの鍵が無いチェックポイントをnum_sectors=0として再構築してしまい、次元不一致で
state_dictのロードに失敗する)。stock_cols/hidden_dim/num_heads/use_cross_ffn等の
アーキテクチャ値は research/stage3_exp/config.py・train_short_model_v2.py と揃える。

実行方法:
    python scripts/export_short_model_v2_to_production.py
"""
import sys
import os

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

import torch
from short_exp.train_short_model_v2 import SHORT_STOCK_COLS_V2, SHORT_MODEL_CACHE_DIR
from stage3_exp.feature_catalog import BASELINE_MACRO_COLS
from stage3_exp.config import (
    MODEL_HIDDEN_DIM, MODEL_NUM_HEADS, USE_CROSS_FFN, SECTOR_EMBED_DIM, SEQ_LEN,
)
from modules.sector_embedding import NUM_SECTORS

BASE_DIR = r"F:\stockModel"
SEEDS = [42, 43, 44, 45, 46]


def main():
    for seed in SEEDS:
        src_path = os.path.join(SHORT_MODEL_CACHE_DIR, f"state_{seed}.pt")
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"{src_path}が見つかりません。先にtrain_short_model_v2.pyを実行してください。")

        state_dict = torch.load(src_path, map_location="cpu")
        checkpoint = {
            "model_state_dict": state_dict,
            "stock_cols": SHORT_STOCK_COLS_V2,
            "macro_cols": BASELINE_MACRO_COLS,
            "hidden_dim": MODEL_HIDDEN_DIM,
            "num_heads": MODEL_NUM_HEADS,
            "dropout": 0.2,
            "use_cross_ffn": USE_CROSS_FFN,
            "num_sectors": NUM_SECTORS,
            "sector_embed_dim": SECTOR_EMBED_DIM,
            "seq_len": SEQ_LEN,
        }
        dst_path = os.path.join(BASE_DIR, f"short_model_v2_seed{seed}.pt")
        torch.save(checkpoint, dst_path)
        print(f"[+] {src_path} -> {dst_path}")

    # 復元チェック(このプロセス内でmodules.model_inference.load_trained_modelに
    # 実際に読ませてforward1回通し、次元不一致等が無いことを確認する)
    from modules.model_inference import load_trained_models_ensemble, predict_probabilities_ensemble
    import numpy as np

    paths = [os.path.join(BASE_DIR, f"short_model_v2_seed{s}.pt") for s in SEEDS]
    models, stock_cols, macro_cols = load_trained_models_ensemble(paths)
    print(f"[+] 復元チェックOK: {len(models)}モデル stock_cols={stock_cols} macro_cols={macro_cols}")

    seq_len = SEQ_LEN
    w_s = np.random.randn(seq_len, len(stock_cols) + 1).astype(np.float32)
    w_s[:, -1] = 5  # ダミーのsector_id
    w_m = np.random.randn(seq_len, len(macro_cols)).astype(np.float32)
    p_win, p_stop, ev = predict_probabilities_ensemble(models, w_s, w_m)
    print(f"[+] ダミー入力での推論確認: p_win={p_win:.3f} p_stop={p_stop:.3f} ev={ev:.3f}")


if __name__ == "__main__":
    main()
