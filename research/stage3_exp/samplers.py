"""Batch samplers and dataset ordering helpers."""

import random
import numpy as np

class SameDateBatchSampler:
    """バッチを「同一取引日(tidx、全銘柄共通のカレンダー日位置)」だけで構成するサンプラー
    (SameTickerBatchSamplerの同日横断/cross-sectional版)。同じバッチ内の全サンプルが
    同じtidxになるため、NearPairRankingLossRegimeAware(cross_sectional=True)のペア制約
    (gap==0)をバッチ構成の時点で自動的に満たす。PAIR_MODE="cross_sectional"のとき
    SameTickerBatchSamplerの代わりに使う(2026-09-21追加)。1バッチ=1取引日の全銘柄横断
    サンプル(batch_size超なら分割)。SameTickerBatchSamplerと同じインターフェース
    (__iter__/__len__のみ、torch.utils.data.Samplerは継承しない——train_model()側も
    Dataset/DataLoaderを経由せずlist(sampler)で直接消費するため不要)。"""

    def __init__(self, tidx_array, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        groups = {}
        for i, d in enumerate(tidx_array):
            groups.setdefault(int(d), []).append(i)
        self._batches = []
        for d, idxs in groups.items():
            for start in range(0, len(idxs), batch_size):
                chunk = idxs[start : start + batch_size]
                if len(chunk) >= 4:
                    self._batches.append(chunk)

    def __iter__(self):
        batches = self._batches
        if self.shuffle:
            batches = batches.copy()
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self._batches)

def reorder_dataset_by_tidx(data):
    """datasetをtidx昇順へ1回だけ並べ替える。

    PAIR_MODE="cross_sectional"では同一日サンプルを連続配置し、
    学習・検証バッチを連続sliceで取得できるようにする。
    同一tidx内の元順序はstable sortで維持する。
    """
    x_s, x_m, y, tid, tidx, regime = data

    order = np.argsort(tidx, kind="stable")

    return (
        np.ascontiguousarray(x_s[order]),
        np.ascontiguousarray(x_m[order]),
        np.ascontiguousarray(y[order]),
        np.ascontiguousarray(tid[order]),
        np.ascontiguousarray(tidx[order]),
        np.ascontiguousarray(regime[order]),
    )

def build_same_date_ranges(tidx_array, batch_size):
    """tidx順に並んだ配列から、同一日だけで構成される連続範囲を作る。

    戻り値:
        [(start, end), ...]

    endはexclusive。
    同一日サンプル数がbatch_sizeを超える場合は、その日付内で分割する。
    4件未満のチャンクは従来のSameDateBatchSamplerと同様に除外する。
    """
    tidx_array = np.asarray(tidx_array)

    if len(tidx_array) == 0:
        return []

    change_points = np.flatnonzero(tidx_array[1:] != tidx_array[:-1]) + 1

    boundaries = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            change_points,
            np.array([len(tidx_array)], dtype=np.int64),
        ]
    )

    ranges = []

    for group_start, group_end in zip(
        boundaries[:-1],
        boundaries[1:],
    ):
        for start in range(
            int(group_start),
            int(group_end),
            batch_size,
        ):
            end = min(
                start + batch_size,
                int(group_end),
            )

            if end - start >= 4:
                ranges.append((start, end))

    return ranges

def validate_same_date_ranges(tidx_array, ranges, label):
    """連続範囲内が同一tidxだけで構成されることを検証する。"""
    for start, end in ranges:
        if end - start < 4:
            raise AssertionError(
                f"{label}: 4件未満のバッチがあります: " f"start={start}, end={end}"
            )

        batch_tidx = tidx_array[start:end]

        if not np.all(batch_tidx == batch_tidx[0]):
            raise AssertionError(
                f"{label}: 異なる日付が同一バッチへ混入しています: "
                f"start={start}, end={end}, "
                f"unique_tidx={np.unique(batch_tidx).tolist()}"
            )
