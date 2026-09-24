# modules/sector_embedding.py
# 空売りモデルv2(USE_SECTOR_EMBEDDING=True学習)を本番で動かすためのセクターID
# マッピング(2026-09-24追加)。research/stage3_exp/feature_catalog.pyの
# TICKER_TO_SECTOR_ID/NUM_SECTORS構築ロジックと完全に同一にする必要がある
# (学習時と推論時でsector_id(embeddingのindex)がズレると学習した重みと整合しなくなるため)。
# データソースはdata/jpx_sector_master.json(research側の_load_sector_map()と同じファイル)。
import json
import os

BASE_DIR = r"F:\stockModel"
SECTOR_MASTER_JSON = os.path.join(BASE_DIR, "data", "jpx_sector_master.json")
SECTOR_EMBED_DIM = 4  # research/stage3_exp/config.py::SECTOR_EMBED_DIMと同じ値


def _load_sector_map():
    """ticker('XXXX.T') -> sector33業種名 の辞書を返す。
    research/_feature_cache_utils.py::_load_sector_map()と同一ロジック。"""
    with open(SECTOR_MASTER_JSON, "r", encoding="utf-8") as f:
        master = json.load(f)
    sector_map = {}
    for code, rec in master.items():
        sector_map[f"{code}.T"] = rec.get("sector33")
    return sector_map


_SECTOR_MAP_RAW = _load_sector_map()
_SECTOR_NAMES_SORTED = sorted({v for v in _SECTOR_MAP_RAW.values() if v is not None})
SECTOR_NAME_TO_ID = {name: i + 1 for i, name in enumerate(_SECTOR_NAMES_SORTED)}
TICKER_TO_SECTOR_ID = {
    t: SECTOR_NAME_TO_ID.get(v, 0) for t, v in _SECTOR_MAP_RAW.items()
}
NUM_SECTORS = len(_SECTOR_NAMES_SORTED) + 1
