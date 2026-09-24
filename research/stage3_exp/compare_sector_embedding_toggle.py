"""USE_SECTOR_EMBEDDING(config.py)のTrue/False切り替えで現行150銘柄ユニバースの
PFがどう変わるかを確認する(2026-09-23)。

背景: sector_embedding_2026-09-21の記録では「USE_SECTOR_EMBEDDING=Falseがデフォルト」
のはずだったが、git履歴を確認するとコミット0502b93(ファイルの統合と修正)で
False->Trueへ変わっており、現在のconfig.pyはTrueのまま(以降のコミットで戻されて
いない)。ファイル統合作業の副作用で意図せず有効化されたままになっている疑いがある。
本スクリプトでconfig.py上の値を一時的に書き換えて両方を実際に学習・評価し、
現在のコード/データで実際どちらが良いかを確認する(実行後、元の値に戻す)。

USE_SECTOR_EMBEDDINGはデータ形状(w_sへのsector-id列追加)とモデル構築(embedding層の
有無)の両方に影響し、sector_embedding_2026-09-21の記録どおりworker再import(並列学習
サブプロセス)にも波及するため、モンキーパッチではなくconfig.py実体を書き換えてから
学習ジョブを起動する必要がある。

本番キャッシュ・universe_150_tickers.txtは変更しない(現行の値をそのまま使い、
キャッシュパスの差し替えも行わない=production _feature_cache_utils キャッシュを
そのまま再利用する)。

実行方法:
    python research/stage3_exp/compare_sector_embedding_toggle.py
"""
import os
import re
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

CONFIG_PATH = r"F:\stockModel\research\stage3_exp\config.py"
TOGGLE_PATTERN = re.compile(r"^USE_SECTOR_EMBEDDING = (True|False)$", re.MULTILINE)


def _read_current_toggle():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    m = TOGGLE_PATTERN.search(text)
    if not m:
        raise RuntimeError("USE_SECTOR_EMBEDDING の行が見つかりません")
    return text, m.group(1) == "True"


def _set_toggle(value: bool):
    text, _ = _read_current_toggle()
    new_text = TOGGLE_PATTERN.sub(f"USE_SECTOR_EMBEDDING = {value}", text)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_text)


def run_one_config(label):
    """現在ディスク上のconfig.py設定のまま、baselineだけを5seed学習・評価する
    (experiment_runner.pyのbaseline部分相当を単独抽出、production caches使用)。"""
    # モジュールキャッシュを完全に切り離すため、このプロセス内でstage3_exp配下を
    # 都度サブプロセスに任せる(parallel_seed_sweep経由のworkerは毎回新規プロセスで
    # config.pyを再読込するため問題ないが、親プロセス側のstage3_exp.*はここで初めて
    # importするので、他の設定で既にimport済みだと反映されない。本スクリプトは
    # 1プロセス内でTrue/False両方を評価するため、都度サブプロセスとして呼び出す)。
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "_run_baseline_once.py")
    result = subprocess.run(
        [sys.executable, script],
        cwd=r"F:\stockModel",
        capture_output=True, text=True, encoding="utf-8",
    )
    print(f"\n{'=' * 30} [{label}] stdout {'=' * 30}")
    print(result.stdout[-6000:])
    if result.returncode != 0:
        print(f"{'=' * 30} [{label}] stderr {'=' * 30}")
        print(result.stderr[-4000:])
    return result.returncode == 0


def main():
    original_text, original_value = _read_current_toggle()
    print(f"[*] config.py上の現在値: USE_SECTOR_EMBEDDING={original_value}")

    try:
        for value in (True, False):
            _set_toggle(value)
            print(f"\n\n{'#' * 40}\n# USE_SECTOR_EMBEDDING = {value}\n{'#' * 40}")
            ok = run_one_config(label=f"USE_SECTOR_EMBEDDING={value}")
            if not ok:
                print(f"[!] USE_SECTOR_EMBEDDING={value} の実行に失敗しました。中断します。")
                break
    finally:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(original_text)
        print(f"\n[+] config.pyを元の値(USE_SECTOR_EMBEDDING={original_value})に復元しました。")


if __name__ == "__main__":
    main()
