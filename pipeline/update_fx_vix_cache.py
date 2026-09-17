# update_fx_vix_cache.py
# VIX・USD/JPYの日次データをFRED(Federal Reserve Economic Data)から再取得する。
# modules/regime_risk_model.pyがこれらのCSVをffill()で読むため、更新を怠ると
# エラーなく古い値を使い続けてしまう(地合い危険度モデルの精度が気づかず劣化する)。
# FREDのCSVエンドポイントは毎回フル履歴を返すため、単純に上書き保存する。
# 注: Python標準のurllibだとこの環境でタイムアウトするため、curlをサブプロセス呼び出しする。
import os
import subprocess

BASE_DIR = r"F:\stockModel"
DATA_DIR = os.path.join(BASE_DIR, "data")

SOURCES = {
    "usdjpy_fred.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXJPUS",
    "vix_fred.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
}


def main():
    print("=" * 60)
    print("【VIX / USD-JPY 日次キャッシュ更新(FRED)】")
    print("=" * 60)
    os.makedirs(DATA_DIR, exist_ok=True)

    for fname, url in SOURCES.items():
        save_path = os.path.join(DATA_DIR, fname)
        try:
            result = subprocess.run(
                ["curl", "-sS", "--max-time", "30", "-o", save_path,
                 "-w", "%{http_code} %{size_download}", url],
                capture_output=True, text=True, timeout=40,
            )
            http_code, size = result.stdout.strip().split()
            size = int(size)
            if http_code != "200" or size < 1000:
                print(f"  [!] {fname}: 異常な応答のためスキップ (http={http_code}, size={size})")
                continue
            with open(save_path, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
            print(f"  [+] {fname}: 更新完了 ({len(lines)}行, 最新: {lines[-1]})")
        except Exception as e:
            print(f"  [!] {fname}: 取得失敗 - {e} (既存ファイルを維持)")

    print("[+] 完了")


if __name__ == "__main__":
    main()
