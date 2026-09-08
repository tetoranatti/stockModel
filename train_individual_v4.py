import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 既存の実験スクリプトからパーツを流用
from swing_pipeline_v4_experimental import (
    DualStreamGRU_v4,
    AsymmetricPenaltyLoss,
    load_and_preprocess_v4,
    generate_triple_barrier_labels,
    TimeSeriesDataset
)

def train_single_ticker(ticker: str, name: str, epochs: int = 8):
    save_path = f"swing_model_v4_{ticker.split('.')[0]}.pt"
    print(f"\n{'='*65}")
    print(f"[*] 【個別最適化学習】銘柄: {name} ({ticker})")
    print(f"[*] モデル保存先: {save_path}")
    print(f"{'='*65}")

    df, stock_cols, macro_cols = load_and_preprocess_v4(ticker)
    if len(df) < 60:
        print(f"[-] {name}: データ数が不足しているため学習をスキップします。")
        return

    df = generate_triple_barrier_labels(df)
    
    stock_data = df[stock_cols].values
    macro_data = df[macro_cols].values
    targets = df['Target'].values

    # 時系列 80:20 分割
    split_idx = int(len(df) * 0.8)
    train_ds = TimeSeriesDataset(stock_data[:split_idx], macro_data[:split_idx], targets[:split_idx])
    val_ds = TimeSeriesDataset(stock_data[split_idx:], macro_data[split_idx:], targets[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = DualStreamGRU_v4(
        stock_dim=len(stock_cols),
        macro_dim=len(macro_cols),
        hidden_dim=44,
        num_classes=3
    )

    # 誤爆防止の非対称ペナルティ損失
    criterion = AsymmetricPenaltyLoss(false_buy_penalty=3.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x_s, x_m, y in train_loader:
            optimizer.zero_grad()
            out = model(x_s, x_m)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if epoch % 4 == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    # チェックポイント保存
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stock_cols": stock_cols,
        "macro_cols": macro_cols,
        "ticker": ticker,
        "model_version": f"v4_{ticker.split('.')[0]}"
    }
    torch.save(checkpoint, save_path)
    print(f"[+] {name} 専用モデルを '{save_path}' に保存しました。")

    # 検証セットでの確率チェック
    model.eval()
    val_p_profit = []
    with torch.no_grad():
        for x_s, x_m, _ in val_loader:
            logits = model(x_s, x_m)
            probs = torch.softmax(logits, dim=-1).numpy()
            val_p_profit.extend(probs[:, 2])

    print(f"[*] 検証期間 利確確率: 平均={np.mean(val_p_profit)*100:.1f}%, 最大={np.max(val_p_profit)*100:.1f}%")

if __name__ == "__main__":
    targets = {
        "7012.T": "川崎重工業",
        "8227.T": "しまむら",
        "3407.T": "旭化成",
        "8593.T": "三菱HCキャピタル"
    }
    for sym, nm in targets.items():
        train_single_ticker(ticker=sym, name=nm, epochs=8)