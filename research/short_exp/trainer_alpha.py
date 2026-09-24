"""Weighted Huber回帰モデルの学習ループ(2026-09-24追加)。stage3_exp.trainer.train_model
(NearPairRankingLoss用、同日クロスセクショナルバッチが必須)とは異なり、単純な教師あり回帰
なのでランダムシャッフルしたミニバッチで良い——ペアリング・同日バッチ構造は不要。
並列マルチプロセス学習(parallel_seed_sweep.py)は分類モデル用のデータ形式・損失を
前提にしているため、v1では5seedを1プロセス内で逐次学習する(学習自体は分類モデルと
同程度に軽い)。"""
import torch
import torch.nn as nn
from stage3_exp.runtime import DEVICE, log, set_seed
from stage3_exp.config import TRAIN_BATCH_SIZE, VAL_BATCH_SIZE
from .losses_huber import WeightedHuberLoss
from .model_alpha import build_alpha_model


def train_alpha_model(seed, tr_data, va_data, s_cols, m_cols, label):
    set_seed(seed)
    tr_x_s, tr_x_m, tr_y, tr_w = tr_data
    va_x_s, va_x_m, va_y, va_w = va_data

    tr_xs_t = torch.tensor(tr_x_s, dtype=torch.float32, device=DEVICE)
    tr_xm_t = torch.tensor(tr_x_m, dtype=torch.float32, device=DEVICE)
    tr_y_t = torch.tensor(tr_y, dtype=torch.float32, device=DEVICE)
    tr_w_t = torch.tensor(tr_w, dtype=torch.float32, device=DEVICE)

    va_xs_t = torch.tensor(va_x_s, dtype=torch.float32, device=DEVICE)
    va_xm_t = torch.tensor(va_x_m, dtype=torch.float32, device=DEVICE)
    va_y_t = torch.tensor(va_y, dtype=torch.float32, device=DEVICE)
    va_w_t = torch.tensor(va_w, dtype=torch.float32, device=DEVICE)

    model = build_alpha_model(s_cols, m_cols)
    criterion = WeightedHuberLoss(delta=1.0)
    fused_adamw = DEVICE.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=3e-2, fused=fused_adamw)

    n_train = len(tr_y)
    log(f"  [{label}] train={n_train} val={len(va_y)}")
    best_val_loss, best_state, patience_cnt = float("inf"), None, 0
    patience, epochs = 7, 30

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, TRAIN_BATCH_SIZE):
            idx = perm[start:start + TRAIN_BATCH_SIZE]
            b_xs, b_xm, b_y, b_w = tr_xs_t[idx], tr_xm_t[idx], tr_y_t[idx], tr_w_t[idx]
            pred = model(b_xs, b_xm).squeeze(-1)
            loss = criterion(pred, b_y, b_w)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            va_pred_parts = []
            for start in range(0, len(va_y), VAL_BATCH_SIZE):
                b_xs = va_xs_t[start:start + VAL_BATCH_SIZE]
                b_xm = va_xm_t[start:start + VAL_BATCH_SIZE]
                va_pred_parts.append(model(b_xs, b_xm).squeeze(-1))
            va_pred_t = torch.cat(va_pred_parts)
            val_loss = criterion(va_pred_t, va_y_t, va_w_t).item()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            marker = " <- best"
        else:
            patience_cnt += 1
            marker = f" (patience {patience_cnt}/{patience})"
        if is_best or patience_cnt >= patience or epoch % 5 == 0 or epoch == epochs:
            log(f"  [{label}] epoch {epoch:2d}/{epochs}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}{marker}")
        if patience_cnt >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"[{label} seed={seed}] val_lossが一度も改善しませんでした")
    model.load_state_dict(best_state)
    model.eval()
    log(f"[+] [{label} seed={seed}] 学習完了 best_val_loss={best_val_loss:.5f}")
    return model
