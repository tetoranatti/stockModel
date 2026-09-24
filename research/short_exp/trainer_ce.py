"""決済理由ラベル(exit_class: 0=Stop/1=Hold/2=Win)向けの交差エントロピー学習ループ
(2026-09-24追加、ユーザー提案「TP/SL到達順序型ラベル」→トレーリング維持の決済理由版+
交差エントロピー損失)。trainer_alpha.py(Weighted Huber回帰)と同じく、NearPairRankingLoss
系(stage3_exp.trainer.train_model)が前提とする同日クロスセクショナルバッチ/ペアリングは
不要——サンプル単位の3クラス分類なのでランダムシャッフルしたミニバッチで良い。

build_model(stage3_exp.model_factory)は既にnum_classes=3固定でlogitsを返す設計
(既存のp_win=probs[:,2]/p_stop=probs[:,0]という命名規約はこのため)ため、モデル本体・
推論(run_backtest_inference_ensemble)は無変更で流用できる——exit_classの0=Stop/2=Winが
そのままこのクラスindexと一致するように定義したのは意図的([[targets_short.py]]参照)。

クラス不均衡対策として、学習データのクラス頻度からinverse-frequency重みを計算し
nn.CrossEntropyLossに渡す(Holdクラスが多数派になりやすいトレーリングストップの性質上、
均等重みだと少数派Stop/Winの学習が弱くなる懸念があるため)。"""
import torch
import torch.nn as nn
from stage3_exp.runtime import DEVICE, log, set_seed
from stage3_exp.config import TRAIN_BATCH_SIZE, VAL_BATCH_SIZE
from stage3_exp.model_factory import build_model


def train_model_ce(seed, tr_data, va_data, s_cols, m_cols, label):
    set_seed(seed)
    tr_x_s, tr_x_m, tr_y = tr_data[0], tr_data[1], tr_data[2]
    va_x_s, va_x_m, va_y = va_data[0], va_data[1], va_data[2]

    tr_xs_t = torch.tensor(tr_x_s, dtype=torch.float32, device=DEVICE)
    tr_xm_t = torch.tensor(tr_x_m, dtype=torch.float32, device=DEVICE)
    tr_y_t = torch.tensor(tr_y, dtype=torch.long, device=DEVICE)

    va_xs_t = torch.tensor(va_x_s, dtype=torch.float32, device=DEVICE)
    va_xm_t = torch.tensor(va_x_m, dtype=torch.float32, device=DEVICE)
    va_y_t = torch.tensor(va_y, dtype=torch.long, device=DEVICE)

    class_counts = torch.bincount(tr_y_t, minlength=3).float()
    class_weight = (class_counts.sum() / (3.0 * class_counts.clamp_min(1.0))).to(DEVICE)
    log(f"  [{label}] クラス分布(train): Stop={int(class_counts[0])} Hold={int(class_counts[1])} "
        f"Win={int(class_counts[2])} -> weight={[round(w, 3) for w in class_weight.tolist()]}")

    model = build_model(s_cols, m_cols)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
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
            b_xs, b_xm, b_y = tr_xs_t[idx], tr_xm_t[idx], tr_y_t[idx]
            logits = model(b_xs, b_xm)
            loss = criterion(logits, b_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            va_logit_parts = []
            for start in range(0, len(va_y), VAL_BATCH_SIZE):
                b_xs = va_xs_t[start:start + VAL_BATCH_SIZE]
                b_xm = va_xm_t[start:start + VAL_BATCH_SIZE]
                va_logit_parts.append(model(b_xs, b_xm))
            va_logits_t = torch.cat(va_logit_parts)
            val_loss = criterion(va_logits_t, va_y_t).item()

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
