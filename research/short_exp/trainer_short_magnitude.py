"""stage3_exp/trainer.py::train_modelのコピー(2026-09-24追加)。criterionだけ
NearPairRankingLossMagnitudeWeighted(|ret_pct差|で重み付け、外れ値はmax_weightで
クリップ)に差し替えている。それ以外の学習ループ(PAIR_MODE分岐・AMP・early stopping等)は
trainer.pyと完全に同一——共有のstage3_exp/trainer.py自体は変更しない
([[feedback_keep_rejected_experiment_code]]の延長、ロング側への影響を避けるための
意図的な複製、trainer_short_asym.pyと同じパターン)。
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile as torch_profile, ProfilerActivity
from training.train_model_v8_exp import SameTickerBatchSampler
from stage3_exp.config import (PAIR_MODE, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, MAX_PAIR_GAP,
    USE_REGIME_AWARE_LOSS, AMP_DTYPE, USE_AMP, EV_WIN_WEIGHT, EV_STOP_WEIGHT)
from stage3_exp.runtime import DEVICE, log, set_seed
from stage3_exp.samplers import reorder_dataset_by_tidx, build_same_date_ranges, validate_same_date_ranges
from stage3_exp.model_factory import build_model
from .losses_short_magnitude import NearPairRankingLossMagnitudeWeighted


def train_model_short_magnitude(
    seed,
    tr_data,
    va_data,
    s_cols,
    m_cols,
    label,
    max_weight=3.0,
    ev_win_weight=EV_WIN_WEIGHT,
    ev_stop_weight=EV_STOP_WEIGHT,
    use_regime_aware=False,
    profile_this_call=False,
):
    set_seed(seed)

    if PAIR_MODE == "cross_sectional":
        tr_data = reorder_dataset_by_tidx(tr_data)
        va_data = reorder_dataset_by_tidx(va_data)
    (tr_x_s, tr_x_m, tr_y, tr_tid, tr_tidx, tr_regime) = tr_data
    (va_x_s, va_x_m, va_y, va_tid, va_tidx, va_regime) = va_data

    tr_xs_t = torch.tensor(tr_x_s, dtype=torch.float32, device=DEVICE)
    tr_xm_t = torch.tensor(tr_x_m, dtype=torch.float32, device=DEVICE)
    tr_y_t = torch.tensor(tr_y, dtype=torch.float32, device=DEVICE)
    tr_tidx_t = torch.tensor(tr_tidx, dtype=torch.long, device=DEVICE)
    tr_regime_t = torch.tensor(tr_regime, dtype=torch.long, device=DEVICE)

    va_xs_t = torch.tensor(va_x_s, dtype=torch.float32, device=DEVICE)
    va_xm_t = torch.tensor(va_x_m, dtype=torch.float32, device=DEVICE)
    va_y_t = torch.tensor(va_y, dtype=torch.float32, device=DEVICE)
    va_tidx_t = torch.tensor(va_tidx, dtype=torch.long, device=DEVICE)
    va_regime_t = torch.tensor(va_regime, dtype=torch.long, device=DEVICE)

    if PAIR_MODE == "cross_sectional":
        train_ranges = build_same_date_ranges(tr_tidx, TRAIN_BATCH_SIZE)
        val_ranges = build_same_date_ranges(va_tidx, VAL_BATCH_SIZE)
        validate_same_date_ranges(tr_tidx, train_ranges, f"{label}/train")
        validate_same_date_ranges(va_tidx, val_ranges, f"{label}/validation")
        train_sampler = None
        val_batches = None
        log(f"  [{label}] cross-sectional連続バッチ: train={len(train_ranges)} val={len(val_ranges)}")
    else:
        train_ranges = None
        val_ranges = None
        train_sampler = SameTickerBatchSampler(tr_tid, TRAIN_BATCH_SIZE, shuffle=True)
        val_batches = list(SameTickerBatchSampler(va_tid, VAL_BATCH_SIZE, shuffle=False))

    model = build_model(s_cols, m_cols)

    criterion = NearPairRankingLossMagnitudeWeighted(
        ev_win_weight=ev_win_weight,
        ev_stop_weight=ev_stop_weight,
        max_gap=MAX_PAIR_GAP,
        cross_sectional=(PAIR_MODE == "cross_sectional"),
        max_weight=max_weight,
    )

    fused_adamw = DEVICE.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=3e-2, fused=fused_adamw)

    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16
    amp_enabled = USE_AMP and DEVICE.type == "cuda"
    use_scaler = amp_enabled and AMP_DTYPE == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_val_loss = float("inf")
    best_state = None
    patience_cnt = 0
    patience = 7
    epochs = 30

    for epoch in range(1, epochs + 1):
        do_profile = profile_this_call and epoch == 1
        prof = None
        if do_profile:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            prof = torch_profile(activities=activities)
            prof.__enter__()

        model.train()

        if PAIR_MODE == "cross_sectional":
            epoch_batches = train_ranges.copy()
            random.shuffle(epoch_batches)
            xs_shuf = xm_shuf = y_shuf = tidx_shuf = regime_shuf = None
        else:
            epoch_batches = list(train_sampler)
            flat_idx = [i for b in epoch_batches for i in b]
            flat_idx_t = torch.tensor(flat_idx, dtype=torch.long, device=DEVICE)
            xs_shuf = tr_xs_t.index_select(0, flat_idx_t)
            xm_shuf = tr_xm_t.index_select(0, flat_idx_t)
            y_shuf = tr_y_t.index_select(0, flat_idx_t)
            tidx_shuf = tr_tidx_t.index_select(0, flat_idx_t)
            regime_shuf = tr_regime_t.index_select(0, flat_idx_t) if use_regime_aware else None

        total_tr = torch.zeros((), device=DEVICE)
        total_tr_pairs = 0
        n_tr_batches = 0
        offset = 0

        for batch_spec in epoch_batches:
            if PAIR_MODE == "cross_sectional":
                start, end = batch_spec
                b_xs = tr_xs_t[start:end]
                b_xm = tr_xm_t[start:end]
                b_y = tr_y_t[start:end]
                b_tidx = tr_tidx_t[start:end]
                b_regime = tr_regime_t[start:end] if use_regime_aware else None
            else:
                batch_indices = batch_spec
                batch_size = len(batch_indices)
                next_offset = offset + batch_size
                b_xs = xs_shuf[offset:next_offset]
                b_xm = xm_shuf[offset:next_offset]
                b_y = y_shuf[offset:next_offset]
                b_tidx = tidx_shuf[offset:next_offset]
                b_regime = regime_shuf[offset:next_offset] if use_regime_aware else None
                offset = next_offset

            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(b_xs, b_xm)
                loss, n_pairs = criterion(logits, b_y, b_tidx, b_regime)

            if n_pairs == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_tr += loss.detach() * n_pairs
            total_tr_pairs += n_pairs
            n_tr_batches += 1

        train_loss = (total_tr / total_tr_pairs).item() if total_tr_pairs > 0 else float("inf")

        if do_profile:
            prof.__exit__(None, None, None)
            sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
            log(f"[Profiler] [{label}] epoch1 学習ループ({n_tr_batches}バッチ)処理時間トップ15:")
            print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

        model.eval()
        total_va = torch.zeros((), device=DEVICE)
        total_va_pairs = 0
        n_batches = 0
        validation_iterator = val_ranges if PAIR_MODE == "cross_sectional" else val_batches

        with torch.no_grad():
            for batch_spec in validation_iterator:
                if PAIR_MODE == "cross_sectional":
                    start, end = batch_spec
                    b_xs = va_xs_t[start:end]
                    b_xm = va_xm_t[start:end]
                    b_y = va_y_t[start:end]
                    b_tidx = va_tidx_t[start:end]
                    b_regime = va_regime_t[start:end] if use_regime_aware else None
                else:
                    batch_idx = batch_spec
                    b_xs = va_xs_t[batch_idx]
                    b_xm = va_xm_t[batch_idx]
                    b_y = va_y_t[batch_idx]
                    b_tidx = va_tidx_t[batch_idx]
                    b_regime = va_regime_t[batch_idx] if use_regime_aware else None

                with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(b_xs, b_xm)
                    loss, n_pairs = criterion(logits, b_y, b_tidx, b_regime)

                total_va += loss * n_pairs
                total_va_pairs += n_pairs
                n_batches += 1

        if n_batches == 0:
            log(f"  [{label}] epoch {epoch:2d}/{epochs}: [!] 警告 検証バッチが0件です")
        elif total_va_pairs == 0:
            log(f"  [{label}] epoch {epoch:2d}/{epochs}: [!] 警告 有効ペアが1つもありません")

        val_loss = (total_va / total_va_pairs).item() if total_va_pairs > 0 else float("inf")

        if np.isnan(val_loss) or np.isnan(train_loss):
            log(f"  [{label}] epoch {epoch:2d}/{epochs}: [!] 警告 train_loss/val_lossにNaNが出ています"
                f"(train_loss={train_loss}, val_loss={val_loss})。学習が発散している可能性")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            marker = " <- best"
        else:
            patience_cnt += 1
            marker = f" (patience {patience_cnt}/{patience})"

        is_last = patience_cnt >= patience or epoch == epochs
        if is_best or is_last or epoch % 5 == 0:
            log(f"  [{label}] epoch {epoch:2d}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}{marker}")

        if patience_cnt >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            f"[{label} seed={seed}] 学習全エポックでval_lossが一度も改善しませんでした(best_state=None)。"
        )

    model.load_state_dict(best_state)
    model.eval()
    log(f"[+] [{label} seed={seed}] 学習完了 best_val_loss={best_val_loss:.4f}")
    return model
