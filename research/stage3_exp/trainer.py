"""Model training loop."""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile as torch_profile, ProfilerActivity
from training.train_model_v8_exp import SameTickerBatchSampler
from .config import (PAIR_MODE, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, MAX_PAIR_GAP,
    USE_REGIME_AWARE_LOSS, AMP_DTYPE, USE_AMP)
from .runtime import DEVICE, log, set_seed
from .losses import NearPairRankingLossRegimeAware
from .samplers import reorder_dataset_by_tidx, build_same_date_ranges, validate_same_date_ranges
from .model_factory import build_model

def train_model(
    seed,
    tr_data,
    va_data,
    s_cols,
    m_cols,
    label,
    profile_this_call=False,
):
    set_seed(seed)

    # cross-sectionalモードでは、同じtidxの日付サンプルを最初に1回だけ
    # 連続配置する。これにより各epochでは日付バッチの順番だけをshuffleし、
    # バッチ取得自体は連続sliceで行える。
    #
    # same_tickerモードは従来のSameTickerBatchSamplerとindex_select方式を
    # そのまま維持する。
    if PAIR_MODE == "cross_sectional":
        tr_data = reorder_dataset_by_tidx(tr_data)
        va_data = reorder_dataset_by_tidx(va_data)
    (
        tr_x_s,
        tr_x_m,
        tr_y,
        tr_tid,
        tr_tidx,
        tr_regime,
    ) = tr_data
    (
        va_x_s,
        va_x_m,
        va_y,
        va_tid,
        va_tidx,
        va_regime,
    ) = va_data

    # Dataset/DataLoaderを経由せず、テンソル化した全データを1回だけGPUへ載せる。
    #
    # memmap入力の場合もtorch.tensor()によって通常のGPU Tensorへコピーされる。
    # read-only memmapをtorch.from_numpy()で共有せず、既存どおり安全なコピーを使う。
    tr_xs_t = torch.tensor(
        tr_x_s,
        dtype=torch.float32,
        device=DEVICE,
    )
    tr_xm_t = torch.tensor(
        tr_x_m,
        dtype=torch.float32,
        device=DEVICE,
    )
    tr_y_t = torch.tensor(
        tr_y,
        dtype=torch.float32,
        device=DEVICE,
    )
    tr_tidx_t = torch.tensor(
        tr_tidx,
        dtype=torch.long,
        device=DEVICE,
    )

    # USE_REGIME_AWARE_LOSS=Falseでも既存コードとの構造を単純に保つため
    # 一度GPUへ載せる。ただしepochごとの並べ替えは行わない。
    tr_regime_t = torch.tensor(
        tr_regime,
        dtype=torch.long,
        device=DEVICE,
    )

    va_xs_t = torch.tensor(
        va_x_s,
        dtype=torch.float32,
        device=DEVICE,
    )
    va_xm_t = torch.tensor(
        va_x_m,
        dtype=torch.float32,
        device=DEVICE,
    )
    va_y_t = torch.tensor(
        va_y,
        dtype=torch.float32,
        device=DEVICE,
    )
    va_tidx_t = torch.tensor(
        va_tidx,
        dtype=torch.long,
        device=DEVICE,
    )
    va_regime_t = torch.tensor(
        va_regime,
        dtype=torch.long,
        device=DEVICE,
    )

    if PAIR_MODE == "cross_sectional":
        # データは上でtidx順に並べ替え済みなので、同一日の各バッチは
        # (start, end)の連続範囲として表現できる。
        train_ranges = build_same_date_ranges(
            tr_tidx,
            TRAIN_BATCH_SIZE,
        )
        val_ranges = build_same_date_ranges(
            va_tidx,
            VAL_BATCH_SIZE,
        )

        # 実装上の前提を学習開始前に検証する。
        validate_same_date_ranges(
            tr_tidx,
            train_ranges,
            f"{label}/train",
        )
        validate_same_date_ranges(
            va_tidx,
            val_ranges,
            f"{label}/validation",
        )

        train_sampler = None
        val_batches = None

        log(
            f"  [{label}] cross-sectional連続バッチ: "
            f"train={len(train_ranges)} val={len(val_ranges)}"
        )
    else:
        # same_ticker側は従来方式を維持する。
        train_ranges = None
        val_ranges = None

        train_sampler = SameTickerBatchSampler(
            tr_tid,
            TRAIN_BATCH_SIZE,
            shuffle=True,
        )
        val_batches = list(
            SameTickerBatchSampler(
                va_tid,
                VAL_BATCH_SIZE,
                shuffle=False,
            )
        )

    model = build_model(s_cols, m_cols)

    # pair数加重・ゼロペアバッチ除外のため、regime制約の有無に関わらず
    # NearPairRankingLossRegimeAwareを使用する。
    #
    # この修正ではcriterion自体は変更していないため、旧実装と同じく
    # B×Bの両方向ペアを計算する。上三角化は別変更として検証する。
    criterion = NearPairRankingLossRegimeAware(
        max_gap=MAX_PAIR_GAP,
        cross_sectional=(PAIR_MODE == "cross_sectional"),
    )

    # CUDAの場合はfused AdamWを使用する。
    fused_adamw = DEVICE.type == "cuda"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0002,
        weight_decay=3e-2,
        fused=fused_adamw,
    )

    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16
    amp_enabled = USE_AMP and DEVICE.type == "cuda"

    # fp16だけGradScalerを使う。bf16はfp32と同じ指数範囲なので使用しない。
    use_scaler = amp_enabled and AMP_DTYPE == "fp16"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
    )

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

            prof = torch_profile(
                activities=activities,
            )
            prof.__enter__()

        model.train()

        # ------------------------------------------------------------
        # epochのバッチ順序を構築
        # ------------------------------------------------------------
        if PAIR_MODE == "cross_sectional":
            # データ本体はtidx順に固定配置したまま、日付バッチの処理順だけを
            # epochごとにshuffleする。
            #
            # 各バッチ内のサンプル順序は固定だが、ランキング損失はバッチ内の
            # 全ペア比較なので、サンプル順序自体はlossへ影響しない。
            epoch_batches = train_ranges.copy()
            random.shuffle(epoch_batches)

            # cross-sectional側では全件index_selectを実行しない。
            xs_shuf = None
            xm_shuf = None
            y_shuf = None
            tidx_shuf = None
            regime_shuf = None
        else:
            # same_ticker側は従来方式を維持する。
            #
            # SameTickerBatchSamplerが決めたバッチ構成を1本へ連結して
            # GPU上で全体を1回だけindex_selectし、個々のバッチは
            # 連続sliceで取り出す。
            epoch_batches = list(train_sampler)

            flat_idx = [
                sample_idx
                for batch_indices in epoch_batches
                for sample_idx in batch_indices
            ]

            flat_idx_t = torch.tensor(
                flat_idx,
                dtype=torch.long,
                device=DEVICE,
            )

            xs_shuf = tr_xs_t.index_select(
                0,
                flat_idx_t,
            )
            xm_shuf = tr_xm_t.index_select(
                0,
                flat_idx_t,
            )
            y_shuf = tr_y_t.index_select(
                0,
                flat_idx_t,
            )
            tidx_shuf = tr_tidx_t.index_select(
                0,
                flat_idx_t,
            )

            if USE_REGIME_AWARE_LOSS:
                regime_shuf = tr_regime_t.index_select(
                    0,
                    flat_idx_t,
                )
            else:
                regime_shuf = None

        # エポック全体のtrain_lossは、有効ペア数による加重平均。
        total_tr = torch.zeros(
            (),
            device=DEVICE,
        )
        total_tr_pairs = 0
        n_tr_batches = 0

        # same_ticker側で、連結済みTensorから各バッチを切り出す位置。
        offset = 0

        for batch_spec in epoch_batches:
            if PAIR_MODE == "cross_sectional":
                start, end = batch_spec

                # 全て連続sliceなので、コピー無しのviewとして取得できる。
                b_xs = tr_xs_t[start:end]
                b_xm = tr_xm_t[start:end]
                b_y = tr_y_t[start:end]
                b_tidx = tr_tidx_t[start:end]

                if USE_REGIME_AWARE_LOSS:
                    b_regime = tr_regime_t[start:end]
                else:
                    b_regime = None
            else:
                batch_indices = batch_spec
                batch_size = len(batch_indices)
                next_offset = offset + batch_size

                b_xs = xs_shuf[offset:next_offset]
                b_xm = xm_shuf[offset:next_offset]
                b_y = y_shuf[offset:next_offset]
                b_tidx = tidx_shuf[offset:next_offset]

                if USE_REGIME_AWARE_LOSS:
                    b_regime = regime_shuf[offset:next_offset]
                else:
                    b_regime = None

                offset = next_offset

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                logits = model(
                    b_xs,
                    b_xm,
                )

                loss, n_pairs = criterion(
                    logits,
                    b_y,
                    b_tidx,
                    b_regime,
                )

            # 有効ペアがないバッチはbackward、optimizer.step、
            # epoch平均の全てから除外する。
            if n_pairs == 0:
                continue

            optimizer.zero_grad(
                set_to_none=True,
            )

            if use_scaler:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            total_tr += loss.detach() * n_pairs
            total_tr_pairs += n_pairs
            n_tr_batches += 1

        if total_tr_pairs > 0:
            train_loss = (total_tr / total_tr_pairs).item()
        else:
            train_loss = float("inf")

        if do_profile:
            prof.__exit__(
                None,
                None,
                None,
            )

            sort_key = (
                "self_cuda_time_total"
                if torch.cuda.is_available()
                else "self_cpu_time_total"
            )

            log(
                f"[Profiler] [{label}] "
                f"epoch1 学習ループ"
                f"({n_tr_batches}バッチ)"
                f"処理時間トップ15:"
            )

            print(
                prof.key_averages().table(
                    sort_by=sort_key,
                    row_limit=15,
                )
            )

        # ------------------------------------------------------------
        # validation
        # ------------------------------------------------------------
        model.eval()

        total_va = torch.zeros(
            (),
            device=DEVICE,
        )
        total_va_pairs = 0
        n_batches = 0

        if PAIR_MODE == "cross_sectional":
            validation_iterator = val_ranges
        else:
            validation_iterator = val_batches

        with torch.no_grad():
            for batch_spec in validation_iterator:
                if PAIR_MODE == "cross_sectional":
                    start, end = batch_spec

                    # validation側も連続sliceで取得する。
                    b_xs = va_xs_t[start:end]
                    b_xm = va_xm_t[start:end]
                    b_y = va_y_t[start:end]
                    b_tidx = va_tidx_t[start:end]

                    if USE_REGIME_AWARE_LOSS:
                        b_regime = va_regime_t[start:end]
                    else:
                        b_regime = None
                else:
                    batch_idx = batch_spec

                    # same_ticker側は従来どおりfancy indexing。
                    b_xs = va_xs_t[batch_idx]
                    b_xm = va_xm_t[batch_idx]
                    b_y = va_y_t[batch_idx]
                    b_tidx = va_tidx_t[batch_idx]

                    if USE_REGIME_AWARE_LOSS:
                        b_regime = va_regime_t[batch_idx]
                    else:
                        b_regime = None

                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    logits = model(
                        b_xs,
                        b_xm,
                    )

                    loss, n_pairs = criterion(
                        logits,
                        b_y,
                        b_tidx,
                        b_regime,
                    )

                total_va += loss * n_pairs
                total_va_pairs += n_pairs
                n_batches += 1

        if n_batches == 0:
            log(
                f"  [{label}] "
                f"epoch {epoch:2d}/{epochs}: "
                f"[!] 警告 検証バッチが0件です"
                f"(regime_filter等でval側の"
                f"サンプルが無くなっている可能性)"
            )
        elif total_va_pairs == 0:
            log(
                f"  [{label}] "
                f"epoch {epoch:2d}/{epochs}: "
                f"[!] 警告 検証バッチは"
                f"{n_batches}件あるが"
                f"有効ペアが1つもありません"
            )

        if total_va_pairs > 0:
            val_loss = (total_va / total_va_pairs).item()
        else:
            val_loss = float("inf")

        if np.isnan(val_loss) or np.isnan(train_loss):
            log(
                f"  [{label}] "
                f"epoch {epoch:2d}/{epochs}: "
                f"[!] 警告 train_loss/val_lossに"
                f"NaNが出ています"
                f"(train_loss={train_loss}, "
                f"val_loss={val_loss})"
                f"。学習が発散している可能性"
            )

        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

            best_state = {
                key: value.clone() for key, value in model.state_dict().items()
            }

            patience_cnt = 0
            marker = " <- best"
        else:
            patience_cnt += 1
            marker = f" (patience " f"{patience_cnt}/{patience})"

        is_last = patience_cnt >= patience or epoch == epochs

        # new best、5epochごと、最終epochだけ出力する。
        if is_best or is_last or epoch % 5 == 0:
            log(
                f"  [{label}] "
                f"epoch {epoch:2d}/{epochs}: "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
                f"{marker}"
            )

        if patience_cnt >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            f"[{label} seed={seed}] "
            f"学習全エポックでval_lossが"
            f"一度も改善しませんでした"
            f"(best_state=None)。"
            f"val側のサンプル数が0、または"
            f"val_lossが常にNaN/infの"
            f"可能性があります。"
            f"regime_filter/特徴量構成/"
            f"学習データ量を確認してください。"
        )

    model.load_state_dict(best_state)
    model.eval()

    log(f"[+] [{label} seed={seed}] " f"学習完了 " f"best_val_loss={best_val_loss:.4f}")

    return model
