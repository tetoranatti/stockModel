# modules/regime_momentum.py
# ロング/ショート2モデルのレジーム連動資金配分(2026-09-24追加)。
# research/short_exp/regime_momentum.py::compute_expanding_momentum_regime_labels、および
# research/regime_blend_walk_forward.pyのgate_strongスキーム(5-fold walk-forwardで
# 固定50/50・ロング単体より優位と確認済み、[[regime_gated_long_short_blend_2026-09-24]])を
# 本番向けに移植したもの。
#
# ロジックはresearch版と完全同一(NK225の20日リターン、拡大窓の1/3・2/3分位点で
# LOW/MID/HIGH/UNKNOWNに3値化、look-ahead無し)。拡大窓の分位点は「その日までの全履歴」で
# 決まるため、macro_dfとして渡すデータの開始日が研究側と厳密に一致しなくても、
# 十分な履歴(概ね120営業日以上)さえあれば直近のzone判定はほぼ揺れない。
import pandas as pd

MOMENTUM_WINDOW = 20
MOMENTUM_MIN_PERIODS = 120

# gate_strong(HIGH:0.0 / MID:0.5 / LOW:1.0)。walk-forward検証(5-fold)でgate_mildより
# 優位だった配分(CAGR平均109.7% vs ロング単体104.5%、MaxDD平均-4.0% vs -9.0%、
# 最悪foldのMaxDD -26.6%→-7.5%)。HIGH regimeではショート資金0・ロング全額、
# LOW regimeではロング資金0・ショート全額という強めのゲートである点に注意
# ([[regime_gated_long_short_blend_2026-09-24]]参照、本番投入前にこの極端さを許容する
# 前提で採用)。
GATE_SHORT_WEIGHT = {"HIGH": 0.0, "MID": 0.5, "LOW": 1.0, "UNKNOWN": 0.5}


def compute_expanding_momentum_regime_labels(macro_df, window=MOMENTUM_WINDOW, min_periods=MOMENTUM_MIN_PERIODS):
    """date->'LOW'/'MID'/'HIGH'/'UNKNOWN'のpd.Series。momentum = NK_Close.pct_change(window)
    (20日リターン、符号付き)を拡大窓の1/3・2/3分位点で3値化する。HIGH=上昇モメンタムが
    強い(空売りに不利、ロングに有利)、LOW=弱い/下落気味(逆)。
    research/short_exp/regime_momentum.pyと同一ロジック。"""
    momentum = macro_df["NK_Close"].pct_change(window)
    q1_exp = momentum.expanding(min_periods=min_periods).quantile(1 / 3)
    q2_exp = momentum.expanding(min_periods=min_periods).quantile(2 / 3)
    known = q1_exp.notna() & q2_exp.notna()
    labels = pd.Series("UNKNOWN", index=macro_df.index)
    labels[known & (momentum < q1_exp)] = "LOW"
    labels[known & (momentum >= q1_exp) & (momentum < q2_exp)] = "MID"
    labels[known & (momentum >= q2_exp)] = "HIGH"
    return labels


def determine_capital_split(macro_df, total_capital, gate=GATE_SHORT_WEIGHT):
    """macro_df(NK_Close列必須、直近日を含む)から最新のモメンタムregimeを判定し、
    ロング/ショートそれぞれに配分する資金額を返す。"""
    labels = compute_expanding_momentum_regime_labels(macro_df)
    zone = str(labels.iloc[-1])
    w_short = gate.get(zone, 0.5)
    w_long = 1.0 - w_short
    return {
        "zone": zone,
        "w_long": w_long,
        "w_short": w_short,
        "capital_long": round(total_capital * w_long),
        "capital_short": round(total_capital * w_short),
    }
