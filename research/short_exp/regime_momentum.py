"""市場モメンタムレジーム(LOW/MID/HIGH)のlook-ahead無しラベル(2026-09-24追加、
ユーザー提案「レジーム別ペアワイズ損失を空売り専用に再検証」)。

stage3_exp.diagnostics.compute_expanding_vol_regime_labels(実現ボラティリティの
拡大窓3分位)と同じ拡大窓分位点パターンを、ボラティリティではなく市場モメンタム
(20日NK225リターン)に適用したもの。動機: [[walk_forward_validation_2026-09-24]]で
判明した「市場の上昇モメンタムが強いほど空売りモデルのPFが悪化する」という
空売り特有の依存性は、既存のボラティリティレジーム(regime_risk_model系で使われている
実現ボラの高低)とは別軸——ボラが低くても強い上昇トレンド中なら空売りには不利、
という状況を区別できない。ペアワイズ損失の比較を同一モメンタムレジーム内に限定すれば、
レジームを跨いだ「地合いの強さの差」に由来するノイズを学習から除去できるという仮説。
"""
import pandas as pd


def compute_expanding_momentum_regime_labels(macro_pool_df, window=20, min_periods=120):
    """date->'LOW'/'MID'/'HIGH'/'UNKNOWN'のpd.Series。momentum = NK_Close.pct_change(window)
    (20日リターン、符号付き)を拡大窓の1/3・2/3分位点で3値化する。HIGH=上昇モメンタムが
    強い(空売りに不利と判明した局面)、LOW=弱い/下落気味。最初のmin_periods日は
    分位点が定まらないため'UNKNOWN'(NearPairRankingLossRegimeAwareのペアタグ付けでは
    自動的に-1=常に除外扱いになる、既存のボラregimeと同じ挙動)。"""
    momentum = macro_pool_df["NK_Close"].pct_change(window)
    q1_exp = momentum.expanding(min_periods=min_periods).quantile(1 / 3)
    q2_exp = momentum.expanding(min_periods=min_periods).quantile(2 / 3)
    known = q1_exp.notna() & q2_exp.notna()
    labels = pd.Series("UNKNOWN", index=macro_pool_df.index)
    labels[known & (momentum < q1_exp)] = "LOW"
    labels[known & (momentum >= q1_exp) & (momentum < q2_exp)] = "MID"
    labels[known & (momentum >= q2_exp)] = "HIGH"
    return labels
