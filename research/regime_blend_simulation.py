"""ロング/ショート2モデルのレジーム連動配分シミュレーション(2026-09-24追加)。

regime_blend_short_long_correlation.pyが保存した日次資金曲線
(research/cache/_cache_long_short_equity_curves.pkl、ロングbaseline top5・
ショートv2 top5、2025-12-19〜2026-09-18の183日)を使い、
short_exp.regime_momentum.compute_expanding_momentum_regime_labels
(NK225の20日モメンタム、拡大窓3分位、look-ahead無し)でゲーティングした
動的配分と、固定50/50配分・ロング単体・ショート単体を比較する。

[[walk_forward_validation_2026-09-24]]で判明した「上昇モメンタムが強いほど
空売りモデルのPFが悪化する」という依存性を根拠に、HIGH regimeでショート比重を
下げ、LOW regimeで上げるゲート関数を試す。

注意: 単一の183日サンプル(2025-12-19〜2026-09-18、単一固定split)での
検証であり、walk-forwardではない([[backtest_methodology_2026-09-23]]と同じ
制約)。ゲート重みもこのサンプルに対する事後決め打ちの複数案を比較しているだけで、
別期間での頑健性はまだ確認していない。

実行方法:
    python research/regime_blend_simulation.py
"""
import sys
import os

sys.path.insert(0, r"F:\stockModel")
sys.path.insert(0, r"F:\stockModel\research")

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd


def compute_max_drawdown_from_curve(equity_curve):
    peak, max_dd = -float("inf"), 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = min(max_dd, (eq - peak) / peak)
    return max_dd


def compute_cagr_from_curve(equity_curve, n_days):
    if n_days <= 0 or equity_curve[-1] <= 0:
        return float("nan")
    total_ret = equity_curve[-1]
    years = n_days / 365.25
    return total_ret ** (1.0 / years) - 1.0 if years > 0 else float("nan")


def build_curve(daily_returns):
    curve = [1.0]
    for r in daily_returns:
        curve.append(curve[-1] * (1.0 + r))
    return curve[1:]


def main():
    import stage3_exp.experiment_context as m
    from short_exp.regime_momentum import compute_expanding_momentum_regime_labels

    eq_path = r"F:\stockModel\research\cache\_cache_long_short_equity_curves.pkl"
    eq_df = pd.read_pickle(eq_path)
    print(f"[+] 資金曲線読み込み: {eq_path} ({eq_df.index.min().date()} 〜 {eq_df.index.max().date()}, {len(eq_df)}日)")

    ret_long = eq_df["eq_long_top5"].pct_change().dropna()
    ret_short = eq_df["eq_short_top5"].pct_change().dropna()
    common_idx = ret_long.index.intersection(ret_short.index)
    ret_long = ret_long.reindex(common_idx)
    ret_short = ret_short.reindex(common_idx)
    print(f"[+] 共通リターン系列: {len(common_idx)}日")

    # --- モメンタムレジームラベル(look-ahead無し、拡大窓3分位) ---
    with open(m.UNIVERSE_PATH, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    macro_pool_df = m.augment_macro_pool_df(m.get_full_macro_pool_df(tickers, force_rebuild=False))
    regime_labels = compute_expanding_momentum_regime_labels(macro_pool_df)
    regime_aligned = regime_labels.reindex(common_idx).ffill()

    print("\n[regime分布(共通期間内)]")
    print(regime_aligned.value_counts())

    print("\n[regime別 日次平均リターン(素の観測、参考値)]")
    for regime in ["LOW", "MID", "HIGH"]:
        mask = regime_aligned == regime
        if mask.sum() == 0:
            print(f"  {regime}: n=0")
            continue
        print(f"  {regime}: n={mask.sum():3d} ロング平均={ret_long[mask].mean():+.4f} "
              f"ショート平均={ret_short[mask].mean():+.4f}")

    n_days = (common_idx.max() - common_idx.min()).days

    def eval_scheme(name, w_short_series):
        w_short = w_short_series.reindex(common_idx)
        w_long = 1.0 - w_short
        blended = w_long * ret_long + w_short * ret_short
        curve = build_curve(blended.values)
        dd = compute_max_drawdown_from_curve(curve)
        cagr = compute_cagr_from_curve(curve, n_days)
        print(f"  {name:<28s} CAGR={cagr*100:6.1f}%  MaxDD={dd*100:6.1f}%  最終資産={curve[-1]:.3f}")
        return dict(name=name, cagr=cagr, max_dd=dd, final=curve[-1])

    print("\n" + "=" * 90)
    print("【配分方式ごとのCAGR/MaxDD比較】")
    print("=" * 90)

    results = []
    results.append(eval_scheme("ロング単体(100%)", pd.Series(0.0, index=common_idx)))
    results.append(eval_scheme("ショート単体(100%)", pd.Series(1.0, index=common_idx)))
    results.append(eval_scheme("固定50/50", pd.Series(0.5, index=common_idx)))

    gate_schemes = {
        "レジームゲート(緩め: 0.3/0.5/0.7)": {"HIGH": 0.3, "MID": 0.5, "LOW": 0.7},
        "レジームゲート(中庸: 0.2/0.5/0.8)": {"HIGH": 0.2, "MID": 0.5, "LOW": 0.8},
        "レジームゲート(強め: 0.0/0.5/1.0)": {"HIGH": 0.0, "MID": 0.5, "LOW": 1.0},
        "レジームゲート(HIGHのみ抑制: 0.1/0.5/0.5)": {"HIGH": 0.1, "MID": 0.5, "LOW": 0.5},
    }
    for name, gate in gate_schemes.items():
        w_short_series = regime_aligned.map(gate).fillna(0.5)
        results.append(eval_scheme(name, w_short_series))

    out_df = pd.DataFrame(results)
    out_path = r"F:\stockModel\research\cache\_cache_regime_blend_sim_results.pkl"
    out_df.to_pickle(out_path)
    print(f"\n[+] 結果を保存: {out_path}")


if __name__ == "__main__":
    main()
