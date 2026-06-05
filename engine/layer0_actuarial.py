"""
Layer 0 精算引擎 — 实战实现

直接在 prices_full.pkl + csi300_regime.pkl 上运行。
输入信号定义 → 找到历史同类信号 → 输出精算报告。

验收标准：
  ✓ 零假设区分力: 信号分布 vs Bootstrap 10K次 → KS p<0.01, 中位差>2pp
  ✓ 前向稳定性: 4轮滚动窗口，每轮中位偏差≤±1.5pp
  ✓ Bootstrap鲁棒性: VaR置信区间宽度≤±3pp
  ✓ 分布形状不撒谎: 偏度≤0.5 (真实世界罕见正偏)
"""
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 常量 ──────────────────────────────────────────────
CACHE_DIR = Path("data/cache")
DEFAULT_FORWARD_DAYS = 40
MIN_SAMPLES = 30
BOOTSTRAP_N = 10_000

WALKFORWARD = [
    ("2019-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2019-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2019-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2019-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]


@dataclass
class ActuarialReport:
    """精算引擎输出"""
    signal_name: str = ""
    n_signals: int = 0
    sufficient: bool = False

    # 信号分布
    median: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0  # avg_win / |avg_loss|

    # 尾部风险
    left_tail_5: float = 0.0
    left_tail_1: float = 0.0
    right_tail_95: float = 0.0
    skewness: float = 0.0

    # 零假设检验
    null_median: float = 0.0
    median_diff: float = 0.0
    win_rate_diff: float = 0.0
    ks_stat: float = 0.0
    ks_pvalue: float = 1.0
    is_significant: bool = False

    # 前向验证
    wf_stable: bool = True
    wf_medians: List[float] = field(default_factory=list)
    wf_range: float = 0.0

    # Bootstrap鲁棒性
    median_ci: tuple = (0.0, 0.0)
    var_ci_width: float = 0.0

    # 逐年效应量
    yearly_medians: Dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        """人类可读摘要"""
        if not self.sufficient:
            return f"[{self.signal_name}] 样本不足 (n={self.n_signals}<{MIN_SAMPLES})"

        lines = [
            f"═══ {self.signal_name} (n={self.n_signals}) ═══",
            f"  中位收益: {self.median:+.2%}  胜率: {self.win_rate:.1%}  盈亏比: {self.profit_loss_ratio:.2f}",
            f"  左尾5%: {self.left_tail_5:+.2%}  左尾1%: {self.left_tail_1:+.2%}  偏度: {self.skewness:+.3f}",
            f"  vs 零假设: 中位差{self.median_diff:+.2%}  胜率差{self.win_rate_diff:+.1%}  KS p={self.ks_pvalue:.4f}",
            f"  显著性: {'[PASS] 通过' if self.is_significant else '[FAIL] 未通过'}",
            f"  前向验证: {'[PASS] 稳定' if self.wf_stable else '[FAIL] 不稳定'} (范围{self.wf_range:.2%})",
        ]
        if self.yearly_medians:
            years_str = " ".join(f"{y}:{v:+.1%}" for y, v in sorted(self.yearly_medians.items()))
            lines.append(f"  逐年中位: {years_str}")
        return "\n".join(lines)


class ActuarialEngine:
    """
    Layer 0 — 精算引擎
    
    信号定义 -> 找历史同类信号 -> 收益分布 -> 零假设 -> 效应量 -> 前向验证
    """

    def __init__(self, forward_days: int = DEFAULT_FORWARD_DAYS):
        self.forward_days = forward_days
        self.prices: Dict[str, pd.DataFrame] = {}
        self.csi300: Optional[pd.DataFrame] = None
        self._all_stocks: List[str] = []
        self._loaded = False

    def load_data(self):
        """加载核心数据"""
        t0 = time.time()
        
        # 股价数据
        with open(CACHE_DIR / "prices_full.pkl", "rb") as f:
            self.prices = pickle.load(f)
        
        # 预处理所有stock的日期列
        for code, df in self.prices.items():
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
        
        self._all_stocks = sorted(self.prices.keys())
        
        # CSI300 regime
        regime = pd.read_pickle(CACHE_DIR / "csi300_regime.pkl")
        cs = pd.read_pickle(CACHE_DIR / "csi300.pkl")
        
        # 合并: CSI300价格 + regime标签
        cs['trade_date'] = pd.to_datetime(cs['trade_date'])
        regime['trade_date'] = pd.to_datetime(regime['trade_date'])
        
        self.csi300 = cs.merge(
            regime[['trade_date', 'regime', 'ma20', 'ma60']], 
            on='trade_date', how='left', suffixes=('', '_r')
        )
        self.csi300.set_index('trade_date', inplace=True)
        
        # 填充缺失的regime: 价格>MA60 = bull
        if 'regime' not in self.csi300.columns:
            self.csi300['regime'] = 'unknown'
        self.csi300['regime'] = self.csi300['regime'].fillna('unknown')
        
        # 补全: close>ma60 → bull
        mask = (self.csi300['regime'] == 'unknown') & (self.csi300['close'] > self.csi300['ma60'])
        self.csi300.loc[mask, 'regime'] = 'bull'
        mask = (self.csi300['regime'] == 'unknown') & (self.csi300['close'] <= self.csi300['ma60'])
        self.csi300.loc[mask, 'regime'] = 'bear'
        
        self._loaded = True
        print(f"数据加载完成: {len(self._all_stocks)}只票, "
              f"CSI300 {len(self.csi300)}天, {time.time()-t0:.1f}s")
        return self

    # ── 核心: 通用信号扫描 ────────────────────────────

    def find_signals(self,
                     signal_filter: Callable[[pd.DataFrame, str, str], bool],
                     date_start: str = "2019-01-01",
                     date_end: str = "2026-06-01",
                     max_signals: int = 50000,
                     sample_stocks: int = 3000,
                     seed: int = 42) -> pd.DataFrame:
        """
        全量扫描，找出所有满足条件的信号。

        signal_filter(df, code, date) -> bool
          在date当天，code的df数据是否触发信号
        """
        rng = np.random.RandomState(seed)
        stocks = self._all_stocks.copy()
        if sample_stocks and sample_stocks < len(stocks):
            stocks = rng.choice(stocks, sample_stocks, replace=False).tolist()
        
        # 获取所有交易日
        cs_dates = self.csi300.loc[date_start:date_end].index.strftime('%Y-%m-%d')
        
        signals = []
        for code in stocks:
            df = self.prices.get(code)
            if df is None or len(df) < 120:
                continue
            
            for date_str in cs_dates:
                dt = pd.Timestamp(date_str)
                if dt not in df.index:
                    continue
                
                try:
                    if signal_filter(df, code, date_str):
                        # 计算前向收益
                        fwd_ret = self._forward_return(df, dt, self.forward_days)
                        if fwd_ret is not None:
                            regime = self._regime_at(date_str)
                            signals.append({
                                'ts_code': code,
                                'signal_date': date_str,
                                'forward_return': fwd_ret,
                                'regime': regime,
                            })
                except Exception:
                    continue
                
                if len(signals) >= max_signals:
                    break
            
            if len(signals) >= max_signals:
                break
        
        return pd.DataFrame(signals)

    def _forward_return(self, df: pd.DataFrame, entry_dt: pd.Timestamp, 
                        days: int, deduct_cost: bool = False,
                        circ_mv: float = None) -> Optional[float]:
        """entry_dt之后days个交易日的收益率。deduct_cost=扣交易成本（动态滑点）。"""
        idx = df.index.get_loc(entry_dt)
        if isinstance(idx, slice):
            return None
        if isinstance(idx, np.ndarray):
            idx = idx[0]
        
        exit_pos = min(idx + days, len(df) - 1)
        if exit_pos <= idx:
            return None
        
        entry_price = df.iloc[idx]['close']
        exit_price = df.iloc[exit_pos]['close']
        
        if entry_price <= 0:
            return None
        
        gross_ret = (exit_price - entry_price) / entry_price
        
        if deduct_cost:
            stamp = 0.0005           # 印花税(卖出)
            commission = 0.00025 * 2 # 佣金(双边)
            # Dynamic slippage based on circulating market cap
            if circ_mv and circ_mv > 0:
                if circ_mv > 200 * 1e8:    slippage = 0.001   # >200亿 → 0.1%
                elif circ_mv > 50 * 1e8:   slippage = 0.002   # 50-200亿 → 0.2%
                else:                       slippage = 0.004   # <50亿 → 0.4%
            else:
                slippage = 0.002  # unknown → default 0.2%
            cost = stamp + commission + slippage
            gross_ret -= cost
        
        return gross_ret

    def _regime_at(self, date_str: str) -> str:
        """获取某日的市场状态。优先用 regime.pkl 标签，fallback 到 MA60。"""
        dt = pd.Timestamp(date_str)
        if dt in self.csi300.index:
            row = self.csi300.loc[dt]
            # Try regime.pkl label first (6 states)
            if 'regime' in row.index and pd.notna(row['regime']) and row['regime'] != 'unknown':
                return str(row['regime'])
            # Fallback: compute MA60
            if 'ma60' in row.index and pd.notna(row['ma60']) and row['ma60'] > 0:
                return 'bull' if row['close'] > row['ma60'] else 'bear'
        return 'unknown'

    # ── 精算分析 ──────────────────────────────────────

    def analyze(self, signal_df: pd.DataFrame, signal_name: str = "signal") -> ActuarialReport:
        """
        对信号DataFrame做全量精算分析。
        输入: signal_df 必须有 'forward_return' 列，可选 'signal_date'
        """
        report = ActuarialReport(signal_name=signal_name)
        rets = signal_df['forward_return'].dropna().values
        
        report.n_signals = len(rets)
        report.sufficient = len(rets) >= MIN_SAMPLES
        
        if not report.sufficient:
            return report
        
        # 1. 分布统计
        report.median = float(np.median(rets))
        report.mean = float(np.mean(rets))
        report.std = float(np.std(rets))
        report.win_rate = float(np.mean(rets > 0))
        
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        if len(losses) > 0 and len(wins) > 0:
            report.profit_loss_ratio = float(np.mean(wins) / abs(np.mean(losses)))
        
        report.left_tail_5 = float(np.percentile(rets, 5))
        report.left_tail_1 = float(np.percentile(rets, 1))
        report.right_tail_95 = float(np.percentile(rets, 95))
        report.skewness = float(pd.Series(rets).skew())
        
        # 2. 零假设 Bootstrap
        null_rets = self._bootstrap_null(len(rets), len(rets) > 0)
        report.null_median = float(np.median(null_rets))
        report.median_diff = float(report.median - report.null_median)
        report.win_rate_diff = float(report.win_rate - np.mean(null_rets > 0))
        
        # KS检验
        try:
            from scipy.stats import ks_2samp
            report.ks_stat, report.ks_pvalue = ks_2samp(rets, null_rets)
        except ImportError:
            # 无scipy时用简化判定
            report.ks_stat = abs(report.median_diff)
            report.ks_pvalue = 0.0 if report.median_diff > 0.02 else 0.5
        
        # 显著性判定
        report.is_significant = (
            report.median_diff > 0.02
            and report.win_rate_diff > 0.05
            and report.ks_pvalue < 0.01
            and report.left_tail_5 >= np.percentile(null_rets, 5)
        )
        
        # 3. 前向验证
        if 'signal_date' in signal_df.columns:
            self._walkforward(signal_df, report)
        
        # 4. Bootstrap鲁棒性
        self._bootstrap_robustness(rets, report)
        
        # 5. 逐年拆开
        if 'signal_date' in signal_df.columns:
            self._yearly_breakdown(signal_df, report)
        
        return report

    def _bootstrap_null(self, n_signals: int, use_csi300: bool = True) -> np.ndarray:
        """
        Bootstrap零假设：随机择时的收益分布。
        从CSI300日收益中随机采样FORWARD_DAYS天的滚动收益。
        """
        rng = np.random.RandomState(42)
        
        if use_csi300 and self.csi300 is not None:
            daily = self.csi300['pct_chg'].dropna().values / 100.0
            null_rets = []
            for _ in range(BOOTSTRAP_N):
                start = rng.randint(0, len(daily) - self.forward_days - 1)
                cum = np.prod(1 + daily[start:start + self.forward_days]) - 1
                null_rets.append(cum)
        else:
            # 无数据时用正态近似
            null_rets = rng.normal(0.01, 0.15, BOOTSTRAP_N)
        
        return np.array(null_rets)

    def _walkforward(self, signal_df: pd.DataFrame, report: ActuarialReport):
        """滚动前向验证"""
        signal_df = signal_df.copy()
        signal_df['signal_date'] = pd.to_datetime(signal_df['signal_date'])
        
        medians = []
        for train_s, train_e, val_s, val_e in WALKFORWARD:
            val = signal_df[
                (signal_df['signal_date'] >= val_s) & 
                (signal_df['signal_date'] <= val_e)
            ]
            if len(val) >= 10:
                medians.append(float(np.median(val['forward_return'])))
        
        report.wf_medians = medians
        if len(medians) >= 2:
            report.wf_range = max(medians) - min(medians)
            if len(medians) >= 3:
                baseline = report.median
                deviations = [abs(m - baseline) for m in medians]
                report.wf_stable = all(d <= 0.015 for d in deviations)

    def _bootstrap_robustness(self, rets: np.ndarray, report: ActuarialReport):
        """Bootstrap重抽样，计算VaR置信区间"""
        rng = np.random.RandomState(42)
        medians_boot = []
        var5_boot = []
        
        for _ in range(1000):
            sample = rng.choice(rets, size=len(rets), replace=True)
            medians_boot.append(np.median(sample))
            var5_boot.append(np.percentile(sample, 5))
        
        ci_lo = np.percentile(medians_boot, 2.5)
        ci_hi = np.percentile(medians_boot, 97.5)
        report.median_ci = (float(ci_lo), float(ci_hi))
        report.var_ci_width = float(np.std(var5_boot) * 2)

    def _yearly_breakdown(self, signal_df: pd.DataFrame, report: ActuarialReport):
        """逐年效应量，检测衰减"""
        df = signal_df.copy()
        df['year'] = pd.to_datetime(df['signal_date']).dt.year
        for year, grp in df.groupby('year'):
            if len(grp) >= 10:
                report.yearly_medians[int(year)] = float(np.median(grp['forward_return']))

    # ── 条件分布 ──────────────────────────────────────

    def analyze_conditional(self, signal_df: pd.DataFrame, condition_col: str,
                            signal_name: str = "signal") -> Dict[str, ActuarialReport]:
        """
        按条件标签分层分析。

        condition_col: signal_df中的列名，如 'pe_quantile' / 'north_flow_direction'
        返回: {tag_value: ActuarialReport}
        """
        results = {}
        for tag, subset in signal_df.groupby(condition_col):
            if len(subset) >= MIN_SAMPLES:
                name = f"{signal_name}[{condition_col}={tag}]"
                results[str(tag)] = self.analyze(subset, signal_name=name)
        return results

    def print_conditional_summary(self, signal_df: pd.DataFrame, condition_col: str,
                                   signal_name: str = "signal"):
        """打印条件分布对比"""
        results = self.analyze_conditional(signal_df, condition_col, signal_name)
        
        if not results:
            print(f"  No condition with >= {MIN_SAMPLES} samples")
            return
        
        print(f"\n  === Conditional: {condition_col} ===")
        print(f"  {'Tag':<20} {'N':>6} {'Median':>8} {'WinRate':>8} {'LeftTail5%':>10}")
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
        
        for tag, r in sorted(results.items()):
            print(f"  {tag:<20} {r.n_signals:>6} {r.median:>+7.1%} {r.win_rate:>7.1%} {r.left_tail_5:>+9.1%}")
        
        # 极值对比
        if len(results) >= 2:
            best = max(results.values(), key=lambda r: r.median)
            worst = min(results.values(), key=lambda r: r.median)
            print(f"  Spread: best={best.signal_name}({best.median:+.1%}) vs "
                  f"worst={worst.signal_name}({worst.median:+.1%}) "
                  f"-> delta={best.median-worst.median:+.2%}")

    # ── 批量对比 ──────────────────────────────────────

    def compare_signals(self, signal_dfs: Dict[str, pd.DataFrame]):
        """
        多个信号类型并排对比。

        signal_dfs: {"动量": df1, "恐慌": df2, "资金迁徙": df3}
        打印对比表并返回reports。
        """
        reports = {}
        for name, sdf in signal_dfs.items():
            reports[name] = self.analyze(sdf, signal_name=name)
        
        print(f"\n  {'='*80}")
        print(f"  MULTI-SIGNAL COMPARISON")
        print(f"  {'='*80}")
        print(f"  {'Signal':<20} {'N':>6} {'Median':>8} {'WinRate':>8} "
              f"{'LeftTail5%':>10} {'vsNull':>8} {'Stable':>8}")
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
        
        for name, r in sorted(reports.items(), key=lambda x: -x[1].median):
            stable = "Y" if r.wf_stable else "N"
            sig = "Y" if r.is_significant else "N"
            print(f"  {name:<20} {r.n_signals:>6} {r.median:>+7.1%} {r.win_rate:>7.1%} "
                  f"{r.left_tail_5:>+9.1%} {r.median_diff:>+7.1%} {stable:>8}")
        
        return reports

    # ── 滚动审计 ──────────────────────────────────────

    def rolling_audit(self, signal_df: pd.DataFrame, signal_name: str = "signal",
                      lookback_months: int = 12, alert_threshold_std: float = 1.0):
        """
        滚动审计：检测信号是否在衰减。

        lookback_months: 回溯窗口（月）
        alert_threshold_std: 近期中位低于历史均值多少个标准差触发警告

        返回: audit dict with status and details
        """
        df = signal_df.copy()
        df['signal_date'] = pd.to_datetime(df['signal_date'])
        df = df.sort_values('signal_date')
        
        if len(df) < MIN_SAMPLES:
            return {"status": "insufficient_data", "n": len(df)}
        
        # 全量历史分布
        hist_median = float(np.median(df['forward_return']))
        hist_std = float(np.std(df['forward_return']))
        
        # 最近lookback_months个月
        cutoff = df['signal_date'].max() - pd.DateOffset(months=lookback_months)
        recent = df[df['signal_date'] >= cutoff]
        
        if len(recent) < 20:
            return {"status": "insufficient_recent", "n_recent": len(recent)}
        
        recent_median = float(np.median(recent['forward_return']))
        z_score = (recent_median - hist_median) / hist_std if hist_std > 0 else 0
        
        # 判定
        if z_score < -alert_threshold_std:
            status = "DECAYING"
        elif z_score < -0.5:
            status = "WATCH"
        else:
            status = "HEALTHY"
        
        # 逐年拆开看趋势
        df['year'] = df['signal_date'].dt.year
        yearly = {}
        for y, g in df.groupby('year'):
            if len(g) >= 10:
                yearly[int(y)] = float(np.median(g['forward_return']))
        
        audit = {
            "status": status,
            "signal_name": signal_name,
            "hist_median": hist_median,
            "hist_std": hist_std,
            "recent_median": recent_median,
            "z_score": z_score,
            "n_total": len(df),
            "n_recent": len(recent),
            "yearly_medians": yearly,
        }
        
        print(f"\n  Rolling Audit: {signal_name}")
        print(f"  Status: {status}")
        print(f"  Hist median: {hist_median:+.2%} (n={len(df)})")
        print(f"  Recent {lookback_months}m: {recent_median:+.2%} (n={len(recent)})")
        print(f"  Z-score: {z_score:+.2f}")
        if yearly:
            trend = " -> ".join(f"{y}:{v:+.1%}" for y, v in sorted(yearly.items()))
            print(f"  Yearly: {trend}")
        
        return audit

    # ── 归因分析（补丁2）──────────────────────────────

    def analyze_attribution(self, signal_df: pd.DataFrame,
                            signal_name: str = "signal") -> dict:
        """
        Alpha/Beta decomposition.
        
        For each signal, regress signal return against CSI300 return over same period.
        Beta ≈ market exposure. Alpha ≈ signal's independent value.
        """
        df = signal_df.copy()
        if 'signal_date' not in df.columns:
            return {"error": "signal_date column required"}

        df['signal_date'] = pd.to_datetime(df['signal_date'])
        df = df.sort_values('signal_date')

        # Compute CSI300 forward returns for each signal date
        cs = self.csi300.copy()
        cs_returns = []

        for _, row in df.iterrows():
            dt = row['signal_date']
            if dt not in cs.index:
                cs_returns.append(np.nan)
                continue
            idx = cs.index.get_loc(dt)
            exit_pos = min(idx + self.forward_days, len(cs) - 1)
            if exit_pos <= idx:
                cs_returns.append(np.nan)
                continue
            cs_ret = (cs.iloc[exit_pos]['close'] - cs.iloc[idx]['close']) / cs.iloc[idx]['close']
            cs_returns.append(cs_ret)

        df['market_return'] = cs_returns
        df = df.dropna(subset=['forward_return', 'market_return'])

        if len(df) < 100:
            return {"error": f"Only {len(df)} valid pairs, need >=100"}

        # Linear regression: signal_return = alpha + beta * market_return
        X = df['market_return'].values
        y = df['forward_return'].values
        
        # Add intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])
        coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        alpha = coeffs[0]
        beta = coeffs[1]

        # R-squared
        y_pred = alpha + beta * X
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Monthly alpha (annualized from 40-day)
        monthly_alpha = alpha * (21 / self.forward_days)
        annual_alpha = alpha * (252 / self.forward_days)

        result = {
            "signal_name": signal_name,
            "n": len(df),
            "alpha": float(alpha),
            "beta": float(beta),
            "r_squared": float(r_squared),
            "monthly_alpha": float(monthly_alpha),
            "annual_alpha": float(annual_alpha),
            "market_corr": float(np.corrcoef(X, y)[0, 1]),
        }

        print(f"\n  Attribution: {signal_name}")
        print(f"  Alpha: {alpha:+.2%} (monthly: {monthly_alpha:+.1%}, annual: {annual_alpha:+.1%})")
        print(f"  Beta:  {beta:.2f} (market sensitivity)")
        print(f"  R-squared: {r_squared:.2%} (explained by market)")
        print(f"  Market correlation: {result['market_corr']:+.3f}")
        
        if beta < 0.3:
            print(f"  → Signal is largely INDEPENDENT of market (beta={beta:.2f})")
        elif beta > 0.7:
            print(f"  → Signal is mostly MARKET-DRIVEN (beta={beta:.2f})")
        else:
            print(f"  → Signal has MIXED exposure (beta={beta:.2f})")

        return result


# ── 便捷函数 ──────────────────────────────────────────

def create_engine() -> ActuarialEngine:
    """创建并加载数据的引擎"""
    return ActuarialEngine().load_data()
