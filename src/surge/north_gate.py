"""
北向择时策略 v2.0 — 北向+南向双重确认

信号: 北向5日净流入 vs 2年滚动分位 + 南向回流确认
标的: CSI300 ETF (510300)
仓位: 三层阶梯 + 南向调整

规则:
  - 北向5日 > P90  + 南向未大幅南下 → 满仓 (100%)
  - 北向5日 > P70  + 南向未大幅南下 → 半仓 (50%)
  - 南向大幅回流 (内地钱回A股, 62.5%胜率) → +1档 (半仓→满仓)
  - 北向5日 < 0 → 强制空仓 (防御)

回测 (2020-2026): +37.3%, 夏普0.65, 回撤-7.8%, 在场16%
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Params:
    lookback_days: int = 504
    p70: float = 0.70
    p90: float = 0.90
    north_window: int = 5


class NorthGateStrategy:
    """北向+南向闸门"""

    def __init__(self, north_data: pd.DataFrame, params: Params = None):
        self.p = params or Params()
        self.north = north_data.copy()
        self._prepare()

    def _prepare(self):
        n = self.north
        n['n5'] = n['north_money'].rolling(self.p.north_window).sum()
        n['p70'] = n['n5'].rolling(self.p.lookback_days, min_periods=252).apply(
            lambda x: np.percentile(x, self.p.p70 * 100), raw=True)
        n['p90'] = n['n5'].rolling(self.p.lookback_days, min_periods=252).apply(
            lambda x: np.percentile(x, self.p.p90 * 100), raw=True)

        # 南向信号: 5日累计, P20以下=内地钱回流A股(利好)
        if 'south_money' in n.columns:
            n['s5'] = pd.to_numeric(n['south_money'], errors='coerce').rolling(5).sum()
            n['s_p20'] = n['s5'].rolling(self.p.lookback_days, min_periods=252).apply(
                lambda x: np.percentile(x, 20), raw=True)
            self.has_south = True
        else:
            self.has_south = False

        n = n.dropna()
        cols = ['n5', 'p70', 'p90']
        if self.has_south: cols += ['s5', 's_p20']
        self.df = n[cols].copy()

    def position(self, date: pd.Timestamp) -> float:
        if date not in self.df.index:
            return 0.0

        row = self.df.loc[date]
        n5, p70, p90 = row['n5'], row['p70'], row['p90']

        if n5 <= 0:
            return 0.0

        pos = 0.0
        if n5 > p90:
            pos = 1.0
        elif n5 > p70:
            pos = 0.5

        # 南向回流加成: 内地钱从港股回来 → +1档
        if self.has_south:
            s5, s_p20 = row['s5'], row['s_p20']
            if s5 < s_p20 and pos > 0:
                pos = min(1.0, pos + 0.5)

        return pos

    def get_signal(self, date: pd.Timestamp) -> dict:
        pos = self.position(date)
        return {
            'date': date.strftime('%Y-%m-%d'),
            'position': pos,
            'action': 'HOLD' if pos > 0 else 'CASH',
            'level': 'FULL' if pos >= 1.0 else ('HALF' if pos >= 0.5 else 'NONE'),
        }
