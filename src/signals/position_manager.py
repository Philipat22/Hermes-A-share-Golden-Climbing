"""
仓位管理 + 出场监控 + 反馈闭环
"""
from __future__ import annotations
import json, os
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════
# 仓位管理器
# ══════════════════════════════════════════════════════

class PositionSizer:
    """根据闸门状态和确信度计算目标仓位"""

    def __init__(self, capital: float = 100_000):
        self.capital = capital

    def size(self, gate_level: str, high_confidence: bool, current_positions: int) -> tuple[int, float]:
        """返回 (最大可买只数, 单只金额)"""
        if gate_level == 'CLOSED':
            return 0, 0

        if gate_level == 'FULL':
            if high_confidence:
                return 4 - current_positions, self.capital * 0.25
            else:
                return min(2, 4 - current_positions), self.capital * 0.15

        if gate_level == 'HALF':
            if high_confidence:
                return min(2, 4 - current_positions), self.capital * 0.15
            else:
                return 0, 0

        return 0, 0


# ══════════════════════════════════════════════════════
# 出场信号引擎
# ══════════════════════════════════════════════════════

class ExitEngine:
    """独立出场信号 — 每天收盘后检查持仓"""

    def __init__(self):
        self.rules = {
            'initial_stop': -0.08,       # 初始止损
            'trail_breakeven': 0.12,     # 浮盈12% → 保本
            'trail_lock8': 0.18,         # 浮盈18% → 锁8%
            'trail_lock15': 0.25,        # 浮盈25% → 锁15%
            'time_decay': 20,            # 持仓>20天浮盈<5%减半
            'gate_close': True,          # 北向关→全出
            'ma_break': True,            # MA20跌破2日→全出
            'crash': -0.07,              # 单日跌>7%+放量→全出
            'rsi_top': 80,               # RSI>80+量>1.5→减半
        }

    def check(self, holding: dict, df, gate_open: bool) -> list[dict]:
        """移动止损出场 — 不设目标价,用移动线保护利润"""
        signals = []
        entry_px = holding['entry_price']
        closes = df['close'].values; highs = df['high'].values
        vols = df['vol'].values if 'vol' in df.columns else np.ones(len(closes))
        c = closes[-1]
        ret = c / entry_px - 1

        # ── 移动止损线 ──
        # 从holdings.json读上一次的止损线,没有就用初始止损
        trail_stop = holding.get('trailing_stop', entry_px * (1 + self.rules['initial_stop']))

        # 根据当前浮盈, 计算应该上移到哪
        if ret >= self.rules['trail_lock15']:
            new_stop = entry_px * 1.15
        elif ret >= self.rules['trail_lock8']:
            new_stop = entry_px * 1.08
        elif ret >= self.rules['trail_breakeven']:
            new_stop = entry_px * 1.00
        else:
            new_stop = trail_stop  # 保持不动

        # 止损线只上移,不下移
        effective_stop = max(trail_stop, new_stop)

        # 更新holding供外部保存
        holding['trailing_stop'] = effective_stop
        holding['stop_pct'] = (effective_stop / entry_px - 1) * 100

        # 触发移动止损?
        if c <= effective_stop:
            signals.append({'action': 'SELL_ALL',
                           'reason': f'trail_stop(锁{holding["stop_pct"]:.0f}%)'})

        # ── RSI顶部减半仓 ──
        rsi = self._rsi(closes)
        vol_ratio = vols[-1] / np.mean(vols[-20:]) if len(vols)>=20 else 1
        if rsi > self.rules['rsi_top'] and vol_ratio > 1.5:
            signals.append({'action': 'SELL_HALF',
                           'reason': f'rsi_top(RSI{rsi:.0f}+量{vol_ratio:.1f})'})

        # ── MA20跌破 ──
        if len(closes) >= 22 and closes[-1] < np_mean(closes[-20:]) and closes[-2] < np_mean(closes[-21:-1]):
            signals.append({'action': 'SELL_ALL', 'reason': 'ma20_break_2d'})

        # ── 崩盘 ──
        if len(closes) >= 2 and closes[-1]/closes[-2]-1 <= self.rules['crash']:
            if vols[-1] > np.mean(vols[-20:]) * 1.5:
                signals.append({'action': 'SELL_ALL', 'reason': 'crash'})

        # ── 时间衰减 ──
        days = (pd.Timestamp.now() - pd.Timestamp(holding.get('entry_date','2020-01-01'))).days
        if days > self.rules['time_decay'] and ret < 0.05:
            signals.append({'action': 'SELL_HALF', 'reason': f'time_decay({days}d,{ret*100:.0f}%)'})

        # ── 北向闸门关闭 ──
        if self.rules['gate_close'] and not gate_open:
            signals.append({'action': 'SELL_ALL', 'reason': 'gate_closed'})

        return signals

    def _rsi(self, closes, period=14):
        if len(closes) < period + 1: return 50
        d = np.diff(closes[-period-1:])
        g = d.copy(); g[g<0]=0
        l = -d.copy(); l[l<0]=0
        return 100 - 100/(1+np.mean(g)/np.mean(l)) if np.mean(l)>0 else 100


import numpy as np
def np_mean(x): return np.mean(x)


# ══════════════════════════════════════════════════════
# 反馈闭环 — 大师准确率追踪
# ══════════════════════════════════════════════════════

class FeedbackLoop:
    """追踪大师推荐后续表现，调整权重"""

    def __init__(self, path: str = "data/cache/master_feedback.json"):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {'picks': [], 'master_stats': {}}

    def record_pick(self, ts_code: str, date: str, price: float, voters: list[str]):
        """记录一次推荐"""
        self.data['picks'].append({
            'ts_code': ts_code, 'date': date, 'price': price,
            'voters': voters, 'outcome': None, 'final_return': None,
        })

    def update_outcomes(self, prices_dict: dict):
        """更新历史推荐的结果"""
        now = datetime.now()
        for pick in self.data['picks']:
            if pick['outcome'] is not None: continue
            pick_date = datetime.strptime(pick['date'], '%Y-%m-%d')
            if (now - pick_date).days < 10: continue  # 至少10天后再评判

            code = pick['ts_code']
            if code not in prices_dict: continue
            df = prices_dict[code]
            closes = df['close'].values
            ret_10d = closes[-1] / pick['price'] - 1 if len(closes) > 0 else 0

            pick['outcome'] = 'WIN' if ret_10d > 0.05 else ('LOSS' if ret_10d < -0.05 else 'FLAT')
            pick['final_return'] = round(ret_10d * 100, 1)

            # 更新大师统计
            for master in pick['voters']:
                if master not in self.data['master_stats']:
                    self.data['master_stats'][master] = {'wins': 0, 'losses': 0, 'flats': 0, 'total': 0}
                self.data['master_stats'][master]['total'] += 1
                if pick['outcome'] == 'WIN': self.data['master_stats'][master]['wins'] += 1
                elif pick['outcome'] == 'LOSS': self.data['master_stats'][master]['losses'] += 1
                else: self.data['master_stats'][master]['flats'] += 1

        self._save()

    def get_weights(self) -> dict[str, float]:
        """返回大师权重 (基于历史胜率, 新大师默认1.0)"""
        weights = {}
        for name, stats in self.data['master_stats'].items():
            if stats['total'] >= 5:
                wr = stats['wins'] / stats['total']
                weights[name] = max(0.5, min(2.0, wr * 2.5))  # 0.5~2.0
            else:
                weights[name] = 1.0
        return weights

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
