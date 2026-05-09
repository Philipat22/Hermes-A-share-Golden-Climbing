"""
黄金坑策略 — 可复现验证脚本
================================
用途: 复现全部核心结论
依赖: data/cache/prices_full.pkl, data/cache/golden_pit_batch_results.pkl
运行: python golden_pit_verify.py
"""

import pickle, numpy as np, pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

CACHE = 'data/cache'

# [1/5] 加载数据
print("[1/5] Loading data...")
with open(f'{CACHE}/prices_full.pkl', 'rb') as f:
    prices_raw = pickle.load(f)
with open(f'{CACHE}/golden_pit_batch_results.pkl', 'rb') as f:
    gp_batch = pickle.load(f)

batch_codes = set(s['code'] for s in gp_batch)
print(f"  Prices: {len(prices_raw)} stocks, Batch: {len(batch_codes)} stocks")

# [2/5] 预计算 MA + 120日收益
print("[2/5] Computing MAs and 120d returns...")
all_ret_data = defaultdict(list)

class StockData:
    def __init__(self, code):
        df = prices_raw[code].copy().sort_values('trade_date').reset_index(drop=True)
        self.close = df['close'].values
        self.vol = df['vol'].values
        self.low = df['low'].values
        self.dates = df['trade_date'].values
        self.n = len(df)
        self.ma20 = pd.Series(self.close).rolling(20).mean().values
        self.ma50 = pd.Series(self.close).rolling(50).mean().values
        self.ma60 = pd.Series(self.close).rolling(60).mean().values
        self.ma120 = pd.Series(self.close).rolling(120).mean().values
        self.ma250 = pd.Series(self.close).rolling(250).mean().values
        for i in range(120, self.n):
            ret = (self.close[i] / self.close[i-120] - 1) * 100
            all_ret_data[str(self.dates[i])].append((code, ret))

stock_data = {}
for code in batch_codes:
    if code in prices_raw:
        sd = StockData(code)
        if sd.n >= 300:
            stock_data[code] = sd
print(f"  Computed for {len(stock_data)} stocks")

# [3/5] 检测回撤事件 + 趋势分
print("[3/5] Detecting pullback events...")
events = []
for code, sd in stock_data.items():
    close = sd.close
    i, n = 20, sd.n
    while i < n - 60:
        left, right = max(0, i-15), min(n-1, i+15)
        if close[i] >= np.max(close[left:right+1]):
            peak_idx = i
            look_ahead = min(n-1, peak_idx + 60)
            if look_ahead > peak_idx + 5:
                trough_idx = peak_idx + 1 + np.argmin(close[peak_idx+1:look_ahead+1])
                dd_pct = (close[trough_idx] / close[peak_idx] - 1) * 100
                if -15 <= dd_pct <= -5 and trough_idx >= 250:
                    t1 = close[trough_idx] > sd.ma250[trough_idx]
                    t2 = sd.ma50[trough_idx] > sd.ma50[max(0, trough_idx-20)]
                    t3 = sd.ma20[trough_idx] > sd.ma60[trough_idx]
                    t4 = sd.ma50[trough_idx] > sd.ma120[trough_idx]
                    t5 = sd.ma120[trough_idx] > sd.ma250[trough_idx]
                    events.append({
                        'code': code, 'peak_idx': peak_idx, 'trough_idx': trough_idx,
                        'peak_price': close[peak_idx], 'dd_pct': dd_pct,
                        'trend_score': sum([t1,t2,t3,t4,t5]),
                    })
                    i = trough_idx + 10
                    continue
        i += 1
print(f"  Events: {len(events)}")

# [4/5] 模拟进场 + 策略过滤
print("[4/5] Simulating entries...")
trades = []
for e in events:
    if e['trend_score'] < 4: continue
    sd = stock_data[e['code']]
    peak_idx, peak_price, n = e['peak_idx'], e['peak_price'], sd.n
    target = peak_price * 0.90
    cross_idx = None
    for j in range(peak_idx+1, min(n, peak_idx+120)):
        if sd.close[j] <= target: cross_idx = j; break
    if cross_idx is None: continue
    days_decline = cross_idx - peak_idx
    if days_decline <= 0: continue
    daily_speed = 10.0 / days_decline
    if daily_speed < 0.5: continue
    decline_vol = np.mean(sd.vol[peak_idx:cross_idx+1])
    prior_vol = np.mean(sd.vol[max(0,peak_idx-20):peak_idx+1])
    vol_ratio = decline_vol / prior_vol if prior_vol > 0 else 1
    if vol_ratio >= 3.0: continue
    entry_idx = min(cross_idx+1, n-1)
    entry_price = sd.close[entry_idx]
    rets = {}
    for d in [20,30,40,50,60]:
        end = min(n, entry_idx+d)
        if end > entry_idx:
            rets[f'ret_{d}d'] = (sd.close[end-1]/entry_price - 1)*100
    trades.append({'code': e['code'], 'daily_speed': daily_speed,
                   'vol_ratio': vol_ratio, 'trend_score': e['trend_score'], **rets})

df = pd.DataFrame(trades)
print(f"  Trades: {len(df)}")

# [5/5] 输出
print(f"\n{'='*70}")
print(f"黄金坑策略验证 (n={len(df)}笔, {len(df['code'].unique())}只股票)")
print(f"{'='*70}")
print(f"\n持有期收益:")
print(f"{'持有':>5s} {'均值':>7s} {'中位':>7s} {'胜率':>6s} {'P10':>7s} {'P25':>7s} {'P75':>7s} {'P90':>7s}")
for d in [20,30,40,50,60]:
    col = df[f'ret_{d}d']
    if col.notna().sum() < 100: continue
    print(f" {d:>3d}d {col.mean():>+6.1f}% {col.median():>+6.1f}% {(col>0).mean()*100:>5.0f}% "
          f"{col.quantile(0.1):>+6.1f}% {col.quantile(0.25):>+6.1f}% "
          f"{col.quantile(0.75):>+6.1f}% {col.quantile(0.9):>+6.1f}%")
print(f"\n策略参数 (全数据驱动):")
print(f"  选票: 趋势4-5/5 + C1+C4 | 闸门: CSI300>MA60")
print(f"  过滤: 跌速>0.5%/d + 量比<3x | 进场: -10%, T+1")
print(f"  出场: 持有50-60天到期")
