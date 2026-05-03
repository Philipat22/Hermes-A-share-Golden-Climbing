"""Compute per-regime Kelly position sizing from V2r2-B trade data."""
import pandas as pd
import numpy as np
import os

base = r'D:\AIHedgeFund\ai-hedge-fund-main'
path = os.path.join(base, 'quant_archive/2026-05/backtest_v2r2_dual_trades.csv')
df = pd.read_csv(path)

# Current V2r2 position caps per regime
current_caps = {
    'bull': '80%',
    'sideways': '60% (↓13%实际通过阈值)',
    'bear': '30%',
    'severe_bear': '15%',
    'rebound': '0%'
}

regime_order = ['bull', 'sideways', 'bear', 'severe_bear', 'rebound']

print(f"{'Regime':<14} {'交易':>5} {'胜率':>6} {'均赢':>7} {'均亏':>7} {'盈亏比':>8} {'笔均收益':>9} {'Kelly全仓':>10} {'Kelly1/4':>9} {'当前仓位':>10}")
print('-' * 95)

rows = []
for regime in regime_order:
    sub = df[df['regime'] == regime]
    n = len(sub)
    if n == 0:
        print(f"{regime:<14} {0:>5} {'N/A':>6} {'N/A':>7} {'N/A':>7} {'N/A':>8} {'N/A':>9} {'N/A':>10} {'N/A':>9} {current_caps.get(regime, '?'):>10}")
        continue
    
    wins = sub[sub['net_return'] > 0]
    losses = sub[sub['net_return'] <= 0]
    n_win = len(wins)
    n_loss = len(losses)
    
    win_rate = n_win / n
    avg_win = wins['net_return'].mean() if n_win > 0 else 0
    avg_loss = abs(losses['net_return'].mean()) if n_loss > 0 else 0.001
    b = avg_win / avg_loss if avg_loss > 0 else 999
    
    # Kelly formula: f* = (p*b - q) / b
    kelly = (win_rate * b - (1 - win_rate)) / b if b > 0 else 0
    kelly = max(0, kelly)
    kelly_q = kelly * 0.25
    
    ev = sub['net_return'].mean() * 100
    
    print(f"{regime:<14} {n:>5} {win_rate*100:>5.1f}% {avg_win*100:>6.2f}% {avg_loss*100:>6.2f}% {b:>7.2f}x {ev:>8.2f}% {kelly*100:>9.1f}% {kelly_q*100:>8.1f}% {current_caps.get(regime, '?'):>10}")
    
    rows.append({
        'regime': regime,
        'trades': n,
        'win_rate': f'{win_rate*100:.1f}%',
        'avg_win': f'{avg_win*100:.2f}%',
        'avg_loss': f'{avg_loss*100:.2f}%',
        'b': f'{b:.2f}x',
        'ev': f'{ev:.2f}%',
        'kelly_full': f'{kelly*100:.1f}%',
        'kelly_quarter': f'{kelly_q*100:.1f}%',
        'current': current_caps.get(regime, '?')
    })

# Total
all_win_rate = len(df[df['net_return'] > 0]) / len(df)
all_avg_win = df[df['net_return'] > 0]['net_return'].mean()
all_avg_loss = abs(df[df['net_return'] <= 0]['net_return'].mean())
all_b = all_avg_win / all_avg_loss
all_kelly = (all_win_rate * all_b - (1 - all_win_rate)) / all_b
all_kelly = max(0, all_kelly)

print()
print(f"总计  交易: {len(df)}, 胜率: {all_win_rate*100:.1f}%, 盈亏比: {all_b:.2f}x")
print(f"总计  Kelly全仓: {all_kelly*100:.1f}%, Kelly1/4: {all_kelly*25:.1f}%")
