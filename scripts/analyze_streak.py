"""Analyze the 18-consecutive loss streak (2023-11-24 to 2024-02-20)"""
import pandas as pd, numpy as np, os

path = r'D:\AIHedgeFund\ai-hedge-fund-main\quant_archive\2026-05\backtest_v2r2_dual_trades.csv'
td = pd.read_csv(path)
td['entry_date'] = pd.to_datetime(td['entry_date'])
td['exit_date'] = pd.to_datetime(td['exit_date'])
td = td.sort_values(['window', 'exit_date']).reset_index(drop=True)

# Find the 18 consecutive loss streak
# Use exit_date order to find the streak
td['is_loss'] = td['net_return'] < 0
streak_id = 0
td['streak_id'] = 0
current_streak = 0
in_loss = False
streak_intervals = []

for i in range(len(td)):
    if td.loc[i, 'is_loss']:
        if not in_loss:
            current_streak += 1
            in_loss = True
            streak_start = i
        else:
            current_streak += 1
    else:
        if in_loss and current_streak >= 10:
            streak_intervals.append((streak_start, i-1, current_streak))
        current_streak = 0
        in_loss = False

if in_loss and current_streak >= 10:
    streak_intervals.append((streak_start, len(td)-1, current_streak))

# Print the longest streak
longest = max(streak_intervals, key=lambda x: x[2])
start_idx, end_idx, n = longest
print(f"=== Longest Loss Streak: {n} consecutive losses ===")
print(f"Range: {td.loc[start_idx, 'exit_date'].date()} to {td.loc[end_idx, 'exit_date'].date()}")
print()

streak = td.loc[start_idx:end_idx].copy()
streak['score'] = streak.get('score', np.nan)  # score column might not exist

# Summary stats
print(f"Window: {streak['window'].unique()}")
print(f"Total net loss: {streak['net_return'].sum():.2%}")
print(f"Avg net loss per trade: {streak['net_return'].mean():.2%}")
print(f"Exit reasons: {streak['exit_reason'].value_counts().to_dict()}")
print(f"Regimes: {streak['regime'].value_counts().to_dict()}")
print()

# Regime transitions
print("=== Regime during streak ===")
regime_seq = streak['regime'].tolist()
transitions = sum(1 for i in range(1, len(regime_seq)) if regime_seq[i] != regime_seq[i-1])
print(f"  {transitions} regime transitions in {n} trades")
print(f"  Regime sequence: {' → '.join(regime_seq)}")
print()

# Sector analysis - extract sector from symbol prefix
print("=== Symbol analysis ===")
sym_counts = streak['symbol'].value_counts()
print(f"  Unique symbols: {len(sym_counts)}")
print(f"  Symbols traded 2+ times:")
for sym, cnt in sym_counts[sym_counts >= 2].items():
    trades = streak[streak['symbol'] == sym]
    print(f"    {sym}: {cnt}x trades, avg ret {trades['net_return'].mean():.2%}")
print()

# Average hold time
print(f"=== Holding period ===")
hold_days = (streak['exit_date'] - streak['entry_date']).dt.days
print(f"  Avg hold: {hold_days.mean():.1f} days")
print(f"  Median hold: {hold_days.median():.1f} days")
print(f"  Hold distribution: ")
for d in sorted(hold_days.unique()):
    cnt = (hold_days == d).sum()
    print(f"    {int(d)}d: {cnt} trades")
print()

# Calendar spread
print(f"=== Calendar ===")
entry_dates = streak['entry_date'].dt.date.unique()
exit_dates = streak['exit_date'].dt.date.unique()
print(f"  Entry dates: {len(entry_dates)} unique days")
print(f"  Exit dates: {len(exit_dates)} unique days")
print(f"  Calendar span: {streak['entry_date'].min().date()} to {streak['exit_date'].max().date()} ({(streak['exit_date'].max() - streak['entry_date'].min()).days} days)")
print()

# Full table
print(f"=== Full trade log ===")
print(f"{'#':>3} {'Symbol':>10} {'Entry':>10} {'Exit':>10} {'Hold':>4} {'Regime':>12} {'Reason':>14} {'Gross':>8} {'Net':>8}")
print("-" * 85)
for i, (_, r) in enumerate(streak.iterrows()):
    hold = (r['exit_date'] - r['entry_date']).days
    print(f"{i+1:>3} {str(r['symbol']):>10} {str(r['entry_date'].date()):>10} {str(r['exit_date'].date()):>10} {hold:>4} {str(r['regime']):>12} {str(r['exit_reason']):>14} {r['gross_return']:>7.2%} {r['net_return']:>7.2%}")

# Also look at the 13-loss streak in 2024-12
print("\n\n" + "=" * 85)
print("=== Also: 13-loss streak (2024-12-03 to 2025-01-10) ===")
streak13 = td[(td['exit_date'] >= '2024-12-03') & (td['exit_date'] <= '2025-01-10') & td['is_loss']]
print(f"  Trades: {len(streak13)}")
print(f"  Total net loss: {streak13['net_return'].sum():.2%}")
print(f"  Exit reasons: {streak13['exit_reason'].value_counts().to_dict()}")
print(f"  Regimes: {streak13['regime'].value_counts().to_dict()}")
print(f"  Unique symbols: {streak13['symbol'].nunique()}")
print(f"\n  Trade log:")
print(f"  {'#':>3} {'Symbol':>10} {'Entry':>10} {'Exit':>10} {'Hold':>4} {'Regime':>12} {'Reason':>14} {'Gross':>8} {'Net':>8}")
print("  " + "-" * 83)
for i, (_, r) in enumerate(streak13.iterrows()):
    hold = (r['exit_date'] - r['entry_date']).days
    print(f"  {i+1:>3} {str(r['symbol']):>10} {str(r['entry_date'].date()):>10} {str(r['exit_date'].date()):>10} {hold:>4} {str(r['regime']):>12} {str(r['exit_reason']):>14} {r['gross_return']:>7.2%} {r['net_return']:>7.2%}")

print("\nDone.")
