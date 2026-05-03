"""Compute chain-composite max drawdown for V2r2-B (dual model)"""
import pandas as pd, numpy as np, os

path = r'D:\AIHedgeFund\ai-hedge-fund-main\quant_archive\2026-05\backtest_v2r2_dual_equity.csv'
df = pd.read_csv(path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['window', 'date']).reset_index(drop=True)

# Chain composite: start $1M, scale each window sequentially
chain = []
capital = 1_000_000
for wname in df['window'].unique():
    w = df[df['window'] == wname].copy()
    w_start = w['value'].iloc[0]
    w['chain_value'] = w['value'] / w_start * capital
    chain.append(w)
    capital = chain[-1]['chain_value'].iloc[-1]

chain_df = pd.concat(chain, ignore_index=True)

# Compute chain composite max drawdown
chain_df['cummax'] = chain_df['chain_value'].cummax()
chain_df['drawdown'] = chain_df['chain_value'] / chain_df['cummax'] - 1
max_dd = chain_df['drawdown'].min()
end_value = chain_df['chain_value'].iloc[-1]
peak_idx = chain_df['chain_value'].idxmax()
peak_date = chain_df.loc[peak_idx, 'date']
trough_idx = chain_df['drawdown'].idxmin()
trough_date = chain_df.loc[trough_idx, 'date']

print("=== V2r2-B CHAIN COMPOSITE ANALYSIS ===")
print(f"Initial capital:  $1,000,000")
print(f"Final capital:    ${end_value:,.0f}")
print(f"Cumulative return: {end_value/1_000_000 - 1:.2%}")
print(f"Chain Max DD:     {max_dd:.2%}")
print(f"Peak at:          {peak_date}")
print(f"Trough at:        {trough_date}")
print(f"Duration:         {(pd.Timestamp(trough_date) - pd.Timestamp(peak_date)).days} days")

# Drawdowns sorted
print(f"\n=== Worst drawdown days ===")
df_dd = chain_df[chain_df['drawdown'] < -0.05].sort_values('drawdown').head(10)
for _, row in df_dd.iterrows():
    print(f"  {row['drawdown']:.2%}  {row['date']}  window={row['window']}  regime={row['regime']}  pos={int(row['num_positions'])}")

# Per-window contributions
print(f"\n=== Window breakdown ===")
cap = 1_000_000
for wname in chain_df['window'].unique():
    w = chain_df[chain_df['window'] == wname]
    w_cr = w['chain_value'].iloc[-1] / w['chain_value'].iloc[0] - 1
    w_maxdd = (w['chain_value'] / w['chain_value'].cummax() - 1).min()
    w_peak = w['chain_value'].max()
    print(f"  {wname}:  +{w_cr:.2%}  |  maxDD {w_maxdd:.2%}  |  peak ${w_peak:,.0f}")
    cap = w['chain_value'].iloc[-1]

# Trade-level analysis
trades_path = r'D:\AIHedgeFund\ai-hedge-fund-main\quant_archive\2026-05\backtest_v2r2_dual_trades.csv'
if os.path.exists(trades_path):
    td = pd.read_csv(trades_path)
    td['exit_date'] = pd.to_datetime(td['exit_date'])
    
    # Consecutive loss analysis
    td_sorted = td.sort_values(['window', 'exit_date']).reset_index(drop=True)
    td_sorted['is_loss'] = td_sorted['net_return'] < 0
    
    # Find longest consecutive loss streaks
    max_streak = 0
    current_streak = 0
    streaks = []
    streak_start = None
    
    for i, row in td_sorted.iterrows():
        if row['is_loss']:
            if current_streak == 0:
                streak_start = row['exit_date']
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append((streak_start, row['exit_date'], current_streak))
                max_streak = max(max_streak, current_streak)
            current_streak = 0
    
    # Streak cumulative loss
    if len(streaks) > 0:
        print(f"\n=== Longest consecutive loss streaks ===")
        for ss, se, n in sorted(streaks, key=lambda x: -x[2])[:5]:
            mask = (td_sorted['exit_date'] >= ss) & (td_sorted['exit_date'] <= se) & td_sorted['is_loss']
            cum_loss = td_sorted.loc[mask, 'net_return'].sum()
            print(f"  {n} consecutive losses  {ss.date()} ~ {se.date()}  cum. {cum_loss:.2%}")

print("\nDone.")
