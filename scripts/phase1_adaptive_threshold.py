"""
Phase 1a: Adaptive Threshold Backtest
======================================
对比固定阈值 vs 自适应阈值 (Top N / Top P%)。

流程:
  1. 加载因子缓存 + 全量模型评分
  2. 按月模拟调仓: 每月第一天选股 → 持仓20个交易日
  3. 对比: 固定阈值 vs Top N 自适应 vs Top P% 自适应
  4. 输出超额收益、胜率、夏普比率

用法: python scripts/phase1_adaptive_threshold.py
"""
import sys, os, pickle, time, json, gc, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import lightgbm as lgb

print('=' * 65)
print('Phase 1a: Adaptive Threshold vs Fixed Threshold')
print('=' * 65)

# ── 1. Load factor cache ──────────────────────────────────────────────────
print('\n[1/5] Loading data...')
t0 = time.time()
FACTOR_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
all_dfs = []
for fn in sorted(os.listdir(FACTOR_DIR)):
    if fn.endswith('.parquet'):
        all_dfs.append(pd.read_parquet(os.path.join(FACTOR_DIR, fn)))
pdf = pd.concat(all_dfs, ignore_index=True)
print(f'  Rows: {len(pdf):,}  Cols: {len(pdf.columns)}  {time.time()-t0:.0f}s')

# ── 2. Create labels ──────────────────────────────────────────────────────
print('\n[2/5] Creating labels...')
HORIZON = 20
pdf = pdf.sort_values(['vt_symbol', 'datetime']).reset_index(drop=True)
pdf['forward_ret'] = np.nan
for name, group in pdf.groupby('vt_symbol', sort=False):
    idx = group.index
    closes = group['close'].values
    for i in range(len(group) - HORIZON):
        pdf.loc[idx[i], 'forward_ret'] = (closes[i+HORIZON] - closes[i]) / closes[i]
pdf['label'] = (pdf['forward_ret'] >= 0.10).astype(int)
pdf['date'] = pd.to_datetime(pdf['datetime'])
pdf['year'] = pdf['date'].dt.year
pdf['year_month'] = pdf['date'].dt.strftime('%Y-%m')

surge_rate = pdf['label'].mean()
print(f'  Surge rate: {surge_rate:.2%}')
print(f'  Date range: {pdf["date"].min().date()} ~ {pdf["date"].max().date()}')

# ── 3. Load model & compute full ML scores ────────────────────────────────
print('\n[3/5] Computing ML scores...')
MODEL_PATH = os.path.join(ROOT, 'data', 'models', 'surge_lgbm_full.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT, 'data', 'models', 'surge_lgbm.pkl')

with open(MODEL_PATH, 'rb') as f:
    lgb_model = pickle.load(f)

with open(os.path.join(ROOT, 'src', 'surge', 'params.json')) as f:
    params = json.load(f)
TOP_40 = params['selected_features']
valid_features = [f for f in TOP_40 if f in pdf.columns]

feature_data = pdf[valid_features].astype(np.float32)
feature_data = np.clip(feature_data, -1e10, 1e10)
fv = feature_data.values
fv[np.isinf(fv)] = np.nan
nan_mask = ~np.isnan(fv).any(axis=1)
print(f'  Dropped {(~nan_mask).sum():,} NaN rows ({nan_mask.mean():.0%} valid)')
X_valid = fv[nan_mask]
clean = pdf[nan_mask].copy()

all_scores = lgb_model.predict(X_valid)
clean['ml_score'] = all_scores
print(f'  Score range: {all_scores.min():.4f} ~ {all_scores.max():.4f}')
print(f'  {time.time()-t0:.0f}s')

# ── 4. Simulate Monthly Rebalance ─────────────────────────────────────────
print('\n[4/5] Simulating monthly rebalance...')

# Get unique month-end dates for rebalance
months = sorted(clean['year_month'].unique())
print(f'  Months: {len(months)} ({months[0]} ~ {months[-1]})')

# We'll use the LAST trading day of each month as rebalance signal date
rebalance_dates = clean.groupby('year_month')['date'].max().reset_index()
rebalance_dates.columns = ['year_month', 'rebal_date']
rebalance_dates = rebalance_dates.sort_values('rebal_date')

def simulate_strategy(clean_df, rebal_dates, pick_fn, label, params_str=''):
    """Generic monthly rebalance simulator

    pick_fn(month_data, date) -> list of stock codes to buy
    """
    total_dates = len(rebal_dates)
    all_trades = []

    for i in range(total_dates):
        ym = rebal_dates.iloc[i]['year_month']
        rebal_date = rebal_dates.iloc[i]['rebal_date']

        # Stocks scored on rebalance date
        month_data = clean_df[clean_df['date'] == rebal_date]

        if len(month_data) < 10:
            continue

        # Pick stocks
        picks = pick_fn(month_data)
        if not picks or len(picks) == 0:
            continue

        picked_codes = set(p['ts_code'] for p in picks)

        # Hold for HORIZON trading days
        for code in picked_codes:
            stock_rows = clean_df[(clean_df['vt_symbol'] == code) &
                                  (clean_df['date'] >= str(rebal_date))]
            if len(stock_rows) < 2:
                continue

            # Forward return at estimated HORIZON trading days
            forward_data = stock_rows.iloc[min(HORIZON, len(stock_rows)-1)]
            entry_close = stock_rows.iloc[0]['close']
            exit_close = forward_data['close']

            ret = (exit_close - entry_close) / entry_close if entry_close > 0 else 0
            all_trades.append({
                'rebal_date': str(rebal_date.date()),
                'code': code,
                'entry_close': float(entry_close),
                'exit_close': float(exit_close),
                'return': float(ret),
            })

        if (i + 1) % 24 == 0 or i == total_dates - 1:
            print(f'    {label}: {i+1}/{total_dates} months, {len(all_trades)} trades...')

    # Stats
    if len(all_trades) < 5:
        print(f'  {label}: NEED MORE TRADES ({len(all_trades)})')
        return {'label': label, 'n_trades': 0, 'error': 'too_few_trades'}

    returns = np.array([t['return'] for t in all_trades])
    win_rate = (returns > 0).mean()
    avg_return = returns.mean()
    excess = avg_return - clean_df['forward_ret'].mean()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    print(f'\n  [{label}] {len(all_trades)} trades, WR={win_rate:.1%}, '
          f'avg={avg_return*100:+.2f}%, excess={excess*100:+.2f}%, Sharpe={sharpe:.2f}')

    return {
        'label': label,
        'params': params_str,
        'n_trades': len(all_trades),
        'win_rate': f'{win_rate:.1%}',
        'avg_return': f'{avg_return*100:+.2f}%',
        'excess': f'{excess*100:+.2f}%',
        'sharpe': round(sharpe, 2),
        'max_return': f'{returns.max()*100:+.2f}%',
        'min_return': f'{returns.min()*100:+.2f}%',
    }


# ════════════════════════════════════════════════════
# Strategy 1: Fixed threshold
# ════════════════════════════════════════════════════
print('\n--- Strategy 1: Fixed Threshold ---')

fixed_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
fixed_results = []

for th in fixed_thresholds:
    def make_fixed_picker(t):
        def picker(md):
            sel = md[md['ml_score'] >= t].sort_values('ml_score', ascending=False)
            return [{'ts_code': row['vt_symbol'], 'score': row['ml_score']}
                    for _, row in sel.iterrows()]
        return picker

    r = simulate_strategy(clean, rebalance_dates, make_fixed_picker(th),
                          f'Fixed>=0.{int(th*100):02d}',
                          f'threshold={th}')
    if r and r.get('n_trades', 0) > 0:
        fixed_results.append(r)
        print(f'    >>> Fixed >= {th:.2f}: {r["excess"]} excess, {r["win_rate"]} WR, {r["sharpe"]} Sharpe')


# ════════════════════════════════════════════════════
# Strategy 2: Adaptive by Rank (Top N)
# ════════════════════════════════════════════════════
print('\n--- Strategy 2: Adaptive Top N ---')

top_n_values = [5, 10, 15, 20, 30, 50]
rank_results = []

for n in top_n_values:
    def make_rank_picker(k):
        def picker(md):
            sel = md.sort_values('ml_score', ascending=False).head(k)
            return [{'ts_code': row['vt_symbol'], 'score': row['ml_score']}
                    for _, row in sel.iterrows()]
        return picker

    r = simulate_strategy(clean, rebalance_dates, make_rank_picker(n),
                          f'Top{n}',
                          f'top_n={n}')
    if r and r.get('n_trades', 0) > 0:
        rank_results.append(r)
        print(f'    >>> Top {n}: {r["excess"]} excess, {r["win_rate"]} WR, {r["sharpe"]} Sharpe')


# ════════════════════════════════════════════════════
# Strategy 3: Adaptive by Percentile (Top P%)
# ════════════════════════════════════════════════════
print('\n--- Strategy 3: Adaptive Top P% ---')

top_pcts = [0.05, 0.10, 0.15, 0.20, 0.25]
pct_results = []

for pct in top_pcts:
    def make_pct_picker(p):
        def picker(md):
            th = md['ml_score'].quantile(1 - p)
            sel = md[md['ml_score'] >= th].sort_values('ml_score', ascending=False)
            return [{'ts_code': row['vt_symbol'], 'score': row['ml_score']}
                    for _, row in sel.iterrows()]
        return picker

    r = simulate_strategy(clean, rebalance_dates, make_pct_picker(pct),
                          f'Top{int(pct*100)}%',
                          f'top_pct={pct}')
    if r and r.get('n_trades', 0) > 0:
        pct_results.append(r)
        print(f'    >>> Top {int(pct*100)}%: {r["excess"]} excess, {r["win_rate"]} WR, {r["sharpe"]} Sharpe')


# ════════════════════════════════════════════════════
# Strategy 4: Baseline (buy all stocks)
# ════════════════════════════════════════════════════
print('\n--- Strategy 4: Baseline (Buy All) ---')

def baseline_picker(md):
    return [{'ts_code': row['vt_symbol'], 'score': 0}
            for _, row in md.iterrows()]

baseline_result = simulate_strategy(clean, rebalance_dates, baseline_picker,
                                    'BuyAll', 'baseline')


# ── 5. Results Summary ─────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('SUMMARY')
print('='*65)

print(f'\n--- Fixed Threshold ---')
print(f'{"Strategy":<18} | {"Trades":>7} | {"Win%":>6} | {"AvgRet":>10} | {"Excess":>10} | {"Sharpe":>7}')
print(f'{"-"*18}-+-{"-"*7}-+-{"-"*6}-+-{"-"*10}-+-{"-"*10}-+-{"-"*7}')
for r in fixed_results:
    print(f'{r["label"]:<18} | {r["n_trades"]:>7,} | {r["win_rate"]:>6} | '
          f'{r["avg_return"]:>10} | {r["excess"]:>10} | {r["sharpe"]:>7}')

print(f'\n--- Adaptive Top N ---')
print(f'{"Strategy":<18} | {"Trades":>7} | {"Win%":>6} | {"AvgRet":>10} | {"Excess":>10} | {"Sharpe":>7}')
print(f'{"-"*18}-+-{"-"*7}-+-{"-"*6}-+-{"-"*10}-+-{"-"*10}-+-{"-"*7}')
for r in rank_results:
    print(f'{r["label"]:<18} | {r["n_trades"]:>7,} | {r["win_rate"]:>6} | '
          f'{r["avg_return"]:>10} | {r["excess"]:>10} | {r["sharpe"]:>7}')

print(f'\n--- Adaptive Top P% ---')
print(f'{"Strategy":<18} | {"Trades":>7} | {"Win%":>6} | {"AvgRet":>10} | {"Excess":>10} | {"Sharpe":>7}')
print(f'{"-"*18}-+-{"-"*7}-+-{"-"*6}-+-{"-"*10}-+-{"-"*10}-+-{"-"*7}')
for r in pct_results:
    print(f'{r["label"]:<18} | {r["n_trades"]:>7,} | {r["win_rate"]:>6} | '
          f'{r["avg_return"]:>10} | {r["excess"]:>10} | {r["sharpe"]:>7}')

if baseline_result and baseline_result.get('n_trades', 0) > 0:
    print(f'\n--- Baseline ---')
    print(f'{"Strategy":<18} | {"Trades":>7} | {"Win%":>6} | {"AvgRet":>10} | {"Excess":>10} | {"Sharpe":>7}')
    print(f'{"-"*18}-+-{"-"*7}-+-{"-"*6}-+-{"-"*10}-+-{"-"*10}-+-{"-"*7}')
    r = baseline_result
    print(f'{r["label"]:<18} | {r["n_trades"]:>7,} | {r["win_rate"]:>6} | '
          f'{r["avg_return"]:>10} | {r["excess"]:>10} | {r["sharpe"]:>7}')

# Best of each category
all_strats = []
for cat, lst in [('Fixed', fixed_results), ('TopN', rank_results), ('TopPct', pct_results)]:
    for r in lst:
        excess_val = float(r['excess'].replace('%','').replace('+','').replace('-','-'))
        all_strats.append((excess_val, cat, r['label'], r['excess'], r['sharpe'], r['n_trades']))

all_strats.sort(key=lambda x: x[0], reverse=True)

print(f'\n--- Leaderboard ---')
for i, (excess, cat, label, excess_str, sharpe, n) in enumerate(all_strats[:10], 1):
    print(f'  #{i} [{cat:8s}] {label:<16s} | Excess: {excess_str:>9s} | Sharpe: {sharpe} | Trades: {n}')

# ── 6. Save results ───────────────────────────────────────────────────────
print(f'\n  Saving results...')
out = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'baseline': baseline_result,
    'fixed': fixed_results,
    'adaptive_top_n': rank_results,
    'adaptive_top_pct': pct_results,
    'leaderboard': [
        {'rank': i+1, 'category': cat, 'label': label,
         'excess': excess_str, 'sharpe': sharpe, 'trades': n}
        for i, (excess, cat, label, excess_str, sharpe, n) in enumerate(all_strats)
    ],
}
out_path = os.path.join(ROOT, 'data', 'models', 'phase1_adaptive_results.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'  Saved: {out_path}')
print(f'\n  Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)')
print('='*65)
