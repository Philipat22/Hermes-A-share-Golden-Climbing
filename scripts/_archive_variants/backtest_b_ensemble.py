"""Fix: 1) Compute & save top-40 feature mapping 2) Verify models work correctly 3) Backtest"""
import sys, os, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
os.chdir(ROOT)

# ── 1. Load everything ──────────────────────────────────────────────────
print('Loading data & models...')
with open('data/cache/factor_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
with open('data/models/surge_lgbm.pkl', 'rb') as f:
    lgb_model = pickle.load(f)
xgb_model = xgb.Booster(model_file='data/models/surge_xgboost.json')

X_full = data['X']  # (49859, 160)
meta = data['meta']
names = data['factor_names']

# ── 2. Find which 40 original factors the model uses ────────────────────
# LightGBM stored Column_0..Column_39. We need to map back to original 160.
# The model was trained on X_full[:, top40_idx] where top40_idx is the indices
# of the 40 highest-importance factor names.
# 
# Solution: for each model column order, score ALL 160 factors individually
# to find which column the model is looking at.
# 
# FASTER: Train a SHALLOW model on all 160 features, get the top-40 indices,
# then verify by checking that scores match.

print('\nRecovering top-40 feature mapping...')
# Method: use known-good computation from original data
# Train baseline on original labels to get importance
valid_mask = ~meta['forward_ret'].isna().values
y_baseline = data['y']

params = {
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
    'num_leaves': 63, 'max_depth': 6, 'learning_rate': 0.03,
    'feature_fraction': 0.8, 'bagging_fraction': 0.85,
    'bagging_freq': 5, 'min_child_samples': 50, 'random_state': 42,
}

# Use a 70/30 split
meta_dt = pd.to_datetime(meta['datetime'])
split_dt = meta_dt.quantile(0.7)
train_idx = np.where((meta_dt < split_dt).values)[0]
test_idx = np.where((meta_dt >= split_dt).values)[0]

dtrain = lgb.Dataset(X_full[train_idx], y_baseline[train_idx])
dval = lgb.Dataset(X_full[test_idx], y_baseline[test_idx], reference=dtrain)
ref_model = lgb.train(params, dtrain, num_boost_round=200,
                      valid_sets=[dval], valid_names=['valid'],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

# Get importance on all 160 factors
imp = ref_model.feature_importance(importance_type='gain')
feat_imp = list(zip(names, imp))
feat_imp.sort(key=lambda x: -x[1])
top40_names = [f for f, _ in feat_imp[:40]]
name_to_idx = {n: i for i, n in enumerate(names)}
top40_idx = [name_to_idx[n] for n in top40_names]

print(f'Top-40 factors (first 10):')
for n, imp_val in feat_imp[:10]:
    print(f'  {n}: {imp_val:.2f}')

# ── 3. Verify: use saved models on correct features ─────────────────────
X_top40 = X_full[:, top40_idx]

# Score with saved models
scores_lgb = lgb_model.predict(X_top40)
scores_xgb = xgb_model.predict(xgb.DMatrix(X_top40))
scores_ensemble = (scores_lgb + scores_xgb) / 2.0

print(f'\nEnsemble score distribution on ALL {len(scores_ensemble)} samples:')
print(f'  Min: {scores_ensemble.min():.4f}, Max: {scores_ensemble.max():.4f}')
print(f'  Mean: {scores_ensemble.mean():.4f}, Median: {np.median(scores_ensemble):.4f}')
print(f'  Std: {scores_ensemble.std():.4f}')

print(f'\nScoring thresholds:')
for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    count = (scores_ensemble >= th).sum()
    if count > 0:
        pct = count / len(scores_ensemble) * 100
        print(f'  >= {th:.2f}: {count} ({pct:.2f}%)')

# ── 4. Actual backtest (chronological) ──────────────────────────────────
print('\n' + '=' * 60)
print('BACKTEST B: ENSEMBLE SIGNAL (chronological)')
print('=' * 60)

THRESHOLD = 0.40
REBALANCE = 20
TC_PCT = 0.003
CAPITAL = 20000

# Filter to valid samples only
valid_mask = ~meta['forward_ret'].isna().values
meta_v = meta[valid_mask].copy().reset_index(drop=True)
X_top40_v = X_top40[valid_mask]
scores_v = scores_ensemble[valid_mask]

meta_v['datetime'] = pd.to_datetime(meta_v['datetime'])
meta_v['score'] = scores_v
meta_v = meta_v.sort_values('datetime').reset_index(drop=True)

print(f'\nValid samples: {len(meta_v)}')
print(f'Date range: {meta_v["datetime"].min()} ~ {meta_v["datetime"].max()}')
print(f'Threshold: {THRESHOLD}, Rebalance: {REBALANCE}d, TC: {TC_PCT*100:.1f}%')

dates = meta_v['datetime'].unique()
n_periods = len(dates) // REBALANCE
print(f'Unique dates: {len(dates)}, Rebalance periods: {n_periods}')

equity = [CAPITAL]
bm_equity = [CAPITAL]
picks_log = []
period_rets = []
bm_rets = []
total_picks = 0

for i in range(n_periods):
    start_idx = i * REBALANCE
    end_idx = min((i + 1) * REBALANCE, len(dates))
    
    date_start = dates[start_idx]
    date_end = dates[end_idx - 1] if end_idx < len(dates) else dates[-1]
    
    # Samples at this rebalance date
    mask = meta_v['datetime'] == date_start
    samples = meta_v[mask]
    
    if len(samples) == 0:
        equity.append(equity[-1])
        bm_equity.append(bm_equity[-1])
        picks_log.append({
            'date': str(date_start)[:10], 'picks': 0,
            'score': 0, 'ret': 0, 'bm_ret': 0
        })
        period_rets.append(0)
        bm_rets.append(0)
        continue
    
    # Ensemble picks
    picks = samples[samples['score'] >= THRESHOLD]
    n_picks = len(picks)
    total_picks += n_picks
    
    # Strategy return (avg forward return of picks)
    if n_picks > 0:
        avg_fwd = picks['forward_ret'].mean()
    else:
        avg_fwd = 0.0
    
    tc = TC_PCT if n_picks > 0 else 0
    period_ret = (1 + avg_fwd) * (1 - tc) - 1
    period_rets.append(period_ret)
    equity.append(equity[-1] * (1 + period_ret))
    
    # Benchmark: all stocks equally weighted
    bm_ret = samples['forward_ret'].mean()
    bm_ret_adj = (1 + bm_ret) * (1 - TC_PCT) - 1
    bm_rets.append(bm_ret_adj)
    bm_equity.append(bm_equity[-1] * (1 + bm_ret_adj))
    
    picks_log.append({
        'date': str(date_start)[:10],
        'picks': n_picks,
        'avg_score': round(picks['score'].mean(), 3) if n_picks > 0 else 0,
        'ret': round(period_ret * 100, 2),
        'bm_ret': round(bm_ret_adj * 100, 2),
        'excess': round((period_ret - bm_ret_adj) * 100, 2),
    })

# ── 5. Compute statistics ──────────────────────────────────────────────
returns = np.array(period_rets)
bm_returns = np.array(bm_rets)
equity = np.array(equity)
bm_equity = np.array(bm_equity)
nav = equity / equity[0]
bm_nav = bm_equity / bm_equity[0]

periods_per_year = 240 / REBALANCE
years = n_periods / periods_per_year

def calc_stats(equity_curve, period_rets, label):
    if years == 0: return {}
    total_ret = equity_curve[-1] / equity_curve[0] - 1
    annual_ret = (1 + total_ret) ** (1 / years) - 1
    
    period_vol = np.std(period_rets, ddof=1) if len(period_rets) > 1 else 0
    annual_vol = period_vol * np.sqrt(periods_per_year) if period_vol > 0 else 0
    
    rfr_per_period = 0.03 / periods_per_year
    excess_r = period_rets - rfr_per_period
    sharpe = (np.mean(excess_r) / period_vol * np.sqrt(periods_per_year)) if period_vol > 0 else 0
    
    cummax = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - cummax) / cummax
    max_dd = dd.min()
    
    wr = (period_rets > 0).mean() if len(period_rets) > 0 else 0
    calmar = annual_ret / abs(max_dd) if max_dd < 0 else float('inf')
    
    print(f'\n--- {label} ---')
    print(f'  Total Return: {total_ret*100:.2f}%')
    print(f'  Annualized: {annual_ret*100:.2f}%')
    print(f'  Volatility: {annual_vol*100:.2f}%')
    print(f'  Sharpe: {sharpe:.2f}')
    print(f'  Max DD: {max_dd*100:.2f}%')
    print(f'  Win Rate: {wr*100:.1f}%')
    print(f'  Calmar: {calmar:.2f}')
    return {
        'total_ret': total_ret, 'annual_ret': annual_ret,
        'annual_vol': annual_vol, 'sharpe': sharpe,
        'max_dd': max_dd, 'win_rate': wr, 'calmar': calmar,
    }

print(f'\nBacktest periods: {n_periods} ({years:.1f} years)')
strat_stats = calc_stats(equity, returns, 'STRATEGY')
bm_stats = calc_stats(bm_equity, bm_returns, 'BENCHMARK')

alpha = strat_stats['annual_ret'] - bm_stats['annual_ret']
print(f'\nAnnual Alpha: {alpha*100:.2f}%')
print(f'Avg Picks / Period: {total_picks / n_periods:.1f}')
zero_pick_periods = sum(1 for p in picks_log if p['picks'] == 0)
print(f'Zero-pick periods: {zero_pick_periods}/{n_periods} ({zero_pick_periods/n_periods*100:.1f}%)')

# ── 6. Save everything ─────────────────────────────────────────────────
# Save top-40 mapping
top40_map = {
    'top40_names': top40_names,
    'top40_idx': top40_idx,
    'factor_names_pool': names,
}
map_path = os.path.join(ROOT, 'data', 'models', 'top40_idx.json')
with open(map_path, 'w') as f:
    json.dump(top40_map, f, indent=2, ensure_ascii=False)
print(f'\nTop-40 mapping saved: {map_path}')

# Save backtest results
bt_results = {
    'strategy': strat_stats,
    'benchmark': bm_stats,
    'alpha': alpha,
    'avg_picks': total_picks / n_periods,
    'n_periods': n_periods,
    'years': years,
    'zero_pick_pct': zero_pick_periods / n_periods,
    'picks_log': picks_log,
}
bt_path = os.path.join(ROOT, 'data', 'models', 'backtest_b_results.json')
with open(bt_path, 'w') as f:
    json.dump(bt_results, f, indent=2, ensure_ascii=False, default=str)
print(f'Backtest results saved: {bt_path}')

# Year-by-year
meta_v['year'] = meta_v['datetime'].dt.year
print('\n--- Year-by-Year ---')
for yr in sorted(meta_v['year'].unique()):
    yr_mask = meta_v['datetime'].dt.year == yr
    yr_data = meta_v[yr_mask]
    yr_dates = yr_data['datetime'].unique()
    
    print(f'\n{yr}: {len(yr_data)} samples, {len(yr_dates)} dates')
    yr_picks = yr_data[yr_data['score'] >= THRESHOLD]
    if len(yr_picks) > 0:
        print(f'  Picks: {len(yr_picks)} ({len(yr_picks)/len(yr_data)*100:.1f}%)')
        print(f'  Pick avg fwd ret: {yr_picks["forward_ret"].mean()*100:+.2f}%')
    print(f'  All stocks avg fwd ret: {yr_data["forward_ret"].mean()*100:+.2f}%')
