"""
Dual-Model + Market Regime Backtest

Architecture:
  1. Two parallel models: 5d_10% (precision) and 10d_15% (deep value)
  2. Market regime classifier (CSI300-based)
  3. Regime-dependent: model selection + threshold + position sizing
  4. A-share transaction costs

Walk-Forward: 3 windows, each with internal 80/20 train/val split
"""
import os, sys, json, warnings, pickle, time
import numpy as np, pandas as pd
import lightgbm as lgb
import tushare as ts

warnings.filterwarnings('ignore')
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
PRICE = r'D:\AIHedgeFund\ai-hedge-fund-main\data\cache\backtest_prices_extended.pkl'
TU_TOKEN = os.getenv('TUSHARE_PRO_TOKEN', '')
np.random.seed(42)

with open(os.path.join(ROOT, 'src', 'surge', 'params.json')) as f:
    LGB_PARAMS = json.load(f)['lgbm_params'].copy()

# ── Costs (A-share) ──────────────────────────────────────────────
COMMISSION = 0.0003
STAMP = 0.0005
SLIPPAGE = 0.001
COST_ENTRY = COMMISSION + SLIPPAGE       # 0.13%
COST_EXIT = COMMISSION + STAMP + SLIPPAGE # 0.18%
COST_RT = COST_ENTRY + COST_EXIT          # 0.31%

# ── Model configurations ─────────────────────────────────────────
MODEL_CFG = [
    {'name': '5d_10%',  'horizon': 5,  'thresh': 0.10},
    {'name': '10d_15%', 'horizon': 10, 'thresh': 0.15},
]

# Walk-Forward windows (train_start, train_end, test_start, test_end, label)
WF_WINDOWS = [
    ('2019-01-01', '2022-01-01', '2022-01-01', '2023-01-01', '2022 Bear'),
    ('2019-01-01', '2023-01-01', '2023-01-01', '2024-01-01', '2023 Sideways'),
    ('2019-01-01', '2024-01-01', '2024-01-01', '2025-07-01', '2024-2025 Recovery'),
]

print("=" * 72)
print("DUAL-MODEL + MARKET REGIME BACKTEST (v1)")
print("=" * 72)
t0 = time.time()

# ════════════════════════════════════════════════════════════════════
# STEP 1: Load Data
# ════════════════════════════════════════════════════════════════════
print("\n[1] Loading factor data...")
FACTOR_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
dfs = []
for fn in sorted(os.listdir(FACTOR_DIR)):
    if fn.endswith('.parquet'):
        dfs.append(pd.read_parquet(os.path.join(FACTOR_DIR, fn)))
pdf = pd.concat(dfs, ignore_index=True)
pdf['date'] = pd.to_datetime(pdf['datetime'])
pdf = pdf.sort_values(['vt_symbol', 'date']).reset_index(drop=True)
print(f"  Factors: {len(pdf):,} rows, {pdf['vt_symbol'].nunique()} stocks, dates {pdf['date'].min().date()} ~ {pdf['date'].max().date()}")

print("  Loading price data...")
price_dict = pd.read_pickle(PRICE)
print(f"  Prices: {len(price_dict)} stocks")

# Feature list
ALL_FEATURES = [c for c in pdf.columns if c.startswith(('alpha','rsi_','macd','bb_','klen','rsqr','slope','std','vma','vosc','beta_'))]
print(f"  {len(ALL_FEATURES)} features")

# ════════════════════════════════════════════════════════════════════
# STEP 2: Fetch CSI300 for Regime Detection
# ════════════════════════════════════════════════════════════════════
print("\n[2] Fetching CSI300 for regime detection...")
pro = ts.pro_api(TU_TOKEN)
csi = pro.index_daily(ts_code='000300.SH', start_date='20150101', end_date='20250701')
csi['trade_date'] = pd.to_datetime(csi['trade_date'])
csi = csi.sort_values('trade_date').reset_index(drop=True)
csi['ma20'] = csi['close'].rolling(20).mean()
csi['ma60'] = csi['close'].rolling(60).mean()
csi['ma120'] = csi['close'].rolling(120).mean()
csi['ret20'] = csi['close'].pct_change(20)
csi['vol20'] = csi['close'].pct_change().rolling(20).std()

def classify_regime(row):
    """Classify market regime using trailing data (no lookahead)."""
    c = row['close']; m20 = row['ma20']; m60 = row['ma60']
    m120 = row['ma120']; r20 = row['ret20']; v = row['vol20']
    if pd.isna(m60) or pd.isna(m120):
        return 'unknown'
    # Severe bear: deep below long-term MA
    if c < m120 * 0.90:
        return 'severe_bear'
    # Bear: below both MAs with negative momentum
    if c < m60 and c < m120 and r20 < -0.03:
        return 'bear'
    # Strong bull: above all MAs with positive momentum
    if c > m20 > m60 and r20 > 0.03:
        return 'bull'
    # Recovery: MA20 crossed up through MA60 recently
    if c > m60 and r20 > 0:
        return 'recovery'
    # Sideways: everything else
    return 'sideways'

csi['regime'] = csi.apply(classify_regime, axis=1)
regime_counts = csi['regime'].value_counts()
print(f"  CSI300: {len(csi)} trading days")
print(f"  Regime distribution:")
for r, c in regime_counts.items():
    print(f"    {r}: {c} days")
csi.to_pickle(os.path.join(ROOT, 'data', 'cache', 'csi300_regime.pkl'))
print(f"  Saved regime data")

# ════════════════════════════════════════════════════════════════════
# STEP 3: Forward returns (for both horizons)
# ════════════════════════════════════════════════════════════════════
print("\n[3] Computing forward returns...")
for mc in MODEL_CFG:
    h = mc['horizon']
    col = f'fwd_{h}d'
    pdf[col] = np.nan
    for sym, idx in pdf.groupby('vt_symbol', sort=False).indices.items():
        idx = sorted(idx)
        closes = pdf.loc[idx, 'close'].values
        n = len(closes)
        if n > h:
            fwd = np.full(n, np.nan)
            fwd[:-h] = (closes[h:] - closes[:-h]) / closes[:-h]
            pdf.loc[idx, col] = fwd

# ════════════════════════════════════════════════════════════════════
# STEP 4: Walk-Forward — Train both models per window
# ════════════════════════════════════════════════════════════════════
print("\n[4] Training models & backtesting...")
all_trades = []
all_equity = []

def get_price(sym, dt, field='close'):
    """Fetch price from dict-of-dfs. Returns scalar or None."""
    if sym not in price_dict:
        return None
    p = price_dict[sym]
    m = p['date'] == pd.Timestamp(dt)
    if m.sum() == 0:
        return None
    return p.loc[m, field].iloc[0]

def get_regime(dt):
    """Get regime on or just before dt."""
    m = csi['trade_date'] <= pd.Timestamp(dt)
    if m.sum() == 0:
        return 'sideways'
    return csi.loc[m, 'regime'].iloc[-1]

# ── Regime → strategy mapping ──────────────────────────────────
# (model_name, threshold, max_positions, capacity_pct)
REGIME_STRAT = {
    'bull':       ('5d_10%',  0.25, 5,  0.80),
    'recovery':   ('10d_15%', 0.30, 3,  0.60),
    'sideways':   ('5d_10%',  0.25, 5,  0.80),
    'bear':       ('10d_15%', 0.30, 3,  0.40),
    'severe_bear':('5d_10%',  0.40, 2,  0.15),
    'unknown':    ('5d_10%',  0.25, 3,  0.50),
}

for wi, (tr_s, tr_e, te_s, te_e, wname) in enumerate(WF_WINDOWS):
    print(f"\n  ── Window {wi+1}: {wname} ({te_s} ~ {te_e}) ──")
    tw = time.time()

    tr_mask = (pdf['date'] >= tr_s) & (pdf['date'] < tr_e)
    te_mask = (pdf['date'] >= te_s) & (pdf['date'] < te_e)

    te_dates = sorted(pdf.loc[te_mask, 'date'].unique())
    regime_log = {d: get_regime(d) for d in te_dates}
    regime_dist = pd.Series(list(regime_log.values())).value_counts()
    print(f"    Regimes during test: {dict(regime_dist)}")

    # ── Train both models ──
    models = {}
    for mc in MODEL_CFG:
        name = mc['name']
        h = mc['horizon']
        t = mc['thresh']
        fwd_col = f'fwd_{h}d'

        # Training data
        X_tr_wide = pdf.loc[tr_mask, ALL_FEATURES].values
        X_tr = np.where(np.isinf(X_tr_wide), np.nan, X_tr_wide)
        y_tr = ((pdf.loc[tr_mask, fwd_col].fillna(0) >= t).astype(int)).values
        keep_tr = ~np.isnan(X_tr).all(axis=1) & ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[keep_tr], y_tr[keep_tr]

        # Validation: last 20% by date
        tr_dates = sorted(pdf.loc[tr_mask, 'date'].unique())
        vl_cut = tr_dates[int(len(tr_dates) * 0.8)]
        vl_mask = (pdf['date'] >= vl_cut) & (pdf['date'] < tr_e)
        X_vl_wide = pdf.loc[vl_mask, ALL_FEATURES].values
        X_vl = np.where(np.isinf(X_vl_wide), np.nan, X_vl_wide)
        y_vl = ((pdf.loc[vl_mask, fwd_col].fillna(0) >= t).astype(int)).values
        keep_vl = ~np.isnan(X_vl).all(axis=1) & ~np.isnan(y_vl)
        X_vl, y_vl = X_vl[keep_vl], y_vl[keep_vl]

        w_model = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(X_tr, y_tr),
            valid_sets=[lgb.Dataset(X_vl, y_vl, reference=lgb.Dataset(X_tr, y_tr))],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        models[name] = {
            'model': w_model,
            'features': ALL_FEATURES,
        }
        print(f"    {name}: trained (val AUC ~{w_model.best_score['valid_0']['auc']:.4f})")

    # ── Score all test data with both models ──
    for mc in MODEL_CFG:
        name = mc['name']
        h = mc['horizon']
        fwd_col = f'fwd_{h}d'
        te_idx = pdf.loc[te_mask].index
        X_te_wide = pdf.loc[te_mask, ALL_FEATURES].values
        X_te = np.where(np.isinf(X_te_wide), np.nan, X_te_wide)
        pdf.loc[te_idx, f'score_{name}'] = models[name]['model'].predict(X_te)

    # ── Trade simulation ──
    positions = {}
    cash = 1_000_000
    window_trades = []
    portfolio_values = []

    for di, d in enumerate(te_dates):
        day_mask = pdf['date'] == d
        regime = regime_log.get(d, 'sideways')
        strat = REGIME_STRAT.get(regime, REGIME_STRAT['sideways'])
        model_name, threshold, max_pos, capacity = strat

        # ── Sell matured positions ──
        to_sell = [sym for sym, pos in positions.items() if pos['sell_date'] <= d]
        for sym in to_sell:
            pos = positions.pop(sym)
            sell_price = get_price(sym, d, 'close')
            if sell_price is None:
                continue
            gross_ret = sell_price / pos['entry_price'] - 1
            net_ret = gross_ret - COST_RT
            cash += pos['shares'] * sell_price
            window_trades.append({
                'window': wname, 'regime': regime, 'model': model_name,
                'symbol': sym,
                'entry_date': str(pd.Timestamp(pos['entry_date']).date()),
                'exit_date': str(pd.Timestamp(d).date()),
                'entry_price': pos['entry_price'], 'exit_price': sell_price,
                'gross_return': gross_ret, 'net_return': net_ret,
                'shares': pos['shares'], 'cost_ratio': COST_RT,
            })

        # ── Select new picks ──
        cap_cash = cash * capacity  # regime-dependent buying power
        open_slots = max(0, max_pos - len(positions))
        if open_slots > 0 and di < len(te_dates) - MODEL_CFG[0]['horizon']:
            day_scores = pdf.loc[day_mask, f'score_{model_name}'].values
            day_syms = pdf.loc[day_mask, 'vt_symbol'].values
            next_date = te_dates[min(di + 1, len(te_dates) - 1)]

            # Filter: not already held, valid entry price
            valid_syms = []; valid_scores = []; valid_prices = []
            for j, sym in enumerate(day_syms):
                if sym in positions:
                    continue
                ep = get_price(sym, next_date, 'open')
                if ep is None or ep <= 0:
                    continue
                valid_syms.append(sym)
                valid_scores.append(day_scores[j])
                valid_prices.append(ep)

            if len(valid_syms) == 0:
                continue

            valid_scores = np.array(valid_scores)
            candidates = valid_scores >= threshold
            if candidates.sum() == 0:
                continue

            pick_scores = valid_scores[candidates]
            pick_syms = [valid_syms[k] for k in range(len(valid_syms)) if candidates[k]]
            pick_prices = [valid_prices[k] for k in range(len(valid_prices)) if candidates[k]]

            # Sort by score, take top N
            order = np.argsort(-pick_scores)[:open_slots]
            pick_syms = [pick_syms[k] for k in order]
            pick_prices = [pick_prices[k] for k in order]

            sell_date = te_dates[min(di + MODEL_CFG[0]['horizon'], len(te_dates) - 1)]
            per_stock = cap_cash / len(pick_syms)

            for k, sym in enumerate(pick_syms):
                ep = pick_prices[k]
                shares = int(per_stock / (ep * 100)) * 100
                if shares < 100:
                    continue
                cost = shares * ep * COST_ENTRY
                cash -= shares * ep + cost
                positions[sym] = {
                    'shares': shares, 'entry_price': ep,
                    'entry_date': d, 'sell_date': sell_date,
                }

        # ── Daily mark-to-market ──
        pos_value = 0
        for sym, pos in list(positions.items()):
            p = get_price(sym, d, 'close')
            if p is not None:
                pos_value += pos['shares'] * p
        total = cash + pos_value
        portfolio_values.append({'date': d, 'value': total, 'cash': cash, 'positions': pos_value,
                                 'regime': regime, 'model': model_name})

    # ── Window analysis ──
    df_val = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(window_trades)
    n_trades = len(window_trades)

    if n_trades > 0:
        cum_ret = df_val['value'].iloc[-1] / df_val['value'].iloc[0] - 1
        df_val['ret'] = df_val['value'].pct_change().fillna(0)
        sharpe = df_val['ret'].mean() / df_val['ret'].std() * np.sqrt(240) if df_val['ret'].std() > 0 else 0
        df_val['cummax'] = df_val['value'].cummax()
        max_dd = (df_val['value'] / df_val['cummax'] - 1).min()
        win_rate = (trades_df['net_return'] > 0).mean()
        avg_nr = trades_df['net_return'].mean()
        pf = ((trades_df.loc[trades_df['net_return'] > 0, 'net_return'].mean() * (trades_df['net_return'] > 0).sum())
              / abs(trades_df.loc[trades_df['net_return'] < 0, 'net_return'].mean() * (trades_df['net_return'] < 0).sum())
              if (trades_df['net_return'] < 0).sum() > 0 else float('inf'))

        print(f"    Trades: {n_trades} | WR: {win_rate:.1%} | AvgNet: {avg_nr:.2%}")
        print(f"    CumRet: {cum_ret:.2%} | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.2%} | PF: {pf:.2f}")
    else:
        print(f"    0 trades")
        cum_ret = sharpe = max_dd = win_rate = avg_nr = pf = 0

    df_val['window'] = wname
    all_equity.append(df_val)
    all_trades.extend(window_trades)
    print(f"    {(time.time()-tw)/60:.1f} min")

# ════════════════════════════════════════════════════════════════════
# STEP 5: Full Period Summary
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("OVERALL RESULTS")
print("=" * 72)

df_trades = pd.DataFrame(all_trades)
df_equity_all = pd.concat(all_equity, ignore_index=True).sort_values('date')

if len(df_trades) > 0:
    df_trades['net_return'] = df_trades['net_return'].clip(-0.30, 0.30)
    wr = (df_trades['net_return'] > 0).mean()
    avg_nr = df_trades['net_return'].mean()
    avg_w = df_trades.loc[df_trades['net_return'] > 0, 'net_return'].mean()
    avg_l = df_trades.loc[df_trades['net_return'] < 0, 'net_return'].mean()
    pf = (avg_w * (df_trades['net_return'] > 0).sum()) / abs(avg_l * (df_trades['net_return'] < 0).sum()) if avg_l != 0 else float('inf')

    print(f"\n  Total trades: {len(df_trades)}")
    print(f"  Win rate: {wr:.1%}")
    print(f"  Avg net return: {avg_nr:.4%}")
    print(f"  Avg win: {avg_w:.4%} | Avg loss: {avg_l:.4%}")
    print(f"  Profit factor: {pf:.2f}")

    # Model breakdown
    print(f"\n  By model:")
    for m in ['5d_10%', '10d_15%']:
        sub = df_trades[df_trades['model'] == m]
        if len(sub) > 0:
            print(f"    {m}: {len(sub)} trades, WR {(sub['net_return']>0).mean():.1%}, avg {sub['net_return'].mean():.4%}")

    # Regime breakdown
    print(f"\n  By regime:")
    for r in sorted(df_trades['regime'].unique()):
        sub = df_trades[df_trades['regime'] == r]
        if len(sub) > 0:
            print(f"    {r}: {len(sub)} trades, WR {(sub['net_return']>0).mean():.1%}, avg {sub['net_return'].mean():.4%}")

if len(df_equity_all) > 0:
    eq = df_equity_all
    eq['equity'] = eq['value'] / eq['value'].iloc[0]
    eq['ret'] = eq['value'].pct_change().fillna(0)
    full_cr = eq['value'].iloc[-1] / eq['value'].iloc[0] - 1
    full_sr = eq['ret'].mean() / eq['ret'].std() * np.sqrt(240) if eq['ret'].std() > 0 else 0
    eq['cummax'] = eq['value'].cummax()
    full_mdd = (eq['value'] / eq['cummax'] - 1).min()

    print(f"\n  Combined equity:")
    print(f"    Cumulative return: {full_cr:.2%}")
    print(f"    Sharpe: {full_sr:.2f}")
    print(f"    Max DD: {full_mdd:.2%}")

# ════════════════════════════════════════════════════════════════════
# STEP 6: Save Results
# ════════════════════════════════════════════════════════════════════
print(f"\n[5] Saving results...")
results = {
    'version': 'dual_model_v1',
    'cost_roundtrip': COST_RT,
    'total_trades': len(df_trades) if len(df_trades) > 0 else 0,
    'win_rate': float(wr) if len(df_trades) > 0 else 0,
    'avg_net_return': float(avg_nr) if len(df_trades) > 0 else 0,
    'profit_factor': float(pf) if len(df_trades) > 0 else 0,
    'full_cum_return': float(full_cr) if len(df_equity_all) > 0 else 0,
    'full_sharpe': float(full_sr) if len(df_equity_all) > 0 else 0,
    'full_max_dd': float(full_mdd) if len(df_equity_all) > 0 else 0,
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return super().default(obj)

os.makedirs(os.path.join(ROOT, 'quant_archive', '2026-05'), exist_ok=True)
with open(os.path.join(ROOT, 'data', 'models', 'backtest_dual_model_results.json'), 'w') as f:
    json.dump(results, f, indent=2, cls=NpEncoder, ensure_ascii=False)

if len(df_trades) > 0:
    df_trades.to_csv(os.path.join(ROOT, 'quant_archive', '2026-05', 'backtest_dual_trades.csv'), index=False)
if len(df_equity_all) > 0:
    df_equity_all.to_csv(os.path.join(ROOT, 'quant_archive', '2026-05', 'backtest_dual_equity.csv'), index=False)

print(f"  Total runtime: {(time.time()-t0)/60:.1f} min")
print("Done!")
