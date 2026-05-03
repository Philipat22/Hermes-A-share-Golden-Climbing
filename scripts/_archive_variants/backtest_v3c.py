"""
Backtest v3c — Targeted Drawdown Protection

Changes from v2:
  1. SIDEWAYS capacity: 60% → 30%
  2. SIDEWAYS max_pos: 5 → 3
  3. SIDEWAYS top-20% score filter (only buy highest-confidence picks)
  4. NEW: Monthly loss limit — if portfolio drops >-15% in a calendar month, 
     halt until next month (catches Dec 2024-style disasters)
     
NOT included (noise reduction):
  - No circuit breaker (triggers too often at 33% WR)
  - No repeat-loser ban (good concept but over-restrictive in mean-reversion)

NOTE: Total equity curve across walk-forward windows is discontinuous
(each window starts fresh at 1M). Compare by INDIVIDUAL WINDOW returns.
"""
import os, sys, json, warnings, pickle, time
sys.stdout.reconfigure(line_buffering=True)
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

# Costs
COMMISSION, STAMP, SLIPPAGE = 0.0003, 0.0005, 0.001
COST_ENTRY = COMMISSION + SLIPPAGE
COST_EXIT  = COMMISSION + STAMP + SLIPPAGE
COST_RT    = COST_ENTRY + COST_EXIT
STOP_LOSS  = -0.05

# v3c parameters
SIDEWAYS_CAPACITY = 0.30       # v2 was 0.60
SIDEWAYS_MAX_POS = 3           # v2 was 5
SIDEWAYS_TOP_PCTILE = 0.20     # only top 20% in sideways
MONTHLY_LOSS_LIMIT = -0.15     # halt if portfolio drops >15% in a month

MODEL_CFG = [
    {'name': '5d_10%',  'horizon': 5,  'thresh': 0.10},
    {'name': '10d_15%', 'horizon': 10, 'thresh': 0.15},
]
WF_WINDOWS = [
    ('2019-01-01', '2022-01-01', '2022-01-01', '2023-01-01', '2022 Bear'),
    ('2019-01-01', '2023-01-01', '2023-01-01', '2024-01-01', '2023 Sideways'),
    ('2019-01-01', '2024-01-01', '2024-01-01', '2025-07-01', '2024-2025'),
]

def build_strat(mode='5d_only'):
    base = {
        'bull':        ('5d_10%',  0.25, 5,  0.80),
        'sideways':    ('5d_10%',  0.25, SIDEWAYS_MAX_POS, SIDEWAYS_CAPACITY),
        'bear':        ('5d_10%',  0.30, 3,  0.30),
        'severe_bear': ('5d_10%',  0.40, 2,  0.15),
        'recovery':    ('5d_10%',  0.40, 0,  0.00),
        'unknown':     ('5d_10%',  0.25, 3,  0.50),
    }
    if mode == 'dual':
        base['bear'] = ('10d_15%', 0.30, 3, 0.30)
        base['severe_bear'] = ('5d_10%', 0.40, 2, 0.15)
    return base

# ════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════
print("=" * 72)
print("BACKTEST v3c — Targeted Drawdown Protection")
print(f"  Sideways: cap={SIDEWAYS_CAPACITY:.0%}, max_pos={SIDEWAYS_MAX_POS}, top{(1-SIDEWAYS_TOP_PCTILE)*100:.0f}%")
print(f"  Monthly loss limit: {MONTHLY_LOSS_LIMIT:.0%}")
print("=" * 72)

t0 = time.time()

print("[Loading factors...]")
FACTOR_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
dfs = []
for fn in sorted(os.listdir(FACTOR_DIR)):
    if fn.endswith('.parquet'):
        dfs.append(pd.read_parquet(os.path.join(FACTOR_DIR, fn)))
pdf = pd.concat(dfs, ignore_index=True)
pdf['date'] = pd.to_datetime(pdf['datetime'])
pdf = pdf.sort_values(['vt_symbol', 'date']).reset_index(drop=True)
print(f"  {len(pdf):,} rows, {pdf['vt_symbol'].nunique()} stocks")

price_dict = pd.read_pickle(PRICE)
print(f"  {len(price_dict)} stocks in price dict")

ALL_FEATURES = [c for c in pdf.columns if c.startswith(
    ('alpha','rsi_','macd','bb_','klen','rsqr','slope','std','vma','vosc','beta_'))]
print(f"  {len(ALL_FEATURES)} features")

# CSI300 regime
print("[Computing CSI300 regimes...]")
pro = ts.pro_api(TU_TOKEN)
csi = pro.index_daily(ts_code='000300.SH', start_date='20150101', end_date='20250701')
csi['trade_date'] = pd.to_datetime(csi['trade_date'])
csi = csi.sort_values('trade_date').reset_index(drop=True)
for m in [20, 60, 120]:
    csi[f'ma{m}'] = csi['close'].rolling(m).mean()
csi['ret20'] = csi['close'].pct_change(20)

def classify_regime(row):
    c = row['close']; m20 = row['ma20']; m60 = row['ma60']; m120 = row['ma120']; r20 = row['ret20']
    if pd.isna(m60) or pd.isna(m120): return 'unknown'
    if c < m120 * 0.90: return 'severe_bear'
    if c < m60 and c < m120 and r20 < -0.03: return 'bear'
    if c > m20 > m60 and r20 > 0.03: return 'bull'
    if c > m60 and r20 > 0: return 'recovery'
    return 'sideways'

csi['regime'] = csi.apply(classify_regime, axis=1)
csi.to_pickle(os.path.join(ROOT, 'data', 'cache', 'csi300_regime.pkl'))

# Forward returns
print("[Computing forward returns...]")
for mc in MODEL_CFG:
    h = mc['horizon']; col = f'fwd_{h}d'
    pdf[col] = np.nan
    for sym, idx in pdf.groupby('vt_symbol', sort=False).indices.items():
        idx = sorted(idx); closes = pdf.loc[idx, 'close'].values; n = len(closes)
        if n > h:
            fwd = np.full(n, np.nan); fwd[:-h] = (closes[h:] - closes[:-h]) / closes[:-h]
            pdf.loc[idx, col] = fwd

# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def get_price(sym, dt, field='close'):
    if sym not in price_dict: return None
    p = price_dict[sym]
    m = p['date'] == pd.Timestamp(dt)
    return p.loc[m, field].iloc[0] if m.sum() > 0 else None

def get_regime(dt):
    m = csi['trade_date'] <= pd.Timestamp(dt)
    return csi.loc[m, 'regime'].iloc[-1] if m.sum() > 0 else 'sideways'

# ════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE (v3c)
# ════════════════════════════════════════════════════════════════════
def run_backtest(mode_label, strategy):
    print(f"\n{'='*72}")
    print(f"  MODE: {mode_label}")
    print(f"{'='*72}")

    all_trades = []; all_equity = []

    for wi, (tr_s, tr_e, te_s, te_e, wname) in enumerate(WF_WINDOWS):
        tr_mask = (pdf['date'] >= tr_s) & (pdf['date'] < tr_e)
        te_mask = (pdf['date'] >= te_s) & (pdf['date'] < te_e)
        te_dates = sorted(pdf.loc[te_mask, 'date'].unique())

        # ── Train models ──
        models = {}
        for mc in MODEL_CFG:
            name = mc['name']; h = mc['horizon']; t = mc['thresh']
            fwd_col = f'fwd_{h}d'
            X_tr = np.where(np.isinf(pdf.loc[tr_mask, ALL_FEATURES].values), np.nan,
                            pdf.loc[tr_mask, ALL_FEATURES].values)
            y_tr = ((pdf.loc[tr_mask, fwd_col].fillna(0) >= t).astype(int)).values
            keep = ~np.isnan(X_tr).all(axis=1) & ~np.isnan(y_tr)
            X_tr, y_tr = X_tr[keep], y_tr[keep]

            tr_dates = sorted(pdf.loc[tr_mask, 'date'].unique())
            vl_cut = tr_dates[int(len(tr_dates) * 0.8)]
            vl_mask = (pdf['date'] >= vl_cut) & (pdf['date'] < tr_e)
            X_vl = np.where(np.isinf(pdf.loc[vl_mask, ALL_FEATURES].values), np.nan,
                           pdf.loc[vl_mask, ALL_FEATURES].values)
            y_vl = ((pdf.loc[vl_mask, fwd_col].fillna(0) >= t).astype(int)).values
            keep_vl = ~np.isnan(X_vl).all(axis=1) & ~np.isnan(y_vl)
            X_vl, y_vl = X_vl[keep_vl], y_vl[keep_vl]

            w_model = lgb.train(
                LGB_PARAMS, lgb.Dataset(X_tr, y_tr),
                valid_sets=[lgb.Dataset(X_vl, y_vl, reference=lgb.Dataset(X_tr, y_tr))],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            models[name] = w_model

        # ── Score test data ──
        te_idx = pdf.loc[te_mask].index
        for mc in MODEL_CFG:
            name = mc['name']
            X_te = np.where(np.isinf(pdf.loc[te_mask, ALL_FEATURES].values), np.nan,
                           pdf.loc[te_mask, ALL_FEATURES].values)
            pdf.loc[te_idx, f'score_{name}'] = models[name].predict(X_te)

        # ── TRADE SIMULATION ──
        positions = {}; cash = 1_000_000
        window_trades = []; portfolio_values = []
        monthly_start_value = None
        monthly_halted = False
        current_month = None
        monthly_limit_triggered = False  # track if it was triggered this window

        for di, d in enumerate(te_dates):
            day_mask = pdf['date'] == d
            regime = get_regime(d)
            strat = strategy.get(regime, strategy['sideways'])
            model_name, threshold, max_pos, capacity = strat
            is_sideways = (regime == 'sideways')
            this_month = d.month

            # ── Monthly reset ──
            if current_month is None:
                current_month = this_month
                pos_value = sum(get_price(sym, d, 'close') * pos['shares']
                              for sym, pos in positions.items()
                              if get_price(sym, d, 'close') is not None)
                monthly_start_value = cash + pos_value
                monthly_halted = False
            elif this_month != current_month:
                # New month
                current_month = this_month
                pos_value = sum(get_price(sym, d, 'close') * pos['shares']
                              for sym, pos in positions.items()
                              if get_price(sym, d, 'close') is not None)
                monthly_start_value = cash + pos_value
                monthly_halted = False

            # ── CHECK monthly loss limit ──
            if monthly_start_value is not None and not monthly_halted:
                pos_value_now = sum(get_price(sym, d, 'close') * pos['shares']
                                  for sym, pos in positions.items()
                                  if get_price(sym, d, 'close') is not None)
                total_now = cash + pos_value_now
                month_loss = (total_now / monthly_start_value) - 1
                if month_loss <= MONTHLY_LOSS_LIMIT:
                    monthly_halted = True
                    monthly_limit_triggered = True

            # ── CHECK STOP-LOSSES (always, even during halt) ──
            for sym in list(positions.keys()):
                pos = positions[sym]
                low = get_price(sym, d, 'low')
                if low is None: continue
                dd = (low / pos['entry_price'] - 1)
                if dd <= STOP_LOSS:
                    stop_price = pos['entry_price'] * (1 + STOP_LOSS)
                    gross_ret = STOP_LOSS
                    cash += pos['shares'] * stop_price * (1 - COST_EXIT)
                    window_trades.append({
                        'window': wname, 'regime': regime, 'model': model_name,
                        'symbol': sym, 'exit_reason': 'stop_loss',
                        'entry_date': str(pd.Timestamp(pos['entry_date']).date()),
                        'exit_date': str(pd.Timestamp(d).date()),
                        'entry_price': pos['entry_price'], 'exit_price': stop_price,
                        'gross_return': gross_ret,
                        'net_return': gross_ret - COST_RT,
                        'shares': pos['shares'], 'cost_ratio': COST_RT,
                    })
                    del positions[sym]

            # ── SELL matured positions ──
            to_sell = [sym for sym, pos in positions.items() if pos['sell_date'] <= d]
            for sym in to_sell:
                pos = positions.pop(sym)
                sell_price = get_price(sym, d, 'close')
                if sell_price is None: continue
                gross_ret = sell_price / pos['entry_price'] - 1
                net_ret = gross_ret - COST_RT
                cash += pos['shares'] * sell_price * (1 - COST_EXIT)
                window_trades.append({
                    'window': wname, 'regime': regime, 'model': model_name,
                    'symbol': sym, 'exit_reason': 'matured',
                    'entry_date': str(pd.Timestamp(pos['entry_date']).date()),
                    'exit_date': str(pd.Timestamp(d).date()),
                    'entry_price': pos['entry_price'], 'exit_price': sell_price,
                    'gross_return': gross_ret, 'net_return': net_ret,
                    'shares': pos['shares'], 'cost_ratio': COST_RT,
                })

            # ── BUY selection (skip if halted or no slots) ──
            if max_pos > 0 and not monthly_halted:
                cap_cash = cash * capacity
                open_slots = max(0, max_pos - len(positions))
                if open_slots > 0 and di < len(te_dates) - MODEL_CFG[0]['horizon']:
                    day_scores = pdf.loc[day_mask, f'score_{model_name}'].values
                    day_syms = pdf.loc[day_mask, 'vt_symbol'].values
                    next_date = te_dates[min(di + 1, len(te_dates) - 1)]

                    valid = [(j, sym, day_scores[j], get_price(sym, next_date, 'open'))
                             for j, sym in enumerate(day_syms)
                             if sym not in positions and get_price(sym, next_date, 'open') is not None]

                    if len(valid) > 0:
                        scores = np.array([v[2] for v in valid])
                        prices = [v[3] for v in valid]
                        syms = [v[1] for v in valid]

                        cand_mask = scores >= threshold
                        if cand_mask.sum() > 0:
                            cand_scores = scores[cand_mask]
                            cand_syms = [syms[k] for k in range(len(syms)) if cand_mask[k]]
                            cand_prices = [prices[k] for k in range(len(syms)) if cand_mask[k]]

                            # v3c: In SIDEWAYS, only buy top 20% by score
                            if is_sideways and len(cand_scores) > open_slots:
                                pct_val = np.percentile(cand_scores, (1 - SIDEWAYS_TOP_PCTILE) * 100)
                                strict_mask = cand_scores >= pct_val
                                if strict_mask.sum() > 0:
                                    cand_scores = cand_scores[strict_mask]
                                    cand_syms = [cand_syms[k] for k in range(len(cand_syms)) if strict_mask[k]]
                                    cand_prices = [cand_prices[k] for k in range(len(cand_prices)) if strict_mask[k]]

                            order = np.argsort(-cand_scores)[:open_slots]
                            pick_syms = [cand_syms[k] for k in order]
                            pick_prices = [cand_prices[k] for k in order]

                            sell_date = te_dates[min(di + MODEL_CFG[0]['horizon'],
                                                     len(te_dates) - 1)]
                            per_stock = cap_cash / len(pick_syms) if len(pick_syms) > 0 else 0

                            for k, sym in enumerate(pick_syms):
                                ep = pick_prices[k]
                                shares = int(per_stock / (ep * 100)) * 100
                                if shares < 100: continue
                                cash -= shares * ep * (1 + COST_ENTRY)
                                positions[sym] = {
                                    'shares': shares, 'entry_price': ep,
                                    'entry_date': d, 'sell_date': sell_date,
                                }

            # ── Daily mark-to-market ──
            pos_value = 0
            for sym, pos in list(positions.items()):
                p = get_price(sym, d, 'close')
                if p is not None: pos_value += pos['shares'] * p
            total = cash + pos_value
            portfolio_values.append({
                'date': d, 'value': total, 'cash': cash, 'positions': pos_value,
                'regime': regime, 'model': model_name, 'num_positions': len(positions),
                'monthly_halted': monthly_halted,
            })

        # ── Window stats ──
        df_val = pd.DataFrame(portfolio_values)
        trades_df = pd.DataFrame(window_trades)
        n = len(window_trades)
        if n > 0:
            cr = df_val['value'].iloc[-1] / df_val['value'].iloc[0] - 1
            df_val['ret'] = df_val['value'].pct_change().fillna(0)
            sr = df_val['ret'].mean() / df_val['ret'].std() * np.sqrt(240) if df_val['ret'].std() > 0 else 0
            df_val['cummax'] = df_val['value'].cummax()
            mdd = (df_val['value'] / df_val['cummax'] - 1).min()
            wr = (trades_df['net_return'] > 0).mean()
            avg_nr = trades_df['net_return'].mean()
            avg_w = trades_df.loc[trades_df['net_return'] > 0, 'net_return'].mean()
            avg_l = trades_df.loc[trades_df['net_return'] < 0, 'net_return'].mean()
            pf = (avg_w * (trades_df['net_return'] > 0).sum()) / abs(avg_l * (trades_df['net_return'] < 0).sum()) if avg_l != 0 else float('inf')

            halted_days = df_val['monthly_halted'].sum()
            print(f"\n  ── {wname} ──")
            print(f"    Trades: {n} | WR: {wr:.1%} | AvgNet: {avg_nr:.2%} | MaxDD: {mdd:.2%}")
            print(f"    CumRet: {cr:.2%} | Sharpe: {sr:.2f} | PF: {pf:.2f}")
            print(f"    AvgWin: {avg_w:.2%} | AvgLoss: {avg_l:.2%}")
            print(f"    Monthly limit triggered: {monthly_limit_triggered} | Halted days: {int(halted_days)}")

            # Regime detail
            print(f"\n    By regime:")
            for r in sorted(trades_df['regime'].unique()):
                sub = trades_df[trades_df['regime'] == r]
                if len(sub) > 0:
                    print(f"      {r}: {len(sub)}t WR {(sub['net_return']>0).mean():.0%} avg{sub['net_return'].mean():.2%}")

        df_val['window'] = wname
        all_equity.append(df_val)
        all_trades.extend(window_trades)

    # ── Full period total (last window return) ──
    print(f"\n  {'='*54}")
    print(f"  TOTAL ({mode_label})")
    print(f"  {'='*54}")
    df_t = pd.DataFrame(all_trades)
    df_e = pd.concat(all_equity, ignore_index=True).sort_values('date')

    if len(df_t) > 0:
        df_t['net_return'] = df_t['net_return'].clip(-0.30, 0.30)
        wr = (df_t['net_return'] > 0).mean()
        avg_nr = df_t['net_return'].mean()
        avg_w = df_t.loc[df_t['net_return'] > 0, 'net_return'].mean()
        avg_l = df_t.loc[df_t['net_return'] < 0, 'net_return'].mean()
        pf = (avg_w * (df_t['net_return'] > 0).sum()) / abs(avg_l * (df_t['net_return'] < 0).sum()) if avg_l != 0 else float('inf')
        print(f"    Total trades: {len(df_t)} | WR: {wr:.1%} | AvgNet: {avg_nr:.4%} | PF: {pf:.2f}")

    return all_trades, all_equity

# ════════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════════
results = {}
for mode_name, mode_key in [('A) 5d_10% Single + v3c', '5d_only'),
                             ('B) Dual Model + v3c', 'dual')]:
    strat = build_strat(mode_key)
    trades, equity = run_backtest(mode_name, strat)
    results[mode_key] = {'trades': trades, 'equity': equity}

# ── Save ──
print(f"\n[Saving...]")
os.makedirs(os.path.join(ROOT, 'quant_archive', '2026-05'), exist_ok=True)
for key in ['5d_only', 'dual']:
    r = results[key]
    df_t = pd.DataFrame(r['trades'])
    df_e = pd.concat(r['equity'], ignore_index=True).sort_values('date')
    if len(df_t) > 0:
        df_t.to_csv(os.path.join(ROOT, 'quant_archive', '2026-05',
                                 f'backtest_v3c_{key}_trades.csv'), index=False)
    if len(df_e) > 0:
        df_e.to_csv(os.path.join(ROOT, 'quant_archive', '2026-05',
                                 f'backtest_v3c_{key}_equity.csv'), index=False)

# ── Compare ──
print(f"\n{'='*72}")
print("V3c vs V2 COMPARISON")
print(f"{'='*72}")
try:
    v2_trades = pd.read_csv(os.path.join(ROOT, 'quant_archive', '2026-05', 'backtest_v2_dual_trades.csv'))
    if len(v2_trades) > 0:
        v2_trades['net_return'] = v2_trades['net_return'].clip(-0.30, 0.30)
        print(f"\n  V2-B (baseline):")
        for regime in sorted(v2_trades['regime'].unique()):
            sub = v2_trades[v2_trades['regime'] == regime]
            print(f"    {regime:12s}: {len(sub):>3}t WR {(sub['net_return']>0).mean():.0%} avg{sub['net_return'].mean():.2%}")

    for key, label in [('5d_only', 'V3c-A'), ('dual', 'V3c-B')]:
        v3_trades = pd.DataFrame(results[key]['trades'])
        if len(v3_trades) > 0:
            v3_trades['net_return'] = v3_trades['net_return'].clip(-0.30, 0.30)
            print(f"\n  {label}:")
            for regime in sorted(v3_trades['regime'].unique()):
                old_sub = v2_trades[v2_trades['regime'] == regime]
                new_sub = v3_trades[v3_trades['regime'] == regime]
                delta_t = len(new_sub) - len(old_sub)
                delta_wr = (new_sub['net_return'] > 0).mean() - (old_sub['net_return'] > 0).mean()
                delta_avg = new_sub['net_return'].mean() - old_sub['net_return'].mean()
                print(f"    {regime:12s}: V2 {len(old_sub):>3}t → V3 {len(new_sub):>3}t ({delta_t:+.0f})"
                      f" WR {delta_wr:+.0%} avg {delta_avg:+.2%}")

except Exception as e:
    print(f"  Comparison error: {e}")

print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
print("Done!")
