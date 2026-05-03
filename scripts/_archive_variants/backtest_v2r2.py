"""
Regime-Aware Backtest v2: Risk Controls Edition

Runs TWO configurations sequentially:
  A) Single model (5d_10%) across all non-recovery regimes
  B) Dual model switching (5d_10% / 10d_15%) across regimes

Both include:
  - No trading during recovery (best to sit out)
  - Hard stop-loss at -5% per trade
  - Regime-dependent position sizing
  - Full A-share transaction costs (0.31% round-trip)
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

# ── Costs ────────────────────────────────────────────────────────
COMMISSION, STAMP, SLIPPAGE = 0.0003, 0.0005, 0.001
COST_ENTRY = COMMISSION + SLIPPAGE          # 0.13%
COST_EXIT  = COMMISSION + STAMP + SLIPPAGE   # 0.18%
COST_RT    = COST_ENTRY + COST_EXIT           # 0.31%
STOP_LOSS  = -0.05  # -5% per trade

# ── Model configs ────────────────────────────────────────────────
MODEL_CFG = [
    {'name': '5d_10%',  'horizon': 5,  'thresh': 0.10},
    {'name': '10d_15%', 'horizon': 10, 'thresh': 0.15},
]
WF_WINDOWS = [
    ('2019-01-01', '2022-01-01', '2022-01-01', '2023-01-01', '2022 Bear'),
    ('2019-01-01', '2023-01-01', '2023-01-01', '2024-01-01', '2023 Sideways'),
    ('2019-01-01', '2024-01-01', '2024-01-01', '2025-01-01', '2024 Bull'),
    ('2019-01-01', '2025-01-01', '2025-01-01', '2026-04-30', '2025-2026'),
]

# ── Strategy configs ─────────────────────────────────────────────
def build_strat(mode='5d_only'):
    """Return dict: regime → (model_name, threshold, max_positions, capacity_pct)."""
    base = {
        'bull':        ('5d_10%',  0.25, 5,  0.80),
        'sideways':    ('5d_10%',  0.35, 5,  0.60),
        'bear':        ('5d_10%',  0.30, 3,  0.30),
        'severe_bear': ('5d_10%',  0.40, 2,  0.15),
        'recovery':    ('5d_10%',  0.40, 0,  0.00),  # no trading
        'unknown':     ('5d_10%',  0.25, 3,  0.50),
    }
    if mode == 'dual':
        # Dual model: use 10d_15% in bear, 5d_10% elsewhere
        base['bear'] = ('10d_15%', 0.30, 3, 0.30)
        base['severe_bear'] = ('5d_10%', 0.40, 2, 0.15)
    return base

# ════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════
print("=" * 72)
print("REGIME-AWARE BACKTEST v2 — Risk Controls Edition")
print("=" * 72)
t0 = time.time()

print("\n[Loading data...]")
FACTOR_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
dfs = []
for fn in sorted(os.listdir(FACTOR_DIR)):
    if fn.endswith('.parquet'):
        dfs.append(pd.read_parquet(os.path.join(FACTOR_DIR, fn)))
pdf = pd.concat(dfs, ignore_index=True)
pdf['date'] = pd.to_datetime(pdf['datetime'])
pdf = pdf.sort_values(['vt_symbol', 'date']).reset_index(drop=True)
print(f"  Factors: {len(pdf):,} rows, {pdf['vt_symbol'].nunique()} stocks")

price_dict = pd.read_pickle(PRICE)
print(f"  Prices: {len(price_dict)} stocks")

ALL_FEATURES = [c for c in pdf.columns if c.startswith(('alpha','rsi_','macd','bb_','klen','rsqr','slope','std','vma','vosc','beta_'))]
print(f"  {len(ALL_FEATURES)} features")

# CSI300 regime data
print("\n[Fetching CSI300 regime data...]")
pro = ts.pro_api(TU_TOKEN)
csi = pro.index_daily(ts_code='000300.SH', start_date='20150101', end_date='20260430')
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
# BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════════
def get_price(sym, dt, field='close'):
    if sym not in price_dict: return None
    p = price_dict[sym]
    m = p['date'] == pd.Timestamp(dt)
    return p.loc[m, field].iloc[0] if m.sum() > 0 else None

def get_regime(dt):
    m = csi['trade_date'] <= pd.Timestamp(dt)
    return csi.loc[m, 'regime'].iloc[-1] if m.sum() > 0 else 'sideways'

def run_backtest(mode_label, strategy):
    """Run full Walk-Forward backtest for one strategy config."""
    print(f"\n{'='*72}")
    print(f"  MODE: {mode_label}")
    print(f"{'='*72}")
    all_trades = []; all_equity = []

    for wi, (tr_s, tr_e, te_s, te_e, wname) in enumerate(WF_WINDOWS):
        tr_mask = (pdf['date'] >= tr_s) & (pdf['date'] < tr_e)
        te_mask = (pdf['date'] >= te_s) & (pdf['date'] < te_e)
        te_dates = sorted(pdf.loc[te_mask, 'date'].unique())

        # ── Train both models ──
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
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )
            models[name] = w_model

        # ── Score all test data ──
        te_idx = pdf.loc[te_mask].index
        for mc in MODEL_CFG:
            name = mc['name']
            X_te = np.where(np.isinf(pdf.loc[te_mask, ALL_FEATURES].values), np.nan,
                           pdf.loc[te_mask, ALL_FEATURES].values)
            pdf.loc[te_idx, f'score_{name}'] = models[name].predict(X_te)

        # ── Trade simulation ──
        positions = {}; cash = 1_000_000
        window_trades = []; portfolio_values = []

        for di, d in enumerate(te_dates):
            day_mask = pdf['date'] == d
            regime = get_regime(d)
            strat = strategy.get(regime, strategy['sideways'])
            model_name, threshold, max_pos, capacity = strat

            # ── CHECK STOP-LOSSES (check BEFORE trading) ──
            for sym in list(positions.keys()):
                pos = positions[sym]
                low = get_price(sym, d, 'low')
                if low is None: continue
                dd = (low / pos['entry_price'] - 1)
                if dd <= STOP_LOSS:
                    # Stop triggered — sell at stop price (entry * 0.95)
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

            # ── SELECT new picks ──
            if max_pos > 0:  # skip if regime says no trading
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
                        cand = scores >= threshold
                        if cand.sum() > 0:
                            order = np.argsort(-scores[cand])[:open_slots]
                            pick_syms = [syms[k] for k in range(len(syms)) if cand[k]]
                            pick_syms = [pick_syms[k] for k in order]
                            pick_prices = [prices[k] for k in range(len(syms)) if cand[k]]
                            pick_prices = [pick_prices[k] for k in order]

                            sell_date = te_dates[min(di + MODEL_CFG[0]['horizon'], len(te_dates) - 1)]
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
            })

        # ── Window analysis ──
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
            avg_hold = (trades_df['exit_date'].apply(lambda x: pd.Timestamp(x)) -
                       trades_df['entry_date'].apply(lambda x: pd.Timestamp(x))).dt.days.mean()

            print(f"\n  ── {wname} ──")
            print(f"    Trades: {n} | WR: {wr:.1%} | AvgNet: {avg_nr:.2%}")
            print(f"    CumRet: {cr:.2%} | Sharpe: {sr:.2f} | MaxDD: {mdd:.2%} | PF: {pf:.2f}")
            print(f"    AvgHold: {avg_hold:.0f}d | AvgWin: {avg_w:.2%} | AvgLoss: {avg_l:.2%}")

        df_val['window'] = wname
        all_equity.append(df_val)
        all_trades.extend(window_trades)

    # ── Full period ──
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
        print(f"    Total trades: {len(df_t)}")
        print(f"    Win rate: {wr:.1%}")
        print(f"    Avg net return: {avg_nr:.4%}")
        print(f"    Avg win: {avg_w:.4%} | Avg loss: {avg_l:.4%}")
        print(f"    Profit factor: {pf:.2f}")

        # Exit reason breakdown
        print(f"\n    Exit reasons:")
        for r in ['matured', 'stop_loss']:
            sub = df_t[df_t['exit_reason'] == r]
            if len(sub) > 0:
                print(f"      {r}: {len(sub)} trades, WR {(sub['net_return']>0).mean():.1%}, avg {sub['net_return'].mean():.4%}")

        # Regime breakdown
        print(f"\n    By regime:")
        for r in sorted(df_t['regime'].unique()):
            sub = df_t[df_t['regime'] == r]
            if len(sub) > 0:
                print(f"      {r}: {len(sub)} trades, WR {(sub['net_return']>0).mean():.1%}, avg {sub['net_return'].mean():.4%}")

    if len(df_e) > 0:
        eq = df_e.copy()
        eq['equity'] = eq['value'] / eq['value'].iloc[0]
        eq['ret_daily'] = eq['value'].pct_change().fillna(0)
        full_cr = eq['value'].iloc[-1] / eq['value'].iloc[0] - 1
        full_sr = eq['ret_daily'].mean() / eq['ret_daily'].std() * np.sqrt(240) if eq['ret_daily'].std() > 0 else 0
        eq['cummax'] = eq['value'].cummax()
        full_mdd = (eq['value'] / eq['cummax'] - 1).min()

        print(f"\n    Full equity curve:")
        print(f"      Cumulative return: {full_cr:.2%}")
        print(f"      Sharpe: {full_sr:.2f}")
        print(f"      Max DD: {full_mdd:.2%}")

    return all_trades, all_equity

# ════════════════════════════════════════════════════════════════════
# RUN BOTH MODES
# ════════════════════════════════════════════════════════════════════
results = {}
for mode_name, mode_key in [('A) 5d_10% Single Model + Risk Controls', '5d_only'),
                             ('B) Dual Model + Risk Controls', 'dual')]:
    strat = build_strat(mode_key)
    trades, equity = run_backtest(mode_name, strat)
    results[mode_key] = {'trades': trades, 'equity': equity}

# ── Save ──
print(f"\n[Saving results...]")
os.makedirs(os.path.join(ROOT, 'quant_archive', '2026-05'), exist_ok=True)
for key in ['5d_only', 'dual']:
    r = results[key]
    df_t = pd.DataFrame(r['trades'])
    df_e = pd.concat(r['equity'], ignore_index=True).sort_values('date')
    if len(df_t) > 0:
        df_t.to_csv(os.path.join(ROOT, 'quant_archive', '2026-05', f'backtest_v2r2_{key}_trades.csv'), index=False)
    if len(df_e) > 0:
        df_e.to_csv(os.path.join(ROOT, 'quant_archive', '2026-05', f'backtest_v2r2_{key}_equity.csv'), index=False)

print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
print("Done!")
