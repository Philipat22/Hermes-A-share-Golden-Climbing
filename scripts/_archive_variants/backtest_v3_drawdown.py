"""
Backtest v3 — Drawdown Optimization Edition

Adds to v2:
  1. SIDEWAYS capacity reduced: 60% → 30%, max_pos 5 → 3
  2. Circuit breaker: 3 consecutive losses → 2-day trading halt
  3. Repeat-loser filter: don't re-buy stocks stopped out <20 trading days ago
  4. Confident-only mode in SIDEWAYS: only buy top-20% of scores

Everything else (costs, models, walk-forward windows) identical to v2.
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
STOP_LOSS  = -0.05

# ── Drawdown protection constants ────────────────────────────────
MAX_CONSECUTIVE_LOSSES = 3          # trigger after N consecutive losses
COOLDOWN_DAYS = 2                   # halt trading for N days
REPEAT_LOSER_BAN_DAYS = 20          # don't re-buy for N trading days
SIDEWAYS_SCORE_PCTILE = 0.20        # in sideways, only buy top 20% of scores
SIDEWAYS_CAPACITY = 0.30            # was 0.60
SIDEWAYS_MAX_POS = 3                # was 5

# ── Model configs ────────────────────────────────────────────────
MODEL_CFG = [
    {'name': '5d_10%',  'horizon': 5,  'thresh': 0.10},
    {'name': '10d_15%', 'horizon': 10, 'thresh': 0.15},
]
WF_WINDOWS = [
    ('2019-01-01', '2022-01-01', '2022-01-01', '2023-01-01', '2022 Bear'),
    ('2019-01-01', '2023-01-01', '2023-01-01', '2024-01-01', '2023 Sideways'),
    ('2019-01-01', '2024-01-01', '2024-01-01', '2025-07-01', '2024-2025'),
]

# ── Strategy configs ─────────────────────────────────────────────
def build_strat(mode='5d_only'):
    """Regime → (model_name, threshold, max_positions, capacity_pct)."""
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
print("BACKTEST v3 — Drawdown Optimization Edition")
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
# BACKTEST ENGINE (v3 with drawdown protection)
# ════════════════════════════════════════════════════════════════════
def run_backtest(mode_label, strategy):
    print(f"\n{'='*72}")
    print(f"  MODE: {mode_label}")
    print(f"  Protections: circuit_breaker({MAX_CONSECUTIVE_LOSSES}loss→{COOLDOWN_DAYS}d)"
          f" | repeat_loser_ban({REPEAT_LOSER_BAN_DAYS}d)"
          f" | sideways_top{(1-SIDEWAYS_SCORE_PCTILE)*100:.0f}%"
          f" | sideways_cap{SIDEWAYS_CAPACITY:.0%}")
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

        # ── TRADE SIMULATION (v3 state tracking) ──
        positions = {}; cash = 1_000_000
        window_trades = []; portfolio_values = []
        consecutive_losses = 0
        cooldown_remaining = 0
        # Track which stocks we've recently stopped out on (symbol → date)
        recent_losers = {}  # {sym: stop_loss_date}

        for di, d in enumerate(te_dates):
            day_mask = pdf['date'] == d
            regime = get_regime(d)
            strat = strategy.get(regime, strategy['sideways'])
            model_name, threshold, max_pos, capacity = strat
            is_sideways = (regime == 'sideways')

            # ── Circuit breaker: in cooldown → skip buying ──
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            # ── CHECK STOP-LOSSES ──
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
                    # Track stop-loss for repeat-loser filter
                    recent_losers[sym] = d
                    # Update consecutive loss counter
                    consecutive_losses += 1
                    del positions[sym]

            # ── Circuit breaker trigger ──
            if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                cooldown_remaining = COOLDOWN_DAYS
                consecutive_losses = 0

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
                # Reset consecutive loss on any matured win
                if net_ret > 0:
                    consecutive_losses = 0

            # ── BUY selection (skip if cooldown or no slots) ──
            if max_pos > 0 and cooldown_remaining == 0:
                cap_cash = cash * capacity
                open_slots = max(0, max_pos - len(positions))
                if open_slots > 0 and di < len(te_dates) - MODEL_CFG[0]['horizon']:
                    day_scores = pdf.loc[day_mask, f'score_{model_name}'].values
                    day_syms = pdf.loc[day_mask, 'vt_symbol'].values
                    next_date = te_dates[min(di + 1, len(te_dates) - 1)]

                    # Build candidates with repeat-loser filter
                    valid = []
                    for j, sym in enumerate(day_syms):
                        if sym in positions:
                            continue
                        # Repeat-loser filter: don't re-buy recently stopped-out stocks
                        if sym in recent_losers:
                            days_since = (d - recent_losers[sym]).days
                            if days_since < REPEAT_LOSER_BAN_DAYS:
                                continue
                        price = get_price(sym, next_date, 'open')
                        if price is None:
                            continue
                        valid.append((j, sym, day_scores[j], price))

                    if len(valid) > 0:
                        scores = np.array([v[2] for v in valid])
                        prices = [v[3] for v in valid]
                        syms = [v[1] for v in valid]

                        # Threshold filter
                        cand_mask = scores >= threshold
                        if cand_mask.sum() > 0:
                            cand_scores = scores[cand_mask]
                            cand_syms = [syms[k] for k in range(len(syms)) if cand_mask[k]]
                            cand_prices = [prices[k] for k in range(len(syms)) if cand_mask[k]]

                            # v3: In SIDEWAYS, only buy top 20% of scores
                            if is_sideways and len(cand_scores) > open_slots:
                                pct = np.percentile(cand_scores, (1 - SIDEWAYS_SCORE_PCTILE) * 100)
                                strict_mask = cand_scores >= pct
                                cand_scores = cand_scores[strict_mask]
                                cand_syms = [cand_syms[k] for k in range(len(cand_syms)) if strict_mask[k]]
                                cand_prices = [cand_prices[k] for k in range(len(cand_prices)) if strict_mask[k]]

                            # Sort by score, take top open_slots
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
            else:
                # In cooldown: cash does nothing but we still decrement
                pass

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
            avg_hold = (trades_df['exit_date'].apply(lambda x: pd.Timestamp(x)) -
                       trades_df['entry_date'].apply(lambda x: pd.Timestamp(x))).dt.days.mean()

            # Check if circuit breaker ever triggered
            stop_syms = trades_df[trades_df['exit_reason'] == 'stop_loss']['symbol'].nunique()
            skipped = len([v for v in portfolio_values if v['num_positions'] == 0 and v['cash'] > 999000])
            # Count unique repeat-loser blocks
            hot = len(recent_losers)

            print(f"\n  ── {wname} ──")
            print(f"    Trades: {n} | WR: {wr:.1%} | AvgNet: {avg_nr:.2%}")
            print(f"    CumRet: {cr:.2%} | Sharpe: {sr:.2f} | MaxDD: {mdd:.2%} | PF: {pf:.2f}")
            print(f"    AvgHold: {avg_hold:.0f}d | AvgWin: {avg_w:.2%} | AvgLoss: {avg_l:.2%}")
            print(f"    Protection stats: {hot} repeat-losers blocked | {skipped} cooldown days")

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
        print(f"    Profit factor: {pf:.2f}")

        print(f"\n    Exit reasons:")
        for r in ['matured', 'stop_loss']:
            sub = df_t[df_t['exit_reason'] == r]
            if len(sub) > 0:
                print(f"      {r}: {len(sub)} trades, WR {(sub['net_return']>0).mean():.1%}, avg {sub['net_return'].mean():.4%}")

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
for mode_name, mode_key in [('A) 5d_10% Single Model + Drawdown Protection', '5d_only'),
                             ('B) Dual Model + Drawdown Protection', 'dual')]:
    strat = build_strat(mode_key)
    trades, equity = run_backtest(mode_name, strat)
    results[mode_key] = {'trades': trades, 'equity': equity}

# ── Save ──
print(f"\n[Saving to quant_archive/2026-05/backtest_v3_*.csv ...]")
os.makedirs(os.path.join(ROOT, 'quant_archive', '2026-05'), exist_ok=True)
for key in ['5d_only', 'dual']:
    r = results[key]
    df_t = pd.DataFrame(r['trades'])
    df_e = pd.concat(r['equity'], ignore_index=True).sort_values('date')
    if len(df_t) > 0:
        df_t.to_csv(os.path.join(ROOT, 'quant_archive', '2026-05',
                                 f'backtest_v3_{key}_trades.csv'), index=False)
    if len(df_e) > 0:
        df_e.to_csv(os.path.join(ROOT, 'quant_archive', '2026-05',
                                 f'backtest_v3_{key}_equity.csv'), index=False)

# ── Compare summary ──
print(f"\n{'='*72}")
print("V3 vs V2 COMPARISON")
print(f"{'='*72}")
# Try to load V2 results
try:
    v2_equity = pd.read_csv(os.path.join(ROOT, 'quant_archive', '2026-05', 'backtest_v2_dual_equity.csv'))
    if len(v2_equity) > 0:
        v2_eq = v2_equity.copy()
        v2_eq['eq'] = v2_eq['value'] / v2_eq['value'].iloc[0]
        v2_eq['ret'] = v2_eq['value'].pct_change().fillna(0)
        v2_cr = v2_eq['value'].iloc[-1] / v2_eq['value'].iloc[0] - 1
        v2_sr = v2_eq['ret'].mean() / v2_eq['ret'].std() * np.sqrt(240) if v2_eq['ret'].std() > 0 else 0
        v2_eq['cmax'] = v2_eq['value'].cummax()
        v2_mdd = (v2_eq['value'] / v2_eq['cmax'] - 1).min()
        print(f"\n  V2-B (baseline):    {v2_cr:.2%} return, Sharpe {v2_sr:.2f}, maxDD {v2_mdd:.2%}")

    for key in ['5d_only', 'dual']:
        r = results[key]
        df_e = pd.concat(r['equity'], ignore_index=True).sort_values('date')
        if len(df_e) > 0:
            eq = df_e.copy()
            eq['eq'] = eq['value'] / eq['value'].iloc[0]
            eq['ret'] = eq['value'].pct_change().fillna(0)
            cr = eq['value'].iloc[-1] / eq['value'].iloc[0] - 1
            sr = eq['ret'].mean() / eq['ret'].std() * np.sqrt(240) if eq['ret'].std() > 0 else 0
            eq['cmax'] = eq['value'].cummax()
            mdd = (eq['value'] / eq['cmax'] - 1).min()
            tag = 'V3-A (single)' if key == '5d_only' else 'V3-B (dual)'
            delta_cr = (cr - v2_cr) if key == 'dual' else (cr - v2_cr)
            delta_dd = (mdd - v2_mdd) if key == 'dual' else (mdd - v2_mdd)
            print(f"  {tag}: {cr:.2%} return, Sharpe {sr:.2f}, maxDD {mdd:.2%}")
            print(f"    vs V2-B: return {delta_cr:+.2%}, maxDD {delta_dd:+.2%}")

        # V2-B trades for regime breakdown comparison
        v2_trades = pd.read_csv(os.path.join(ROOT, 'quant_archive', '2026-05', 'backtest_v2_dual_trades.csv'))
        v2_trades['net_return'] = v2_trades['net_return'].clip(-0.30, 0.30)
        v3_trades = pd.DataFrame(results['dual']['trades'])
        if len(v3_trades) > 0:
            v3_trades['net_return'] = v3_trades['net_return'].clip(-0.30, 0.30)
            print(f"\n  Regime comparison (V2-B → V3-B):")
            for regime in ['sideways', 'bear', 'severe_bear', 'bull', 'recovery']:
                old = v2_trades[v2_trades['regime'] == regime]
                new = v3_trades[v3_trades['regime'] == regime]
                if len(old) > 0 or len(new) > 0:
                    o_wr = (old['net_return'] > 0).mean() if len(old) > 0 else 0
                    n_wr = (new['net_return'] > 0).mean() if len(new) > 0 else 0
                    o_avg = old['net_return'].mean() if len(old) > 0 else 0
                    n_avg = new['net_return'].mean() if len(new) > 0 else 0
                    print(f"    {regime:12s}: V2 {len(old):>3}t WR {o_wr:.0%} avg{o_avg:.2%}"
                          f" → V3 {len(new):>3}t WR {n_wr:.0%} avg{n_avg:.2%}")

except Exception as e:
    print(f"  Comparison skipped: {e}")

print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
print("Done!")
