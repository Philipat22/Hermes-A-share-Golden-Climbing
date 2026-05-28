"""牛回头v2.0 持仓模拟 — 全量逐日 + N=3仓位 + 滚动复利"""
import pickle, pandas as pd, numpy as np, time

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'
print("Loading data...", flush=True)

with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)

sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = dict(zip(sw['ts_code'], sw['industry']))

MFG_17 = {'半导体', '纺织', '纺织机械', '专用机械', '轻工机械', '铝', '电器仪表', '矿物制品', '特种钢', '铜', '钢加工', '机床制造', '机械基件', '工程机械', '电气设备', '汽车配件', '通信设备'}

DD_LO, DD_HI = -28, -24
R120_LO, R120_HI = 20, 45
MAX_HOLD = 90
MAX_POSITIONS = 3

# Build a daily index: for each trading day, which stocks have what prices
print("Building daily price matrix...", flush=True)
# Use CSI300 dates as canonical calendar
csi = pd.read_pickle(BASE + r'\data\cache\csi300.pkl')
csi['trade_date'] = pd.to_datetime(csi['trade_date'], errors='coerce')
csi = csi.sort_values('trade_date').reset_index(drop=True)

# Filter to 2008+
csi = csi[csi['trade_date'] >= '2008-01-01'].reset_index(drop=True)
print(f"Trading days: {len(csi)} from {csi['trade_date'].iloc[0].date()} to {csi['trade_date'].iloc[-1].date()}", flush=True)

# Build dict: date_str -> {code: {close, ma60, high120_idx, r120, dd}}
# This is expensive but needed for daily simulation
t0 = time.time()
daily_data = {}  # date_str -> {code: price_info}
codes_processed = []

mfg_codes = [c for c in prices_dict if not c.startswith('688') and c[:3] not in ('300','301') and sector_map.get(c,'') in MFG_17]
print(f"Processing {len(mfg_codes)} stocks...", flush=True)

for i, code in enumerate(mfg_codes):
    if i % 100 == 0:
        print(f"  {i}/{len(mfg_codes)} ({time.time()-t0:.0f}s)", flush=True)
    
    df = prices_dict[code].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    if len(df) < 390: continue
    
    c = df['close'].values.astype(float)
    n = len(c)
    sec = sector_map.get(code, '')
    
    # Pre-compute
    ma60_arr = np.full(n, np.nan)
    high120_idx = np.full(n, -1, dtype=int)
    for e in range(249, n):
        ma60_arr[e] = np.mean(c[e-59:e+1])
        lb = min(120, e)
        high120_idx[e] = e - lb + np.argmax(c[e-lb:e+1])
    
    for e in range(249, n):
        date_str = str(df['trade_date'].iloc[e].date())
        if date_str not in daily_data:
            daily_data[date_str] = {}
        
        pi = high120_idx[e]
        dd = (c[e]/c[pi]-1)*100 if pi >= 0 else 0
        r120 = (c[e]/c[e-120]-1)*100
        
        # Store enough info for signal check
        daily_data[date_str][code] = {
            'close': c[e],
            'ma60': ma60_arr[e],
            'dd': dd,
            'r120': r120,
            'speed': abs(dd)/max(e-pi,1) if pi >= 0 and e>pi else 0,
            'sec': sec,
            'index': e,  # position in the original array for forward lookup
        }

print(f"Built daily_data: {len(daily_data)} days in {time.time()-t0:.0f}s", flush=True)

# Store forward close prices for exit checking
# For each code, store the full close array for forward lookups
print("Building forward price lookup...", flush=True)
forward_prices = {}
for code in mfg_codes:
    df = prices_dict[code]
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    forward_prices[code] = df['close'].values.astype(float)

# Also build MA60 forward array for exit checking
forward_ma60 = {}
for code in mfg_codes:
    c = forward_prices[code]
    n = len(c)
    ma60_arr = np.full(n, np.nan)
    for e in range(249, n):
        ma60_arr[e] = np.mean(c[e-59:e+1])
    forward_ma60[code] = ma60_arr

# --- SIMULATION ---
print("\n=== Running Simulation ===", flush=True)
capital = 100000.0  # Start with 100k
positions = []  # [{code, entry_idx, entry_price, entry_date, sec}]
daily_equity = []
trade_log = []

dates_sorted = sorted(daily_data.keys())
trade_count = 0

for di, date_str in enumerate(dates_sorted):
    day_data = daily_data[date_str]
    
    # Step 1: Check existing positions for exit
    new_positions = []
    for pos in positions:
        code = pos['code']
        if code not in forward_prices:
            new_positions.append(pos)
            continue
        c_arr = forward_prices[code]
        ma_arr = forward_ma60[code]
        idx = pos['entry_idx']
        days_held = di - dates_sorted.index(pos['entry_date'])  # approximate
        
        # Find actual forward index
        # We need to find where we are in the forward array
        # Simple approach: find the date in the stock's data
        df_code = prices_dict[code]
        df_code['trade_date'] = pd.to_datetime(df_code['trade_date'], errors='coerce')
        df_code = df_code.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
        
        # Find current index
        curr_idx = None
        for j in range(pos['entry_idx'], len(df_code)):
            if str(df_code['trade_date'].iloc[j].date()) == date_str:
                curr_idx = j
                break
        
        if curr_idx is None:
            new_positions.append(pos)
            continue
        
        # Check exit
        c_curr = c_arr[curr_idx]
        ma_curr = ma_arr[curr_idx] if curr_idx < len(ma_arr) else np.nan
        days_held = curr_idx - pos['entry_idx']
        
        should_exit = False
        exit_reason = ""
        
        # MA60 recovery
        if not np.isnan(ma_curr) and c_curr > ma_curr:
            should_exit = True
            exit_reason = "MA60"
        # 90-day cap
        elif days_held >= MAX_HOLD:
            should_exit = True
            exit_reason = "90d cap"
        
        if should_exit:
            pnl = (c_curr / pos['entry_price'] - 1)
            capital += pos['allocated'] * (1 + pnl)
            trade_count += 1
            trade_log.append({
                'code': code, 'entry_date': pos['entry_date'],
                'exit_date': date_str, 'days': days_held,
                'pnl': pnl*100, 'reason': exit_reason,
                'allocated': pos['allocated']
            })
        else:
            new_positions.append(pos)
    
    positions = new_positions
    
    # Step 2: Check for new entries
    slots = MAX_POSITIONS - len(positions)
    if slots <= 0:
        daily_equity.append({'date': date_str, 'equity': capital})
        continue
    
    # Scan for signals on this day
    candidates = []
    for code, info in day_data.items():
        if info['close'] >= info['ma60']: continue
        if info['dd'] > DD_HI or info['dd'] < DD_LO: continue
        if info['speed'] < 0.5: continue
        if info['r120'] < R120_LO or info['r120'] > R120_HI: continue
        # Don't enter same sector twice
        current_secs = {p['sec'] for p in positions}
        if info['sec'] in current_secs: continue
        candidates.append((code, info))
    
    # Sort by DD (deeper first = better)
    candidates.sort(key=lambda x: x[1]['dd'])
    
    # Enter positions
    for code, info in candidates[:slots]:
        alloc = capital / MAX_POSITIONS
        capital -= alloc
        positions.append({
            'code': code, 'entry_idx': info['index'],
            'entry_price': info['close'], 'entry_date': date_str,
            'sec': info['sec'], 'allocated': alloc
        })
    
    daily_equity.append({'date': date_str, 'equity': capital})

# Close remaining positions at end
for pos in positions:
    # Use last known close
    c_arr = forward_prices.get(pos['code'])
    if c_arr is not None:
        last_close = c_arr[-1]
        pnl = (last_close / pos['entry_price'] - 1)
        capital += pos['allocated'] * (1 + pnl)
        trade_log.append({
            'code': pos['code'], 'entry_date': pos['entry_date'],
            'exit_date': '2026-05-28', 'days': len(c_arr)-1-pos['entry_idx'],
            'pnl': pnl*100, 'reason': 'EOD',
            'allocated': pos['allocated']
        })

print(f"\nFinal capital: ¥{capital:,.0f}  Trades: {trade_count}", flush=True)

# Annual breakdown
df_log = pd.DataFrame(trade_log)
df_eq = pd.DataFrame(daily_equity)
df_eq['year'] = pd.to_datetime(df_eq['date']).dt.year

print(f"\n=== Annual Returns ===")
print(f"{'Year':>6s} {'Trades':>6s} {'Avg PnL':>8s} {'WR':>5s} {'End Eq':>10s} {'Return':>8s}")
print("-" * 50)

prev_eq = 100000.0
for yr in sorted(df_eq['year'].unique()):
    ye = df_eq[df_eq['year'] == yr]
    if len(ye) == 0: continue
    end_eq = ye['equity'].iloc[-1]
    yr_ret = (end_eq / prev_eq - 1) * 100
    
    yt = df_log[pd.to_datetime(df_log['exit_date']).dt.year == yr] if len(df_log) > 0 else pd.DataFrame()
    n_tr = len(yt)
    avg_pnl = yt['pnl'].mean() if n_tr > 0 else 0
    wr = (yt['pnl'] > 0).sum() / n_tr * 100 if n_tr > 0 else 0
    
    print(f" {yr:>4d}  {n_tr:>6d}  {avg_pnl:>+7.2f}%  {wr:>4.0f}%  ¥{end_eq:>9,.0f}  {yr_ret:>+7.1f}%")
    prev_eq = end_eq

total_ret = (capital / 100000 - 1) * 100
years = sorted(df_eq['year'].unique())
cagr = ((capital/100000)**(1/len(years))-1)*100 if len(years) > 0 else 0
print(f"\nTotal: ¥{capital:,.0f} (+{total_ret:.0f}%)  CAGR: {cagr:.1f}%  Trades: {trade_count}")

# Trade stats
if len(df_log) > 0:
    print(f"\nTrade stats:")
    print(f"  Win rate: {(df_log['pnl']>0).sum()/len(df_log)*100:.0f}%")
    print(f"  Avg win: {df_log[df_log['pnl']>0]['pnl'].mean():+.2f}%")
    print(f"  Avg loss: {df_log[df_log['pnl']<0]['pnl'].mean():+.2f}%")
    print(f"  Median PnL: {df_log['pnl'].median():+.2f}%")
    print(f"  Median days: {df_log['days'].median():.0f}d")
    # Exit reasons
    for reason in df_log['reason'].unique():
        rdf = df_log[df_log['reason'] == reason]
        print(f"  {reason}: n={len(rdf)} avg={rdf['pnl'].mean():+.2f}%")
