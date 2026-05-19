"""
抓取今日A股日线数据 - 修复版 (处理混合日期类型)
"""
import pickle, pandas as pd, time, sys
from datetime import datetime

TOKEN = "6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7"
ROOT = r"D:\AIHedgeFund\ai-hedge-fund-main"
PRICES = ROOT + r"\data\cache\prices_full.pkl"
CSI300 = ROOT + r"\data\cache\csi300.pkl"

today = datetime.now()
today_str = today.strftime('%Y%m%d')
lookback = (today - pd.Timedelta(days=5)).strftime('%Y%m%d')
csi_lb = (today - pd.Timedelta(days=20)).strftime('%Y%m%d')

print("[1/4] Connect Tushare...")
import tushare as ts
pro = ts.pro_api(TOKEN)

cal = pro.trade_cal(exchange='SSE', start_date=today_str, end_date=today_str)
if len(cal) == 0 or cal.iloc[0]['is_open'] != 1:
    print("Not a trading day"); sys.exit(1)
print("Trading day [OK]")

print("\n[2/4] Load prices...")
with open(PRICES, 'rb') as f:
    prices = pickle.load(f)
all_codes = list(prices.keys())
print(f"{len(all_codes)} stocks")

# Fix mixed date types silently
print("[2.5/4] Normalize dates...")
for code in all_codes:
    try:
        df = prices[code]
        if hasattr(df, 'columns'):
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
            df = df.dropna(subset=['trade_date'])
            prices[code] = df
    except:
        pass

print("\n[3/4] Fetch today...")
new_count = 0; skip_count = 0; err_count = 0

for i, code in enumerate(all_codes):
    try:
        df = prices[code]
        if not hasattr(df, 'columns') or len(df) == 0:
            skip_count += 1; continue
        
        # Check if already has today
        max_dt = df['trade_date'].max()
        if hasattr(max_dt, 'strftime'):
            last_str = max_dt.strftime('%Y%m%d')
        else:
            last_str = str(max_dt)[:8]
        
        if last_str >= today_str:
            skip_count += 1; continue
        
        # Fetch
        df_new = pro.daily(ts_code=code, start_date=lookback, end_date=today_str)
        if df_new is None or len(df_new) == 0:
            continue
        
        df_new = df_new.rename(columns={
            'ts_code':'ts_code','trade_date':'trade_date',
            'open':'open','high':'high','low':'low','close':'close','vol':'vol',
            'pre_close':'pre_close','change':'change','pct_chg':'pct_chg'
        })
        df_new['amount'] = df_new.get('amount', 0)
        df_new['trade_date'] = pd.to_datetime(df_new['trade_date'])
        
        combined = pd.concat([df, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=['trade_date'], keep='last')
        combined = combined.sort_values('trade_date').reset_index(drop=True)
        prices[code] = combined
        new_count += 1
        
        if new_count % 500 == 0:
            print(f"  Progress: {new_count} updated, {skip_count} skipped")
        
        time.sleep(0.03)
    except Exception as e:
        err_count += 1
        if err_count <= 3:
            print(f"  Error #{err_count}: {code} - {e}")

print(f"  Done: {new_count} updated, {skip_count} skipped, {err_count} errors")

print("\n[4/4] Save...")
with open(PRICES, 'wb') as f:
    pickle.dump(prices, f)
print("prices_full.pkl saved")

# CSI300
print("Update CSI300...")
try:
    df_csi = pro.index_daily(ts_code='000300.SH', start_date=csi_lb, end_date=today_str)
    if df_csi is not None and len(df_csi) > 0:
        csi_old = pd.read_pickle(CSI300)
        df_csi['trade_date'] = pd.to_datetime(df_csi['trade_date'])
        csi_new = pd.concat([csi_old, df_csi], ignore_index=True)
        csi_new = csi_new.drop_duplicates(subset=['trade_date'], keep='last')
        csi_new = csi_new.sort_values('trade_date').reset_index(drop=True)
        csi_new.to_pickle(CSI300)
        print(f"CSI300: +{len(df_csi)} rows")
except Exception as e:
    print(f"CSI300 failed: {e}")

print("\n[DONE]")
