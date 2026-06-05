"""
Daily Data Update — North flow + Margin only.
(Price/CSI300 update: run scripts/update_prices_today.py separately)

Usage:
  1. D:\Python\python.exe scripts/update_prices_today.py   # prices + CSI300
  2. D:\Python\python.exe scripts/update_data.py            # north flow + margin

Or run both in sequence.
"""

import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

TOKEN = "6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"

sys.path.insert(0, str(ROOT))

today = datetime.now()
today_str = today.strftime("%Y%m%d")

import tushare as ts
pro = ts.pro_api(TOKEN)

cal = pro.trade_cal(exchange='SSE', start_date=today_str, end_date=today_str)
if len(cal) == 0 or cal.iloc[0]['is_open'] != 1:
    print(f"[SKIP] {today.date()} not a trading day.")
    sys.exit(0)

print(f"=== Macro Update — {today.date()} ===")

# ═══ North Flow ═══
print("\n[1/2] North flow...")
nf = pd.read_pickle(CACHE / "macro_north_flow.pkl")
nf['trade_date'] = pd.to_datetime(nf['trade_date'])
last = nf['trade_date'].max()
start = (last + timedelta(days=1)).strftime('%Y%m%d')

new = pro.moneyflow_hsgt(start_date=start, end_date=today_str)
if new is not None and len(new) > 0:
    new['trade_date'] = pd.to_datetime(new['trade_date'])
    nf = pd.concat([nf, new], ignore_index=True).drop_duplicates('trade_date', keep='last')
    nf.to_pickle(CACHE / "macro_north_flow.pkl")
    print(f"  +{len(new)} rows, total {len(nf)}")
else:
    print(f"  No new data (last: {last.date()})")

# ═══ Margin ═══
print("\n[2/2] Margin...")
mg = pd.read_pickle(CACHE / "macro_margin_daily.pkl")
mg['trade_date'] = pd.to_datetime(mg['trade_date'])
last = mg['trade_date'].max()
start = (last + timedelta(days=1)).strftime('%Y%m%d')

new = pro.margin(start_date=start, end_date=today_str)
if new is not None and len(new) > 0:
    new['trade_date'] = pd.to_datetime(new['trade_date'])
    daily = new.groupby('trade_date')['rzye'].sum().reset_index()
    daily.columns = ['trade_date', 'rzye']
    mg = pd.concat([mg, daily], ignore_index=True).drop_duplicates('trade_date', keep='last')
    mg.to_pickle(CACHE / "macro_margin_daily.pkl")
    print(f"  +{len(daily)} rows, total {len(mg)}")
else:
    print(f"  No new data (last: {last.date()})")

print("\n=== Done ===")
