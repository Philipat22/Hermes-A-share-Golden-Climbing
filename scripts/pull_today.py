"""拉取今天(2026-05-11)全A股日线, 增量追加到prices_full.pkl"""
import tushare as ts
import pickle, pandas as pd, time, os
from datetime import datetime

TOKEN = "6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7".strip()
CACHE = r"D:\AIHedgeFund\ai-hedge-fund-main\data\cache"
pro = ts.pro_api(TOKEN)

TODAY = "20260511"
prices_path = os.path.join(CACHE, "prices_full.pkl")

print(f"加载现有数据...")
prices = pickle.load(open(prices_path, "rb"))
last_date = None
for code, df in prices.items():
    last = str(df.sort_values("trade_date")["trade_date"].iloc[-1])[:8]
    if last > (last_date or ""):
        last_date = last
print(f"  现有 {len(prices)} 只股票, 最新日期: {last_date}")

if last_date >= TODAY:
    print(f"数据已是最新({last_date}), 无需拉取")
    exit(0)

# 拉取全A股列表
print(f"拉取股票列表...")
stocks = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name")
codes = stocks["ts_code"].tolist()
print(f"  {len(codes)} 只上市股票")

# 拉取2026-05-08到今天的数据
start = "20260508"
print(f"拉取 {start} ~ {TODAY} 日线...")
# 分批拉取
batch_size = 500
new_count = 0
for i in range(0, len(codes), batch_size):
    batch = codes[i:i+batch_size]
    ts_codes = ",".join(batch)
    try:
        df = pro.daily(ts_code=ts_codes, start_date=start, end_date=TODAY)
        if df is None or len(df) == 0:
            continue
        for code in batch:
            sub = df[df["ts_code"] == code]
            if len(sub) == 0:
                continue
            sub = sub.sort_values("trade_date")
            if code not in prices:
                prices[code] = sub[["trade_date", "open", "high", "low", "close", "vol", "amount"]]
            else:
                old = prices[code]
                # 去重追加
                old_dates = set(old["trade_date"].astype(str).str[:8].values)
                new_rows = sub[~sub["trade_date"].astype(str).str[:8].isin(old_dates)]
                if len(new_rows) > 0:
                    prices[code] = pd.concat([old, new_rows], ignore_index=True).sort_values("trade_date")
                    new_count += 1
        if (i // batch_size) % 5 == 0:
            print(f"  {i}/{len(codes)}...")
        time.sleep(0.15)
    except Exception as e:
        print(f"  batch {i} 出错: {e}")
        time.sleep(1)

print(f"保存... ({new_count} 只股票有新增数据)")
pickle.dump(prices, open(prices_path, "wb"))
print("完成!")
