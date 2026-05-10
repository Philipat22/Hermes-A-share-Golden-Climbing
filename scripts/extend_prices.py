"""
扩展数据时间周期: 2014-01-02 → 2018-12-31
合并到现有 prices_full.pkl
用法: python extend_prices.py [--token YOUR_TOKEN]
"""
import pickle, pandas as pd, numpy as np
import time, os, sys, argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE = 'data/cache'
MAX_WORKERS = 4  # 积分15000+, 可以开高并发

def get_trading_days(start='20140102', end='20181231'):
    """获取交易日列表"""
    import tushare as ts
    pro = ts.pro_api()
    df = pro.trade_cal(exchange='SSE', start_date=start, end_date=end)
    days = df[df['is_open']==1]['cal_date'].tolist()
    return sorted(days)

def fetch_one_day(date_str, pro):
    """拉取一天的全市场日线"""
    try:
        df = pro.daily(trade_date=date_str)
        if df is not None and len(df) > 0:
            return date_str, df
        return date_str, None
    except Exception as e:
        print(f"  ⚠ {date_str}: {e}")
        time.sleep(1)
        return date_str, None

def main(token=None):
    import tushare as ts
    if token:
        ts.set_token(token)
    pro = ts.pro_api()
    
    # 1. 加载现有数据
    print("[1/3] Loading existing prices_full.pkl...")
    existing_path = os.path.join(CACHE, 'prices_full.pkl')
    if os.path.exists(existing_path):
        existing = pickle.load(open(existing_path, 'rb'))
        existing_dates = set()
        for df in existing.values():
            for d in df['trade_date'].values:
                existing_dates.add(str(d))
        print(f"  Existing: {len(existing)} stocks, {len(existing_dates)} unique dates")
    else:
        existing = {}
        existing_dates = set()
        print(f"  No existing data, building from scratch")
    
    # 2. 获取需要补的交易日
    print("\n[2/3] Fetching trading days 2014-2018...")
    all_days = get_trading_days('20140102', '20181231')
    missing_days = [d for d in all_days if d not in existing_dates]
    print(f"  Total: {len(all_days)}, Missing: {len(missing_days)}")
    
    if not missing_days:
        print("  Already up to date!")
        return
    
    # 3. 并发拉取
    print(f"\n[3/3] Fetching {len(missing_days)} days ({MAX_WORKERS} workers)...")
    t0 = time.time()
    new_data = {}  # {date_str: DataFrame}
    done = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_one_day, d, pro): d for d in missing_days}
        for fut in as_completed(futs):
            date_str, df = fut.result()
            done += 1
            if df is not None:
                new_data[date_str] = df
            if done % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed/done * len(missing_days) - elapsed
                print(f"  {done}/{len(missing_days)} ({done/len(missing_days)*100:.0f}%) "
                      f"got {len(new_data)} days, {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
    
    print(f"  Done: {len(new_data)}/{len(missing_days)} days fetched in {time.time()-t0:.0f}s")
    
    # 4. 合并: 把新数据按股票归类, 追加到existing
    print("\n[4/4] Merging into prices_full.pkl...")
    # 新数据结构: {date: DataFrame with columns [ts_code, trade_date, open, high, low, close, vol, ...]}
    # 现有结构: {ts_code: DataFrame with columns [trade_date, open, high, low, close, vol, ts_code]}
    
    new_by_stock = {}
    for date_str, df in new_data.items():
        for _, row in df.iterrows():
            code = row['ts_code']
            if code not in new_by_stock:
                new_by_stock[code] = []
            new_by_stock[code].append({
                'trade_date': int(date_str),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'vol': float(row['vol']),
                'ts_code': code,
            })
    
    # 合并到existing
    added = 0
    for code, rows in new_by_stock.items():
        new_df = pd.DataFrame(rows).sort_values('trade_date')
        if code in existing:
            combined = pd.concat([new_df, existing[code]]).drop_duplicates('trade_date').sort_values('trade_date').reset_index(drop=True)
            existing[code] = combined
        else:
            existing[code] = new_df
        added += 1
    
    print(f"  Added {added} stocks, updated existing")
    
    # 5. 保存
    out_path = os.path.join(CACHE, 'prices_full_extended.pkl')
    pickle.dump(existing, open(out_path, 'wb'))
    size = os.path.getsize(out_path) / 1e6
    total_stocks = len(existing)
    print(f"\n✅ Saved to {out_path} ({size:.0f}MB, {total_stocks} stocks)")
    print(f"   To use: mv {out_path} {existing_path}")
    print(f"   Or: copy the extended file as prices_full.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, help='Tushare token')
    args = parser.parse_args()
    main(args.token)
