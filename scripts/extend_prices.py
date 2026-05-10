"""
扩展数据时间周期: 2014-01-02 → 2018-12-31
合并到现有 prices_full.pkl
用法: python extend_prices.py --token YOUR_TOKEN
"""
import pickle, pandas as pd, numpy as np
import time, os, sys, argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import tushare as ts

CACHE = 'data/cache'
MAX_WORKERS = 4  # 积分15000+, 可以开高并发


def get_trading_days_2014_2018():
    """生成2014-2018 A股交易日（跳过元旦/春节/清明/劳动节/端午/中秋/国庆）"""
    days = []
    year = 2014
    # 已知A股每年约245-252个交易日
    start_dt = datetime(2014, 1, 1)
    end_dt = datetime(2018, 12, 31)

    # 用pro.trade_cal，如果token不支持就跳过，备用逻辑兜底
    try:
        pro = ts.pro_api()
        df = pro.trade_cal(exchange='SSE', start_date='20140102', end_date='20181231')
        if df is not None and len(df) > 100:
            return sorted(df[df['is_open']==1]['cal_date'].tolist())
    except:
        pass

    # 兜底：用Python生成（排除中国法定假日规则）
    # 每年元旦1天 + 春节3天(不固定)+ 清明1天 + 劳动节1天 + 端午1天(不固定)+ 中秋1天(不固定)+ 国庆3天
    # 不精确但够用——配合missing_dates去重最终会补到最准
    from dateutil.relativedelta import WE, SA
    import dateutil

    dt = start_dt
    while dt <= end_dt:
        w = dt.weekday()
        # 排除周六(5)、周日(6)
        if w < 5:
            days.append(dt.strftime('%Y%m%d'))
        dt += timedelta(days=1)

    # 手动排除主要固定假期（补齐会有误差，拉数据时去重）
    fixed_holidays = [
        # 2014元旦
        '20140101','20140102','20140103',
        # 2014春节
        '20140128','20140129','20140130','20140131',
        # 2014清明
        '20140405','20140406','20140407',
        # 2014劳动节
        '20140501','20140502','20140503',
        # 2014端午
        '20140531',
        # 2014国庆
        '20141001','20141002','20141003','20141004','20141005','20141006','20141007',
        # 2015元旦
        '20150101','20150102','20150103',
        # 2015春节
        '20150218','20150219','20150220','20150221','20150223','20150224',
        # 2015清明
        '20150404','20150405','20150406',
        # 2015劳动节
        '20150501','20150502','20150503',
        # 2015端午
        '20150620','20150621','20150622',
        # 2015中秋
        '20150926','20150927',
        # 2015国庆
        '20151001','20151002','20151003','20151004','20151005','20151006','20151007',
        # 2016元旦
        '20160101','20160102','20160103',
        # 2016春节
        '20160207','20160208','20160209','20160210','20160211','20160212','20160213',
        # 2016清明
        '20160402','20160403','20160404',
        # 2016劳动节
        '20160430','20160501','20160502',
        # 2016端午
        '20160609','20160610','20160611',
        # 2016中秋
        '20160915','20160916','20160917',
        # 2016国庆
        '20161001','20161002','20161003','20161004','20161005','20161006','20161007',
        # 2017元旦
        '20170101','20170102',
        # 2017春节
        '20170127','20170128','20170129','20170130','20170131','20170201','20170202',
        # 2017清明
        '20170402','20170403','20170404',
        # 2017劳动节
        '20170429','20170430','20170501',
        # 2017端午
        '20170528','20170529','20170530',
        # 2017中秋+国庆
        '20171001','20171002','20171003','20171004','20171005','20171006','20171007','20171008',
        # 2018元旦
        '20180101',
        # 2018春节
        '20180215','20180216','20180217','20180218','20180219','20180220','20180221',
        # 2018清明
        '20180405','20180406','20180407',
        # 2018劳动节
        '20180429','20180430','20180501',
        # 2018端午
        '20180616','20180617','20180618',
        # 2018中秋
        '20180922','20180923','20180924',
        # 2018国庆
        '20181001','20181002','20181003','20181004','20181005','20181006','20181007',
    ]
    days = [d for d in days if d not in fixed_holidays]
    return sorted(days)


def fetch_one_day(date_str, token):
    """拉取一天的全市场日线"""
    try:
        pro = ts.pro_api(token)
        df = pro.daily(trade_date=date_str)
        if df is not None and len(df) > 0:
            return date_str, df
        return date_str, None
    except Exception as e:
        print(f"  ⚠ {date_str}: {e}")
        time.sleep(1)
        return date_str, None


def main(token=None):
    if not token:
        raise ValueError('Tushare token required')
    # ⚠️ 必须直接传token给pro_api()，ts.set_token()+pro_api()组合有bug
    pro = ts.pro_api(token)

    # 1. 加载现有数据
    print("[1/3] Loading existing prices_full.pkl...")
    existing_path = os.path.join(CACHE, 'prices_full.pkl')
    if os.path.exists(existing_path):
        existing = pickle.load(open(existing_path, 'rb'))
        existing_dates = set()
        for df in existing.values():
            for d in df['trade_date'].values:
                existing_dates.add(str(int(d)))
        print(f"  Existing: {len(existing)} stocks, {len(existing_dates)} unique dates")
    else:
        existing = {}
        existing_dates = set()
        print(f"  No existing data, building from scratch")

    # 2. 获取需要补的交易日
    print("\n[2/3] Generating trading days 2014-2018...")
    all_days = get_trading_days_2014_2018()
    missing_days = [d for d in all_days if d not in existing_dates]
    print(f"  Total trading days: {len(all_days)}, Missing: {len(missing_days)}")

    if not missing_days:
        print("  Already up to date!")
        return

    # 3. 并发拉取
    print(f"\n[3/3] Fetching {len(missing_days)} days ({MAX_WORKERS} workers)...")
    t0 = time.time()
    new_data = {}  # {date_str: DataFrame}
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_one_day, d, token): d for d in missing_days}
        for fut in as_completed(futs):
            date_str, df = fut.result()
            done += 1
            if df is not None:
                new_data[date_str] = df
            else:
                errors += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(missing_days) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(missing_days)} ({done/len(missing_days)*100:.0f}%) "
                      f"got {len(new_data)} days, {errors} errors, "
                      f"{elapsed:.0f}s elapsed, ETA {eta/60:.0f}min")

    print(f"  Done: {len(new_data)}/{len(missing_days)} days fetched in {time.time()-t0:.0f}s ({errors} errors)")

    if not new_data:
        print("  No data fetched. Check your token and internet connection.")
        return

    # 4. 合并: 把新数据按股票归类, 追加到existing
    print("\n[4/4] Merging into prices_full.pkl...")
    new_by_stock = {}
    for date_str, df in new_data.items():
        for _, row in df.iterrows():
            code = row['ts_code']
            if code not in new_by_stock:
                new_by_stock[code] = []
            new_by_stock[code].append({
                'trade_date': str(date_str),  # 统一string格式
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'vol': float(row['vol']),
                'ts_code': code,
            })

    added = 0
    for code, rows in new_by_stock.items():
        new_df = pd.DataFrame(rows).sort_values('trade_date')
        if code in existing:
            combined = pd.concat([new_df, existing[code]]).drop_duplicates('trade_date').sort_values('trade_date').reset_index(drop=True)
            existing[code] = combined
        else:
            existing[code] = new_df
        added += 1

    print(f"  Added/updated {added} stocks")

    # 5. 保存
    out_path = os.path.join(CACHE, 'prices_full_extended.pkl')
    pickle.dump(existing, open(out_path, 'wb'))
    size = os.path.getsize(out_path) / 1e6
    total_stocks = len(existing)
    print(f"\n✅ Saved to {out_path} ({size:.0f}MB, {total_stocks} stocks)")
    print(f"   To activate: mv {out_path} {existing_path}")
    print(f"   Or: copy the extended file as prices_full.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, help='Tushare token')
    args = parser.parse_args()
    main(args.token)