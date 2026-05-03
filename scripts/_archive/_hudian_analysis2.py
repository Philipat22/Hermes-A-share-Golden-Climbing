# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())
os.environ["PYTHONIOENCODING"] = "utf-8"

import tushare as ts
pro = ts.pro_api('5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7')

# 重点关注：沪电股份 + 同板块核心标的
stocks = [
    ('沪电股份', '002463.SZ'),
    ('立讯精密', '002475.SZ'),
    ('京东方A', '000725.SZ'),
    ('东山精密', '002384.SZ'),
    ('歌尔股份', '002241.SZ'),
    ('信维通信', '300136.SZ'),
    ('TCL科技', '000100.SZ'),
    ('兴森科技', '002436.SZ'),
    ('深科技', '000021.SZ'),
    ('风华高科', '000636.SZ'),
]

print(f"{'名称':8s} {'现价':>7s} {'5日涨跌':>8s} {'20日涨跌':>8s} {'PE_TTM':>8s} {'PB':>6s} {'总市值亿':>9s} {'近5日日均成交额':>14s}")
print("-" * 90)

for name, code in stocks:
    try:
        # 日线数据
        df = pro.daily(ts_code=code, start_date='20260330', end_date='20260429')
        if len(df) < 2:
            print(f"  {name:8s}: 数据不足")
            continue
        
        closes = df.sort_values('trade_date')['close']
        chg_5d = (closes.iloc[-1] / closes.iloc[-5] - 1) * 100 if len(closes) >= 5 else 0
        chg_20d = (closes.iloc[-1] / closes.iloc[-20] - 1) * 100 if len(closes) >= 20 else 0
        latest_close = closes.iloc[-1]
        vol_avg = df['vol'].mean()
        turnover = vol_avg / latest_close / 1e4 * 100  # 简算
        
        # PE/PB from daily_basic
        db = pro.daily_basic(ts_code=code, trade_date='20260429')
        if len(db) == 0:
            db = pro.daily_basic(ts_code=code)  # 最新
        if len(db) > 0:
            pe = db.iloc[0].get('pe_ttm')
            pb = db.iloc[0].get('pb')
            mktcap = db.iloc[0].get('total_market_cap')
        else:
            pe = pb = mktcap = None
        
        pe_str = f"{pe:.1f}" if pe and pe > 0 else "N/A"
        pb_str = f"{pb:.2f}" if pb and pb > 0 else "N/A"
        mkt_str = f"{mktcap:.0f}" if mktcap else "N/A"
        chg5_str = f"{chg_5d:+.1f}%"
        chg20_str = f"{chg_20d:+.1f}%"
        
        print(f"  {name:8s} {latest_close:>7.2f} {chg5_str:>8s} {chg20_str:>8s} {pe_str:>8s} {pb_str:>6s} {mkt_str:>9s} {vol_avg/10000:>14.0f}万")
    except Exception as e:
        print(f"  {name:8s}: {e}")

print("\n=== 沪电股份近期逐日明细 ===")
code = '002463.SZ'
df = pro.daily(ts_code=code, start_date='20260401', end_date='20260429')
for _, row in df.sort_values('trade_date').iterrows():
    print(f"  {row['trade_date']}: 收={row['close']:.2f} 涨跌={row['pct_chg']:+.2f}% 量={row['vol']/10000:.0f}万")
