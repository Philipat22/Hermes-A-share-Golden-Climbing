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

print(f"{'名称':8s} {'现价':>7s} {'5日涨跌':>8s} {'20日涨跌':>8s} {'ROE_TTM':>8s} {'PE_TTM':>7s} {'PB':>5s} {'近1月换手%':>10s}")
print("-" * 80)

for name, code in stocks:
    try:
        # 价格数据
        df = pro.daily(ts_code=code, start_date='20260330', end_date='20260429')
        if len(df) < 2:
            print(f"  {name:8s}: 数据不足")
            continue
        
        # 计算各周期涨跌
        closes = df.sort_values('trade_date')['close']
        chg_5d = (closes.iloc[-1] / closes.iloc[-5] - 1) * 100 if len(closes) >= 5 else 0
        chg_20d = (closes.iloc[-1] / closes.iloc[-20] - 1) * 100 if len(closes) >= 20 else 0
        latest_close = closes.iloc[-1]
        
        # 基本面
        fi = pro.fina_indicator(ts_code=code, start_date='20260101', limit=1)
        roe = fi.iloc[0]['roe'] if len(fi) > 0 else None
        pe = fi.iloc[0]['pe_ttm'] if len(fi) > 0 else None
        pb = fi.iloc[0]['pb'] if len(fi) > 0 else None
        if roe and abs(roe) > 1:
            roe = roe / 100
        
        # 换手率（近1月日均）
        vol_avg = df['vol'].mean()
        # 换手率 = 日均成交额/流通市值估算
        turnover = vol_avg / latest_close / 1e4  # 简算
        
        roe_str = f"{roe*100:.1f}%" if roe else "N/A"
        pe_str = f"{pe:.1f}" if pe else "N/A"
        pb_str = f"{pb:.2f}" if pb else "N/A"
        chg5_str = f"{chg_5d:+.1f}%"
        chg20_str = f"{chg_20d:+.1f}%"
        
        bar = ""
        if chg_5d > 5: bar = "▲▲"
        elif chg_5d > 0: bar = "▲"
        elif chg_5d < -5: bar = "▼▼"
        elif chg_5d < 0: bar = "▼"
        
        print(f"  {name:8s} {latest_close:>7.2f} {chg5_str:>8s} {chg20_str:>8s} {roe_str:>8s} {pe_str:>7s} {pb_str:>5s} {turnover:>10.1f}")
    except Exception as e:
        print(f"  {name:8s}: {e}")

print("\n=== 沪电股份专项分析 ===")
# 沪电股份详细数据
code = '002463.SZ'
df = pro.daily(ts_code=code, start_date='20260301', end_date='20260429')
closes = df.sort_values('trade_date')['close']
print(f"近期价格：{closes.iloc[0]:.2f} -> {closes.iloc[-1]:.2f}")
for i in range(min(5, len(closes))):
    idx = -(i+1)
    if idx >= -len(closes):
        chg = (closes.iloc[idx] / closes.iloc[idx-1] - 1) * 100 if idx > -len(closes) else 0
        print(f"  {df.sort_values('trade_date').iloc[idx]['trade_date']}: {closes.iloc[idx]:.2f} ({chg:+.2f}%)")
