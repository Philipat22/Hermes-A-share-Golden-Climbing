# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.getcwd())
os.environ["PYTHONIOENCODING"] = "utf-8"

import tushare as ts
pro = ts.pro_api('5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7')

print("=== 今日涨停统计 ===")
try:
    df = pro.limit_list_d(trade_date='20260429')
    print(f"今日涨停股数量: {len(df)}")
    sectors = df.groupby('industry').size().sort_values(ascending=False).head(15)
    print("\n涨停股行业分布 TOP15:")
    for ind, cnt in sectors.items():
        print(f"  {ind}: {cnt}只")
except Exception as e:
    print(f"涨停数据报错: {e}")

print("\n=== 近期热门板块涨跌 ===")
# 获取主要指数近期表现
indices = [
    ('科创50', '000688.SH'),
    ('创业板指', '399006.SZ'),
    ('中证AI', '931070.CSI'),
    ('中证半导体', '931051.CSI'),
    ('中证机器人', '931025.CSI'),
    ('中证医疗', '399282.SZ'),
    ('中证军工', '399967.SZ'),
    ('中证新能源', '399808.SZ'),
]
try:
    for name, code in indices:
        df = pro.index_daily(ts_code=code, start_date='20260421', end_date='20260429')
        if len(df) >= 2:
            first = df.iloc[-1]['close']
            last = df.iloc[0]['close']
            chg = (last - first) / first * 100
            print(f"  {name}: {first:.2f} → {last:.2f} ({chg:+.2f}%)")
except Exception as e:
    print(f"指数数据报错: {e}")

print("\n=== 概念板块行情 ===")
try:
    # 申万行业涨跌榜
    sw = pro.index_daily(ts_code='852011.SH', start_date='20260425', end_date='20260429')  # 申万综合
    print(f"申万行业数量: {len(sw)}")
except:
    pass

# 直接用行业指数
try:
    ind_df = pro.index_daily(start_date='20260425', end_date='20260429')
    print("指数数量:", len(ind_df))
    print(ind_df.head(5))
except Exception as e:
    print(f"获取报错: {e}")