# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import tushare as ts
pro = ts.pro_api('5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7')

print("=== 今日涨跌停概览 ===")
try:
    df = pro.limit_list_d(trade_date='20260429')
    print(f"今日涨停股数量: {len(df)}")
    print(f"今日跌停股数量: {len(df[df['pct_chg'] < 0])}")
    if len(df) > 0:
        up = df[df['pct_chg'] > 9.5]
        down = df[df['pct_chg'] < -9.5]
        print(f"\n涨停股按行业分布:")
        ind_s = up.groupby('industry').size().sort_values(ascending=False)
        for k, v in ind_s.items():
            print(f"  {k}: {v}只")
except Exception as e:
    print(f"涨跌停数据: {e}")

print("\n=== 近5日各行业指数涨跌 ===")
# 用申万一级行业指数
sw_codes = [
    ('半导体', '801083.SH'),
    ('计算机', '801750.SH'),
    ('通信', '801770.SH'),
    ('传媒', '801760.SH'),
    ('军工', '801750.SI'),  # 这个可能不对，查一下
    ('医药', '801150.SI'),
    ('银行', '801780.SI'),
    ('电子', '801080.SI'),
    ('电力设备', '801730.SI'),
    ('汽车', '801880.SI'),
]

for name, code in sw_codes:
    try:
        df = pro.index_daily(ts_code=code, start_date='20260422', end_date='20260429')
        if len(df) >= 2:
            first = df.iloc[-1]['close']
            last = df.iloc[0]['close']
            chg = (last - first) / first * 100
            bars = '█' * int(max(0, chg / 2))
            print(f"  {name:8s}: {first:.1f} -> {last:.1f} ({chg:+.1f}%) {bars}")
    except Exception as e:
        print(f"  {name:8s}: 数据不可用 ({e})")

print("\n=== 近期市场热点题材（根据涨跌停和板块动量）===")
# 通过成交量异动找热门板块
try:
    # 沪深主要指数近期表现
    big_idx = [
        ('科创50', '000688.SH'),
        ('上证指数', '000001.SH'),
        ('深证成指', '399001.SZ'),
        ('创业板', '399006.SZ'),
        ('沪深300', '000300.SH'),
    ]
    print("\n大盘指数近5日:")
    for name, code in big_idx:
        df = pro.index_daily(ts_code=code, start_date='20260422', end_date='20260429')
        if len(df) >= 2:
            first = df.iloc[-1]['close']
            last = df.iloc[0]['close']
            chg = (last - first) / first * 100
            print(f"  {name}: {first:.2f} -> {last:.2f} ({chg:+.2f}%)")
except Exception as e:
    print(f"大盘指数报错: {e}")

print("\n=== 成交额异动（今日 vs 上周均值）===")
# 找今日成交额最大/异动的股票
try:
    df = pro.daily(trade_date='20260429')
    df = df.sort_values('vol', ascending=False).head(20)
    print("今日成交额最大的20只股票:")
    for _, row in df.iterrows():
        print(f"  {row['ts_code']}: {row['vol']/10000:.0f}万元")
except Exception as e:
    print(f"成交额数据报错: {e}")