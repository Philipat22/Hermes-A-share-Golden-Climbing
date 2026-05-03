# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

import tushare as ts
pro = ts.pro_api('5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7')

print("=== 北向资金（判断外资偏好）===")
try:
    north = pro.moneyflow_hsgt()
    if len(north) > 0:
        north = north.sort_values('trade_date')
        recent = north.tail(10)
        print("近10日北向资金:")
        for _, r in recent.iterrows():
            print(f"  {r['trade_date']}: 净买入{r['hgt_close']}亿 (沪股通{r.get('sht_close','-')}, 深股通{r.get('szt_close','-')})")
        total = north.tail(10)['hgt_close'].sum()
        print(f"近10日合计北向: {total:.1f}亿")
except Exception as e:
    print(f"北向资金报错: {e}")

print("\n=== 融资融券（判断杠杆资金偏好）===")
try:
    # 取沪深300成分股的融资余额变化
    margin = pro.margin_detail(trade_date='20260429')
    print(f"今日融资融券数据条数: {len(margin)}")
    if len(margin) == 0:
        # 试一下没有日期的
        margin = pro.margin_detail()
        print(f"最近交易日数据条数: {len(margin)}")
        if len(margin) > 0:
            print(margin.head(3))
except Exception as e:
    print(f"融资融券报错: {e}")

print("\n=== 行业资金流向（板块资金轮动）===")
try:
    # 用申万行业指数涨跌来近似
    # 先找几只热门龙头股的表现
    hot_stocks = [
        ('北方华创', '002371.SZ'),
        ('韦尔股份', '603501.SH'),
        ('中芯国际', '688981.SH'),
        ('宁德时代', '300750.SZ'),
        ('比亚迪', '002594.SZ'),
        ('东方财富', '300059.SZ'),
        ('浪潮信息', '000977.SZ'),
        ('科大讯飞', '002230.SZ'),
        ('工业富联', '601138.SH'),
        ('中国软件', '600536.SH'),
    ]
    print("热门龙头股近5日表现:")
    for name, code in hot_stocks:
        df = pro.daily(ts_code=code, start_date='20260422', end_date='20260429')
        if len(df) >= 2:
            first = df.iloc[-1]['close']
            last = df.iloc[0]['close']
            chg = (last - first) / first * 100
            bar = '▲' if chg > 0 else '▼'
            print(f"  {name:8s}({code}): {first:.1f} -> {last:.1f} {bar}{chg:.1f}%")
        else:
            print(f"  {name:8s}: 数据不足")
except Exception as e:
    print(f"龙头股数据报错: {e}")

print("\n=== 换手率异动（活跃资金）===")
try:
    df = pro.daily(trade_date='20260429')
    # 换手率 = vol / 流动股本 (这里用vol/总股本近似)
    df['turnover'] = df['vol'] / 1e4  # 简化
    top_turn = df.sort_values('vol', ascending=False).head(15)
    print("成交额最大的15只股票:")
    for _, r in top_turn.iterrows():
        print(f"  {r['ts_code']}: {r['vol']/10000:.0f}万元")
except Exception as e:
    print(f"换手率数据报错: {e}")