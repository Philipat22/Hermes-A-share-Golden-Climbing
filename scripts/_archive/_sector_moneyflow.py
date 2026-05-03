#!/usr/bin/env python3
"""板块资金流向分析 - 挖掘主线板块"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import tushare as ts
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

pro = ts.pro_api(os.getenv('TUSHARE_PRO_TOKEN'))

def to_num(x, default=0.0):
    """安全转数值（万→亿）"""
    try:
        return float(x) / 10000  # 万→亿
    except:
        return default

# ====== 1. 北向资金 ======
print('=' * 60)
print('一、北向资金(沪深港通)日净流入')
print('=' * 60)

df = pro.moneyflow_hsgt(start_date='20260401', end_date='20260429')
if df is not None and len(df) > 0:
    df = df.sort_values('trade_date', ascending=False)
    print('  %-10s %10s %10s %10s' % ('日期','沪股通(亿)','深股通(亿)','合计(亿)'))
    for _, r in df.head(8).iterrows():
        hgt = to_num(r.get('hgt', 0))
        sgt = to_num(r.get('sgt', 0))
        total = hgt + sgt
        tag = '↗流入' if total > 0 else '↘流出'
        print('  %-10s %+9.1f %+9.1f %+9.1f %s' % (r['trade_date'], hgt, sgt, total, tag))
    
    # 均值
    for label, n in [('5日均',5), ('10日均',min(10,len(df)))]:
        hgt_avg = df.head(n)['hgt'].apply(lambda x: to_num(x)).mean()
        sgt_avg = df.head(n)['sgt'].apply(lambda x: to_num(x)).mean()
        print('  %s: 沪%+.1f亿 深%+.1f亿 合计%+.1f亿' % (label, hgt_avg, sgt_avg, hgt_avg+sgt_avg))
else:
    print('无数据')

# ====== 2. 18板块量价异动 ======
print()
print('=' * 60)
print('二、18板块代表股 近3日 vs 近10日 量价扫描')
print('=' * 60)

top_picks = {
    '交通运输':'000089.SZ','军工':'002023.SZ','医药':'000028.SZ',
    '半导体':'002049.SZ','房地产':'000029.SZ','有色金属':'000426.SZ',
    '机械设备':'000519.SZ','环保':'000546.SZ','电力设备':'000576.SZ',
    '银行':'002142.SZ','食品饮料':'000596.SZ','黄金':'000506.SZ',
    '消费电子':'000050.SZ','新能源':'000400.SZ','计算机':'000034.SZ',
    '建筑材料':'000401.SZ','传媒':'000676.SZ','化工':'000420.SZ',
}

results = []
for sector, ticker in top_picks.items():
    try:
        df = pro.daily(ts_code=ticker, start_date='20260401', end_date='20260429')
        if df is None or len(df) == 0: continue
        df = df.sort_values('trade_date', ascending=False)
        close = float(df.iloc[0]['close'])
        
        p3 = (close / float(df.iloc[min(2, len(df)-1)]['close']) - 1) * 100 if len(df) >= 3 else 0
        p10 = (close / float(df.iloc[min(9, len(df)-1)]['close']) - 1) * 100 if len(df) >= 10 else None
        
        vol3 = float(df.head(3)['vol'].mean())
        vol10 = float(df.head(10)['vol'].mean()) if len(df) >= 10 else vol3
        vol_ratio = vol3 / vol10 if vol10 > 0 else 1.0
        
        amp = (float(df.head(3)['high'].max()) - float(df.head(3)['low'].min())) / float(df.head(3)['low'].min()) * 100
        
        results.append({'sector':sector,'ticker':ticker,'price':close,
                        'pct_3d':p3,'pct_10d':p10,'vol_ratio':vol_ratio,'amplitude':amp})
    except: pass

results.sort(key=lambda x: -x['pct_3d'])
print('  %-4s %-10s %-10s %6s %6s %8s %6s %s' % ('排行','板块','代表','价位','3日涨','10日涨','量比','活跃度'))
print('  ' + '-' * 66)
for i, r in enumerate(results):
    vtag = '↑' if r['vol_ratio'] > 1.2 else '↓' if r['vol_ratio'] < 0.8 else '─'
    p10s = '%+.1f%%' % r['pct_10d'] if r['pct_10d'] is not None else 'N/A'
    print('  %-4d %-10s %-10s %5.1f %+5.1f%% %6s %5.2fx%s %5.1f%%' % (
        i+1, r['sector'], r['ticker'], r['price'], r['pct_3d'], p10s, r['vol_ratio'], vtag, r['amplitude']))

# ====== 3. 综合评分 ======
print()
print('=' * 60)
print('三、综合评分 → 主线板块判断')
print('    评分逻辑: 3日动量30% + 量比30% + 10日趋势20% + 活跃度20%')
print('=' * 60)

scored = []
for r in results:
    s3 = max(-10, min(10, r['pct_3d'])) / 10 * 30
    sv = max(0, min(3, r['vol_ratio'])) / 3 * 30
    s10 = (max(-10, min(10, r['pct_10d'] or 0)) / 10 * 20) if r['pct_10d'] is not None else 0
    sa = max(0, min(15, r['amplitude'])) / 15 * 20
    total = s3 + sv + s10 + sa
    scored.append((total, r['sector'], r['pct_3d'], r['vol_ratio'], r['pct_10d']))

scored.sort(key=lambda x: -x[0])
print('  %6s %-10s %8s %6s %10s' % ('评分','板块','3日涨幅','量比','判断'))
print('  ' + '-' * 46)
for s in scored:
    if s[0] >= 65: act = '★★★ 主线'
    elif s[0] >= 55: act = '★★ 次级'
    elif s[0] >= 40: act = '★ 观察'
    else: act = '忽略'
    p10s = '%+.1f%%' % s[4] if s[4] is not None else 'N/A'
    print('  %5.1f  %-10s %+7.2f%% %5.2fx  %s' % (s[0], s[1], s[2], s[3], act))

# ====== 4. 结论 ======
print()
print('=' * 60)
print('四、回测 & 北向资金交叉验证')
print('=' * 60)

# top 3 by composite score
top3 = scored[:3]
tickers_top3 = []
for s in top3:
    for r in results:
        if r['sector'] == s[1]:
            tickers_top3.append(r['ticker'])
            break

# 查这3只近5日北向持仓变化（用个股资金流替代）
print('Top3板块近5日个股资金流:')
for i, (sc, sec, p3, vr, _) in enumerate(top3):
    t = None
    for r in results:
        if r['sector'] == sec:
            t = r['ticker']
            break
    if not t: continue
    try:
        mf = pro.moneyflow(ts_code=t, start_date='20260422', end_date='20260429')
        if mf is not None and len(mf) > 0:
            mf = mf.sort_values('trade_date', ascending=False)
            buy_sm = float(mf.head(5)['buy_sm_vol'].sum() or 0)
            sell_sm = float(mf.head(5)['sell_sm_vol'].sum() or 0)
            net = buy_sm - sell_sm
            print('  %s %s: 近5日大单净%+.0f万' % (sec, t, net/10000))
        else:
            print('  %s %s: 无资金流数据' % (sec, t))
    except Exception as e:
        print('  %s %s: %s' % (sec, t, str(e)[:30]))
