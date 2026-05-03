#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""半导体板块选股 - 业绩+资金双因子扫描"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import tushare as ts
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

pro = ts.pro_api(os.getenv('TUSHARE_PRO_TOKEN'))

print('=' * 65)
print('半导体板块全量扫描')
print('=' * 65)

# 1. 全量股票
all_stocks = pro.stock_basic(list_status='L', fields='ts_code,symbol,name,area,industry,list_date,market')
if all_stocks is None or len(all_stocks) == 0:
    print('无法获取股票列表')
    sys.exit(1)

# 2. 概念板块查半导体
print('查概念板块...')
concept_list = pro.concept()
semi_concept_codes = set()
if concept_list is not None and len(concept_list) > 0:
    kw_list = ['半导体', '芯片', '集成电路', '光刻机', '存储芯片', 'AI芯片']
    for kw in kw_list:
        matched = concept_list[concept_list['name'].str.contains(kw, na=False)]
        for _, c in matched.iterrows():
            try:
                cd = pro.concept_detail(id=c['code'])
                if cd is not None and len(cd) > 0:
                    semi_concept_codes.update(cd['ts_code'].tolist())
                    print('  找到概念: %s (%d只)' % (c['name'], len(cd)))
            except:
                pass

# 3. 直接关键词匹配股票名称和行业
kw_stock = set()
for _, r in all_stocks.iterrows():
    n = str(r.get('name', ''))
    ind = str(r.get('industry', ''))
    all_text = n + '|' + ind
    for kw in ['半导体', '芯片', '集成电路', '分立器件', '封测', '晶圆', '光刻', '硅片', 'IGBT']:
        if kw in all_text:
            kw_stock.add(r['ts_code'])
            break

# 合并
semi_set = semi_concept_codes | kw_stock
print('\n半导体相关标的总数: %d只' % len(semi_set))
print('  - 概念板块匹配: %d只' % len(semi_concept_codes))
print('  - 名称/行业匹配: %d只' % len(kw_stock))

# 4. 财务筛选
today = '20260429'
print('\n财务筛选条件: 上市>1年 | ROE>5%% | 营收增长>0%% | PE<100')
print()

candidates = []
total = len(semi_set)
count = 0
for ts_code in sorted(semi_set)[:100]:
    count += 1
    if count % 20 == 0:
        print('  处理中: %d/%d' % (count, min(total, 100)))
    try:
        info = all_stocks[all_stocks['ts_code'] == ts_code]
        if len(info) == 0:
            continue
        name = info.iloc[0]['name']
        list_date = str(info.iloc[0].get('list_date', ''))

        # 上市>1年
        if list_date >= '20250429':
            continue

        # 近20日行情
        daily = pro.daily(ts_code=ts_code, start_date='20260401', end_date=today)
        if daily is None or len(daily) < 5:
            continue
        daily = daily.sort_values('trade_date', ascending=False)
        close = float(daily.iloc[0]['close'])
        pct_5d = (close / float(daily.iloc[min(4, len(daily)-1)]['close']) - 1) * 100
        pct_20d = (close / float(daily.iloc[min(19, len(daily)-1)]['close']) - 1) * 100 if len(daily) >= 20 else None
        vol_5 = float(daily.head(5)['vol'].mean())
        vol_20 = float(daily.head(20)['vol'].mean()) if len(daily) >= 20 else vol_5
        vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0

        # 财务
        fina = pro.fina_indicator(ts_code=ts_code, start_date='20250101', end_date=today, limit=1)
        if fina is None or len(fina) == 0:
            continue
        f = fina.iloc[0]

        roe = float(f.get('roe', 0) or 0)
        if roe > 1:
            roe /= 100
        if roe < 0.05:
            continue

        rev_growth = float(f.get('tr_yoy', 0) or 0)
        if rev_growth > 1:
            rev_growth /= 100
        if rev_growth <= 0:
            continue

        net_margin = float(f.get('netprofit_margin', 0) or 0)
        if net_margin > 1:
            net_margin /= 100

        eps = float(f.get('eps', 0) or 0)

        # PE
        latest_trade_date = daily.iloc[0]['trade_date']
        db = pro.daily_basic(ts_code=ts_code, trade_date=latest_trade_date)
        if db is None or len(db) == 0:
            db = pro.daily_basic(ts_code=ts_code, start_date='20260401', end_date=today, limit=1)
        pe = 100.0
        if db is not None and len(db) > 0:
            pe = float(db.iloc[0].get('pe', 100) or 100)
        if pe > 100:
            continue

        candidates.append(dict(ts_code=ts_code, name=name, price=close,
            roe=roe, rev_growth=rev_growth, net_margin=net_margin,
            eps=eps, pe=pe, pct_5d=pct_5d, pct_20d=pct_20d, vol_ratio=vol_ratio))
    except Exception as e:
        pass

# 5. 评分
def score_c(c):
    s = 0
    s += max(-20, min(20, c['pct_5d']))
    s += max(0, min(20, c['pct_20d'] or 0)) if c['pct_20d'] is not None else 10
    s += max(0, min(20, c['roe'] * 100))
    s += max(0, min(20, c['rev_growth'] * 100))
    pe_score = 10 if c['pe'] <= 20 else max(0, 10 - (c['pe'] - 20) / 8)
    s += pe_score
    s += max(-10, min(10, (c['vol_ratio'] - 1) * 10))
    return s

for c in candidates:
    c['score'] = score_c(c)

candidates.sort(key=lambda x: -x['score'])

print('\n通过筛选: %d 只' % len(candidates))
print()
hdr = '排名|代码|名称|现价|ROE|营收增|净利率|PE|5日涨|20日涨|量比|评分'
sep = '|' + '-' * (len(hdr) - 2) + '|'
print(hdr.replace('|', ' '))
print(sep.replace('|', ' '))

for i, c in enumerate(candidates[:15]):
    r = '%+.1f%%' % c['pct_5d']
    r20 = '%+.1f%%' % c['pct_20d'] if c['pct_20d'] is not None else 'N/A'
    print(' %2d  %-10s %-10s %6.2f %5.1f%% %6.1f%% %5.1f%% %4.0f  %7s %7s %5.2fx %5.1f' % (
        i+1, c['ts_code'], c['name'], c['price'],
        c['roe']*100, c['rev_growth']*100, c['net_margin']*100,
        c['pe'], r, r20, c['vol_ratio'], c['score']))

# 6. 精选推荐
print()
print('=' * 65)
print('精选推荐（Top 5）')
print('=' * 65)

for i, c in enumerate(candidates[:5]):
    print()
    print('【%d】%s (%s)  ¥%.2f' % (i+1, c['name'], c['ts_code'], c['price']))
    print('  基本面: ROE %.1f%% | 营收增长 %.1f%% | 净利率 %.1f%% | PE %.0f' % (
        c['roe']*100, c['rev_growth']*100, c['net_margin']*100, c['pe']))
    print('  量价: 5日%s | 20日%s | 量比%.2f' % (
        '%+.1f%%' % c['pct_5d'],
        '%+.1f%%' % c['pct_20d'] if c['pct_20d'] is not None else 'N/A',
        c['vol_ratio']))
    try:
        hk = pro.hk_hold(ts_code=c['ts_code'], start_date='20260301', end_date=today)
        if hk is not None and len(hk) > 0:
            hk = hk.sort_values('trade_date', ascending=False)
            ratio = float(hk.iloc[0].get('ratio', 0) or 0)
            print('  陆股通持股: %.2f%%' % ratio)
    except:
        pass
