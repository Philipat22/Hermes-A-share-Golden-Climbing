#!/usr/bin/env python3
import sys, re, os
sys.stdout.reconfigure(encoding='utf-8')
import tushare as ts
from dotenv import load_dotenv; load_dotenv()

pro = ts.pro_api(os.getenv('TUSHARE_PRO_TOKEN'))

# 读取得分数据
with open('scan_report_20260429_1400_phase1.md', encoding='utf-8') as f:
    t = f.read()

pairs = []
last_t = None
for line in t.split('\n'):
    tm = re.search(r'\((\d{6}\.S[ZH])\)', line)
    if tm: last_t = tm.group(1)
    sm = re.search(r'([+-]\d+\.\d+)', line)
    if sm and last_t:
        pairs.append((last_t, float(sm.group(1))))
        last_t = None

scores = {}
for tkr, sc in pairs:
    scores[tkr] = sc

all_tickers = list(scores.keys())
print(f'提取得分: {len(all_tickers)} 只', file=sys.stderr)

# 查股价
prices = {}
for tkr in all_tickers:
    try:
        df = pro.daily(ts_code=tkr, start_date='20260414', end_date='20260415')
        if df is not None and len(df) > 0:
            prices[tkr] = float(df.iloc[0]['close'])
    except:
        pass
print(f'获取股价: {len(prices)} 只', file=sys.stderr)

# 筛选 <=80
hits = []
for tkr, sc in scores.items():
    px = prices.get(tkr, 999)
    if px <= 80:
        info = pro.stock_basic(ts_code=tkr, fields='name')
        name = info.iloc[0]['name'] if info is not None and len(info) > 0 else '??'
        hits.append((sc, tkr, name, px))

hits.sort(key=lambda x: -x[0])

print(f'\n=== 股价<=80元的标的 (共{len(hits)}只) ===')
print(f'{"得分":>6s} {"代码":12s} {"名称":10s} {"股价":>6s}')
print('-' * 50)
for sc, tkr, name, px in hits[:30]:
    print(f'{sc:+5.1f}  {tkr} {name:10s} ￥{px:.2f}')
print(f'...共{len(hits)}只, 显示前30只')
