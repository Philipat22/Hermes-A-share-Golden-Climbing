#!/usr/bin/env python3
import sys, re, os
sys.stdout.reconfigure(encoding='utf-8')
import tushare as ts
from dotenv import load_dotenv; load_dotenv()
from src.tools.a_stock_api import get_stock_info

pro = ts.pro_api(os.getenv('TUSHARE_PRO_TOKEN'))

# 读Phase1报告提取ticker
text = open('scan_report_20260429_1400_phase1.md', encoding='utf-8').read()
tickers = re.findall(r'\((\d{6}\.S[ZH])\)', text)
tickers = list(dict.fromkeys(tickers))
print(f'Total unique tickers: {len(tickers)}', file=sys.stderr)

# 查股价
prices = {}
for t in tickers:
    try:
        df = pro.daily(ts_code=t, start_date='20260414', end_date='20260415')
        if df is not None and len(df) > 0:
            prices[t] = float(df.iloc[0]['close'])
    except Exception as e:
        pass
print(f'Got prices for: {len(prices)}', file=sys.stderr)

# 从报告解析板块和得分
sector_scores = {}
current_sector = ''
last_ticker = None
for line in text.split('\n'):
    if line.startswith('### '):
        current_sector = line.replace('### ', '').strip()
        last_ticker = None
    tkm = re.search(r'\((\d{6}\.S[ZH])\)', line)
    if tkm:
        last_ticker = tkm.group(1)
        # 如果当前行还包含得分（同一行）
        scm = re.search(r'得分: ([+-]?\d+\.\d+)', line)
        if scm and current_sector:
            sector_scores[last_ticker] = (current_sector, float(scm.group(1)))
    # 得分在单独一行
    scm = re.search(r'综合得分: ([+-]?\d+\.\d+)', line)
    if scm and last_ticker and current_sector and last_ticker not in sector_scores:
        sector_scores[last_ticker] = (current_sector, float(scm.group(1)))

# 输出价格<=80的
print('\n=== 股价<=80元的标的 ===')
hits = []
for t in sorted(prices, key=lambda x: prices[x]):
    px = prices[t]
    if px <= 80:
        info = get_stock_info(t)
        name = info.name if info else '??'
        sec, sc = sector_scores.get(t, ('??', 0))
        hits.append((sc, t, name, sec, px))

hits.sort(key=lambda x: -x[0])
for sc, t, name, sec, px in hits:
    print(f'{sc:+5.1f}  {t} {name:8s} ￥{px:.2f}  ({sec})')
print(f'\nTotal under 80: {len(hits)}')
