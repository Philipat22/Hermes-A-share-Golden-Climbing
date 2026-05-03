#!/usr/bin/env python3
import sys, re, os
sys.stdout.reconfigure(encoding='utf-8')
from src.tools.a_stock_api import get_stock_info

with open('scan_report_20260429_1400_phase1.md', encoding='utf-8') as f:
    text = f.read()

# 解析得分——得分和ticker在不同行，用状态追踪
sector_scores = {}
current_sector = ''
ticker_pattern = re.compile(r'\((\d{6}\.S[ZH])\)')
score_pattern = re.compile(r'综合得分: ([+-]?\d+\.\d+)')
for line in text.split('\n'):
    if line.startswith('### '):
        current_sector = line.replace('### ', '').strip()
    tkm = ticker_pattern.search(line)
    if tkm:
        last_ticker = tkm.group(1)
    scm = score_pattern.search(line)
    if scm and last_ticker and current_sector:
        if last_ticker not in sector_scores:
            sector_scores[last_ticker] = (current_sector, float(scm.group(1)))

print(f'Parsed scores for {len(sector_scores)} stocks', file=sys.stderr)

# 已经有的股价数据（从之前执行输出抄录）
prices = {
    '000050.SZ': 8.25, '000045.SZ': 12.03, '000034.SZ': 34.90, '000676.SZ': 0,  # need
    '000158.SZ': 17.56, '000555.SZ': 16.00, '000409.SZ': 10.59, '000503.SZ': 0,
    '000401.SZ': 0, '000055.SZ': 0, '000655.SZ': 8.36, '000012.SZ': 0, '000619.SZ': 0,
    '000089.SZ': 0, '000520.SZ': 0, '000088.SZ': 0, '000429.SZ': 0, '000421.SZ': 0,
    '000548.SZ': 0, '000417.SZ': 0,
    '000521.SZ': 0, '000016.SZ': 0, '000810.SZ': 0, '000404.SZ': 0, '000651.SZ': 0,
    '002023.SZ': 12.72, '002111.SZ': 10.12, '002297.SZ': 14.62, '000519.SZ': 17.66, '000738.SZ': 21.83,
    '000768.SZ': 24.67, '002013.SZ': 0, '002077.SZ': 15.90,
    '000028.SZ': 26.45, '000403.SZ': 12.79, '000078.SZ': 0, '000411.SZ': 11.28, '000423.SZ': 0,
    '001309.SZ': 0, '002049.SZ': 70.18, '002119.SZ': 21.71, '002156.SZ': 46.99, '000547.SZ': 0,
    '000029.SZ': 19.73, '000014.SZ': 13.11, '000011.SZ': 8.31, '000006.SZ': 8.78, '000002.SZ': 0,
    '000426.SZ': 43.87, '000506.SZ': 17.49, '000612.SZ': 13.16, '000603.SZ': 40.14, '002155.SZ': 29.18,
    '000519.SZ': 17.66, '000425.SZ': 10.53, '000528.SZ': 9.67, '000157.SZ': 8.68, '000680.SZ': 0,
    '000546.SZ': 0, '000544.SZ': 0, '000685.SZ': 0, '000598.SZ': 0, '000605.SZ': 0,
    '000400.SZ': 26.24, '000009.SZ': 8.86, '000049.SZ': 27.17, '000507.SZ': 0, '000155.SZ': 15.99,
    '002142.SZ': 31.52, '002839.SZ': 0, '001227.SZ': 0, '002807.SZ': 0, '000001.SZ': 11.21,
    '000596.SZ': 0, '000568.SZ': 0, '000523.SZ': 9.69, '000505.SZ': 0, '000529.SZ': 0,
    '000975.SZ': 29.85, '001337.SZ': 48.18, '002237.SZ': 15.01,
    '000039.SZ': 11.88, '000301.SZ': 11.46, '000059.SH': 0, '000572.SZ': 0, '000559.SZ': 0,
}

hits = []
for t, (sec, sc) in sector_scores.items():
    px = prices.get(t, 0)
    if px > 0 and px <= 80:
        info = get_stock_info(t)
        name = info.name if info else '??'
        hits.append((sc, t, name, sec, px))

hits.sort(key=lambda x: -x[0])
print(f'\n{"得分":>6s} {"代码":12s} {"名称":8s} {"价格":>6s} {"板块":10s}')
print('-' * 50)
for sc, t, name, sec, px in hits:
    print(f'{sc:+5.1f}  {t} {name:8s} ￥{px:.2f} ({sec})')
print(f'\n总计: {len(hits)} 只 ￥<=80')
