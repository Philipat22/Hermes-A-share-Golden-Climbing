#!/usr/bin/env python3
import re
with open(r'D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_fetcher.py','r',encoding='utf-8') as f:
    content = f.read()
all_funcs = set()
for m in re.finditer(r'^def (\w+)\(', content, re.MULTILINE):
    all_funcs.add(m.group(1))
print('Functions in data_fetcher.py:')
for f in sorted(all_funcs):
    print('  ' + f)
print()
missing = ['get_pro_api','get_company_news','get_insider_trades','get_limit_list','get_north_money','get_market_context','get_stock_name_for_ticker','_simple_sentiment']
print('Missing from facade:')
for fn in missing:
    status = 'MISSING' if fn not in all_funcs else 'present'
    print('  ' + fn + ': ' + status)
