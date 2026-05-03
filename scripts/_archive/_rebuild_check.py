#!/usr/bin/env python3
import ast, sys
sys.path.insert(0, r'D:\AIHedgeFund\ai-hedge-fund-main')

with open(r'D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_fetcher.py','r',encoding='utf-8') as f:
    src = f.read()

# Find where classes are USED (not defined - there are no classes in data_fetcher)
CLASS_NAMES = {'FinancialMetrics','FinancialLineItem','PriceBar','NewsItem','MarginTrade','LimitUpData','AStockInfo'}
usage_lines = []
for i, line in enumerate(src.splitlines(), 1):
    for cn in CLASS_NAMES:
        if cn in line and 'class ' not in line and '#' not in line.split(cn)[0]:
            usage_lines.append((i, cn, line.strip()[:100]))
            break

print(f'Class usages in data_fetcher.py:')
for lineno, cn, snippet in usage_lines:
    print(f'  L{lineno:3d} [{cn}]: {snippet}')

# Check what each class needs based on usage
print()
print('=== Key usage patterns ===')
# Find return statements that construct classes
for i, line in enumerate(src.splitlines(), 1):
    if any(f'return {cn}(' in line or f'return {cn}(' in line.replace(' ','') for cn in CLASS_NAMES):
        print(f'  L{i}: {line.strip()[:120]}')
    if any(f'{cn}(' in line and 'FinancialLineItem' in src[:src.index(line.split(cn)[0]+cn)] src[max(0,src.find(line)-500):src.find(line)+200]' in line for cn in ['FinancialMetrics','PriceBar']):
        pass
