#!/usr/bin/env python3
"""Phase 2: 13位LLM大师分析5只低价候选股"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.main_astock import run_astock_analysis
from src.utils.display import print_trading_card

tickers = ['002023.SZ','000028.SZ','002049.SZ','002142.SZ','000426.SZ']
names = {'002023.SZ':'海特高新','000028.SZ':'国药一致','002049.SZ':'紫光国微',
         '002142.SZ':'宁波银行','000426.SZ':'兴业银锡'}
sectors = {'002023.SZ':'军工','000028.SZ':'医药','002049.SZ':'半导体',
           '002142.SZ':'银行','000426.SZ':'有色金属'}

llm_agents = [
    'warren_buffett','charlie_munger','ben_graham','aswath_damodaran',
    'cathie_wood','peter_lynch','phil_fisher','bill_ackman',
    'michael_burry','stanley_druckenmiller','nassim_taleb',
    'rakesh_jhunjhunwala','mohnish_pabrai',
]

print('=' * 65)
print('PHASE 2: 13位LLM大师 x 5只候选股 = 65次DeepSeek调用')
for t in tickers:
    print(f'  {t} {names[t]:8s} ¥? ({sectors[t]})')
print(f'开始分析...')
print('=' * 65)
sys.stdout.flush()

result = run_astock_analysis(
    tickers=tickers,
    selected_analysts=llm_agents,
    show_reasoning=False,
)

print('\n' + '=' * 65)
print('交易指令卡')
print('=' * 65)

for t in tickers:
    print()
    try:
        print_trading_card(t, result['analyst_signals'])
    except Exception as e:
        print(f'[ERROR] 打印{t}指令卡失败: {e}')
    print()

print('\n' + '=' * 65)
print('大师推理摘要')
print('=' * 65)

signals = result.get('analyst_signals', {})
for t in tickers:
    print(f'\n{"─" * 60}')
    print(f'📋 {t} {names[t]} ({sectors[t]})')
    print(f'{"─" * 60}')
    
    count = 0
    for agent_key in sorted(signals.keys()):
        if t in signals[agent_key]:
            s = signals[agent_key][t]
            reasoning = s.get('reasoning', '')
            sig = s.get('signal', '?')
            conf = s.get('confidence', 0)
            
            # reasoning可能是dict或str
            if isinstance(reasoning, dict):
                reasoning = str(reasoning)
            
            if reasoning and len(str(reasoning).strip()) > 5:
                dn = agent_key.replace('_agent','').replace('_',' ').title()
                brief = str(reasoning)[:200].replace('\n',' ')
                print(f'\n  {dn} ({sig}, {conf}%):')
                print(f'    {brief}')
                count += 1
    
    if count == 0:
        print('  (大师未给出有效推理)')

print(f'\n{"=" * 65}')
print(f'分析完成')
print(f'{"=" * 65}')
