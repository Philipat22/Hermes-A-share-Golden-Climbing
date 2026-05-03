#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""半导体精选标的 × 19位大师全量分析"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.main_astock import run_astock_analysis

# 半导体精选标的
tickers = [
    '603501.SH',  # 韦尔股份 - Tier 1
    '002371.SZ',  # 北方华创 - Tier 1
    '688012.SH',  # 中微公司 - Tier 2
    '688008.SH',  # 澜起科技 - Tier 2
    '002185.SZ',  # 华天科技 - Tier 2
    '000636.SZ',  # 风华高科 - Tier 2
]

names = {
    '603501.SH':'韦尔股份','002371.SZ':'北方华创','688012.SH':'中微公司',
    '688008.SH':'澜起科技','002185.SZ':'华天科技','000636.SZ':'风华高科'
}

# 全部19位分析师
all_analysts = [
    'warren_buffett','charlie_munger','ben_graham','aswath_damodaran',
    'cathie_wood','peter_lynch','phil_fisher','bill_ackman',
    'michael_burry','stanley_druckenmiller','nassim_taleb',
    'rakesh_jhunjhunwala','mohnish_pabrai',
    'technical_analyst','sentiment_analyst','growth_analyst',
    'valuation_analyst','news_sentiment_analyst',
    'risk_manager',
]

print('=' * 70)
print('半导体精选 × 19位大师分析')
print('6只标的 × 19位分析师 = 114次DeepSeek调用')
print('=' * 70)
for t in tickers:
    print('  %s  %s' % (t, names[t]))
print('正在运行，约5-8分钟...')
print()
sys.stdout.flush()

result = run_astock_analysis(
    tickers=tickers,
    selected_analysts=all_analysts,
    show_reasoning=False,
)

# 输出格式
signals = result.get('analyst_signals', {})

# LLM大师
llm_masters = [
    ('Warren Buffett','warren_buffett_agent'),('Charlie Munger','charlie_munger_agent'),
    ('Ben Graham','ben_graham_agent'),('Aswath Damodaran','aswath_damodaran_agent'),
    ('Cathie Wood','cathie_wood_agent'),('Peter Lynch','peter_lynch_agent'),
    ('Phil Fisher','phil_fisher_agent'),('Bill Ackman','bill_ackman_agent'),
    ('Michael Burry','michael_burry_agent'),('Stanley Druckenmiller','stanley_druckenmiller_agent'),
    ('Nassim Taleb','nassim_taleb_agent'),('Rakesh Jhunjhunwala','rakesh_jhunjhunwala_agent'),
    ('Mohnish Pabrai','mohnish_pabrai_agent'),
]
comp_agents = [
    ('Technical','technical_analyst_agent'),('Sentiment','sentiment_analyst_agent'),
    ('Growth','growth_analyst_agent'),('Valuation','valuation_analyst_agent'),
    ('News','news_sentiment_analyst_agent'),
]

for t in tickers:
    print()
    print('=' * 70)
    print('  %s (%s)' % (names[t], t))
    print('=' * 70)
    
    bull, bear, neutral = 0, 0, 0
    
    # LLM大师
    print('  [LLM大师]')
    for dn, ak in llm_masters:
        s = signals.get(ak, {}).get(t, {})
        sig = s.get('signal', '?')
        conf = s.get('confidence', 0)
        rsn = s.get('reasoning', '')
        if isinstance(rsn, dict): rsn = str(rsn)
        
        if sig == 'bullish': sig_c = '🟢'; bull += 1
        elif sig == 'bearish': sig_c = '🔴'; bear += 1
        else: sig_c = '🟡'; neutral += 1
        
        rsn_short = str(rsn)[:150].replace('\n',' ')
        print('  %s %-18s (%3.0f%%) %s' % (sig_c, dn, (conf or 0), rsn_short))
    
    # 计算型
    print('  [计算型Agent]')
    for dn, ak in comp_agents:
        s = signals.get(ak, {}).get(t, {})
        sig = s.get('signal', '?')
        conf = s.get('confidence', 0)
        
        if sig == 'bullish': sig_c = '🟢'; bull += 1
        elif sig == 'bearish': sig_c = '🔴'; bear += 1
        else: sig_c = '🟡'; neutral += 1
        
        rsn = s.get('reasoning', '')
        if isinstance(rsn, dict): rsn = str(rsn)
        rsn_short = str(rsn)[:100].replace('\n',' ')
        print('  %s %-18s (%3.0f%%) %s' % (sig_c, dn, (conf or 0), rsn_short))
    
    print('  → 大师共识: 🟢%d多 / 🔴%d空 / 🟡%d中' % (bull, bear, neutral))

# 汇总对比表格
print()
print()
print('=' * 70)
print('  汇总: 多/空/中 分布')
print('=' * 70)

hdr = '%-14s' % '标的'
for dn, _ in llm_masters:
    hdr += ' %8s' % dn.split()[-1][:6]
print('  ' + hdr)

for t in tickers:
    line = '%-14s' % names[t]
    for _, ak in llm_masters:
        s = signals.get(ak, {}).get(t, {})
        sig = s.get('signal', '?')
        conf = s.get('confidence', 0)
        if sig == 'bullish': sc = '🟢%.0f' % (conf or 0)
        elif sig == 'bearish': sc = '🔴%.0f' % (conf or 0)
        else: sc = '🟡%.0f' % (conf or 0)
        line += ' %8s' % sc
    print('  ' + line)
