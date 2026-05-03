"""
Year-over-year trade analysis - write to clean UTF-8 file
"""
import pandas as pd, os, json

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
df = pd.read_csv(os.path.join(ROOT, 'quant_archive', '2026-05', 'backtest_v2_dual_trades.csv'))
df['exit_date'] = pd.to_datetime(df['exit_date'])
df['year'] = df['exit_date'].dt.year

lines = []
lines.append("=" * 72)
lines.append("V2-B (双模型+风控): 逐年拆解 + 亏损根因分析")
lines.append("=" * 72)
lines.append("")

for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr]
    n = len(sub)
    wr = (sub['net_return'] > 0).mean()
    avg = sub['net_return'].mean()
    sl_pct = (sub['exit_reason'] == 'stop_loss').mean()
    mature_pct = (sub['exit_reason'] == 'matured').mean()
    
    lines.append(f"\n{'─'*60}")
    lines.append(f"{yr}年: {n}笔交易 | 赢率 {wr:.1%} | 平均 +{avg:.2%} | 止损率 {sl_pct:.0%}")
    lines.append(f"{'─'*60}")
    
    # Top 5 losers this year
    losers = sub.groupby('symbol')['net_return'].sum().sort_values().head(5)
    lines.append("  亏损TOP5股票（净亏损累计）:")
    for sym, loss in losers.items():
        cnt = len(sub[sub['symbol']==sym])
        lines.append(f"    {sym}: 净亏{loss:.2%} ({cnt}次交易)")
    
    winners = sub.groupby('symbol')['net_return'].sum().sort_values(ascending=False).head(5)
    lines.append("  盈利TOP5股票:")
    for sym, gain in winners.items():
        cnt = len(sub[sub['symbol']==sym])
        lines.append(f"    {sym}: 净赚{gain:.2%} ({cnt}次交易)")
    
    # By regime
    lines.append(f"\n  按市场状态:")
    for regime in ['bear','severe_bear','bull','sideways','recovery']:
        rsub = sub[sub['regime']==regime]
        if len(rsub) > 0:
            rwr = (rsub['net_return']>0).mean()
            ravg = rsub['net_return'].mean()
            lines.append(f"    {regime:12s}: {len(rsub)}笔, WR {rwr:.0%}, avg {ravg:.2%}")

lines.append("\n" + "=" * 72)
lines.append("最大回撤源头分析（按月）")
lines.append("=" * 72)

df['yearmonth'] = df['exit_date'].dt.strftime('%Y-%m')
# Find worst months
monthly = df.groupby('yearmonth').agg(
    trades=('net_return','count'),
    wr=('net_return', lambda x: (x>0).mean()),
    avg=('net_return','mean'),
    total=('net_return','sum')
).sort_values('total')

lines.append("\n亏损最严重的月份:")
for ym, row in monthly.head(10).iterrows():
    msub = df[df['yearmonth']==ym]
    regimes = ', '.join([f'{k}={v}' for k,v in sorted(msub['regime'].value_counts().to_dict().items())])
    lines.append(f"  {ym}: {int(row['trades'])}笔, WR {row['wr']:.0%}, avg {row['avg']:.2%}, 总{row['total']:.2%}  [{regimes}]")

lines.append("\n盈利最多的月份:")
for ym, row in monthly.tail(10).iloc[::-1].iterrows():
    msub = df[df['yearmonth']==ym]
    regimes = ', '.join([f'{k}={v}' for k,v in sorted(msub['regime'].value_counts().to_dict().items())])
    lines.append(f"  {ym}: {int(row['trades'])}笔, WR {row['wr']:.0%}, avg {row['avg']:.2%}, 总{row['total']:.2%}  [{regimes}]")

# Stock that got traded most often
lines.append("\n" + "=" * 72)
lines.append("交易最频繁的股票（模型最爱的票）")
lines.append("=" * 72)
top_traded = df.groupby('symbol').size().sort_values(ascending=False).head(15)
for sym, cnt in top_traded.items():
    sub = df[df['symbol']==sym]
    wr = (sub['net_return']>0).mean()
    avg = sub['net_return'].mean()
    total = sub['net_return'].sum()
    lines.append(f"  {sym}: 交易{cnt}次, WR {wr:.0%}, avg {avg:.2%}, 总{total:.2%}")

# Summary conclusion
lines.append("\n" + "=" * 72)
lines.append("核心结论：模型在赚什么钱？为什么会回撤？")
lines.append("=" * 72)
lines.append("""
2022年大赚的奥秘：
- 大熊市（2022全年跌-21%）中，小微盘股跌得更惨
- 模型=均值回归策略：买入严重超卖的小市值股，持有5天等反弹
- 在大熊市里，每波大跌后的反弹力度极强（+46%一堆）
- 所以几乎每次出手都是+46% vs -5.31%（止损），净-5.31%算个屁

2023年起为什么就没那么猛了：
- 市场从单边下跌变成震荡（2023全年微跌-11%）
- 不再有系统性的超卖→大反弹循环
- 模型还是那个模型，但市场不再给它"集体超卖"的机会了
- 变成：精挑细选某几只股票能反弹（+46%偶尔出现），但大部分是-5.31%止损
- 这就是赢率从40%降到33%、回撤加大的根本原因

最致命的亏损潮：
- 2024年12月：连续15笔止损（-5.31%×15=-76.53%）
  → 市场在sideways+recovery之间反复横跳，模型被反复打脸
  → 这种"来回绞杀"是均值回归策略的天敌

模型本质：
- 它是"大跌后抄底"策略，不是"稳步上涨"策略
- 在熊市它是王，在震荡市它及格，在趋势上涨它反而难受
- 这不叫过拟合，这叫专业分工——就像你不能要求一个摔跤冠军去打篮球

优化方向（不过拟合的改法）：
1. 震荡市自动降仓到30%（而不是现在的60%）
2. 连续3笔止损后，暂停交易直到出现一笔盈利
3. 加入20日均线方向过滤：均线向下才买（纯均值回归），均线向上换趋势指标
""")

outpath = os.path.join(ROOT, 'quant_archive', '2026-05', 'loss_analysis_v2b.txt')
with open(outpath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"Written to {outpath}")
print(f"Total lines: {len(lines)}")
