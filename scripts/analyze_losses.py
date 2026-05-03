"""
Analyze WHY the strategy lost money: which stocks, sectors, and patterns.
"""
import pandas as pd, numpy as np, os, warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
load_dotenv(os.path.join(ROOT, '.env'))
PRICE = os.path.join(ROOT, 'data', 'cache', 'backtest_prices_extended.pkl')

df = pd.read_csv(os.path.join(ROOT, 'quant_archive', '2026-05', 'backtest_v2_dual_trades.csv'))
df['entry_date'] = pd.to_datetime(df['entry_date'])
df['exit_date'] = pd.to_datetime(df['exit_date'])
df['year'] = df['exit_date'].dt.year

print('=' * 72)
print('V2-B (最佳版本): 各年拆解')
print('=' * 72)

for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr]
    n = len(sub)
    wr = (sub['net_return'] > 0).mean()
    avg = sub['net_return'].mean()
    total = sub['net_return'].sum()
    sl = (sub['exit_reason'] == 'stop_loss').mean()
    mature = (sub['exit_reason'] == 'matured').mean()

    print(f'\n── {yr}年: {n}笔交易, 赢率{wr:.1%}, 平均{avg:.4%}, 累计{total:.2%} ──')
    print(f'  止损占比{sl:.1%},  正常到期{mature:.1%}')

    # Top losers
    bot = sub.nsmallest(10, 'net_return')
    print(f'  亏损TOP10:')
    for _, r in bot.iterrows():
        print(f'    {r["net_return"]:>7.2%}  {r["symbol"]:12s}  [{r["regime"]:12s}]  {r["exit_reason"]}')

    print(f'  盈利TOP10:')
    top = sub.nlargest(10, 'net_return')
    for _, r in top.iterrows():
        print(f'    +{r["net_return"]:.2%}  {r["symbol"]:12s}  [{r["regime"]:12s}]  {r["exit_reason"]}')

print('\n' + '=' * 72)
print('V2-B: 亏损最多的股票（净亏损累计）')
print('=' * 72)
stock_loss = df.groupby('symbol')['net_return'].sum().sort_values()
for sym, val in stock_loss.head(25).items():
    sub = df[df['symbol'] == sym]
    cnt = len(sub)
    wr = (sub['net_return'] > 0).mean()
    avg = sub['net_return'].mean()
    print(f'  {sym:12s}  净亏{val:.2%}  平均{avg:.4%}  交易{cnt}次  赢率{wr:.1%}')

print('\n' + '=' * 72)
print('V2-B: 盈利最多的股票（净盈利累计）')
print('=' * 72)
stock_win = df.groupby('symbol')['net_return'].sum().sort_values(ascending=False)
for sym, val in stock_win.head(25).items():
    sub = df[df['symbol'] == sym]
    cnt = len(sub)
    wr = (sub['net_return'] > 0).mean()
    avg = sub['net_return'].mean()
    print(f'  {sym:12s}  净赚{val:.2%}  平均{avg:.4%}  交易{cnt}次  赢率{wr:.1%}')

# Regime transition analysis
print('\n' + '=' * 72)
print('按退出月份看表现（识别市场切换点）')
print('=' * 72)
df['yearmonth'] = df['exit_date'].dt.strftime('%Y-%m')
for ym in sorted(df['yearmonth'].unique()):
    sub = df[df['yearmonth'] == ym]
    n = len(sub)
    wr = (sub['net_return'] > 0).mean()
    avg = sub['net_return'].mean()
    total = sub['net_return'].sum()
    regimes = sub['regime'].value_counts().to_dict()
    regime_str = ', '.join([f'{k}={v}' for k, v in sorted(regimes.items())])
    print(f'  {ym}: {n}笔, WR {wr:.1%}, avg {avg:.4%}, total {total:.2%}  [{regime_str}]')

# Stock sector mapping
print('\n' + '=' * 72)
print('尝试分解股票所属行业...')
print('=' * 72)
try:
    import tushare as ts
    pro = ts.pro_api(os.getenv('TUSHARE_PRO_TOKEN', ''))

    # Get stock basic info
    symbols_bought = df['symbol'].unique()
    stock_info = {}
    # Batch query
    for i in range(0, len(symbols_bought), 50):
        batch = symbols_bought[i:i+50]
        for s in batch:
            try:
                info = pro.stock_basic(ts_code=s, fields='ts_code,name,industry,area')
                if info is not None and len(info) > 0:
                    stock_info[s] = info.iloc[0]
            except:
                pass

    # Add sector info to trades
    df['industry'] = df['symbol'].map(lambda x: stock_info[x]['industry'] if x in stock_info else 'unknown')
    df['name'] = df['symbol'].map(lambda x: stock_info[x]['name'] if x in stock_info else 'unknown')

    # By industry
    print('\n按行业汇总（净收益）:')
    ind_agg = df.groupby('industry')['net_return'].agg(['sum','mean','count']).sort_values('sum')
    for ind, row in ind_agg.iterrows():
        print(f'  {ind:12s}  净{row["sum"]:.2%}  平均{row["mean"]:.4%}  交易{int(row["count"])}次')

    # Top 10 losers with names
    print('\n亏损最大单笔交易（含股票名称）:')
    bot_all = df.nsmallest(20, 'net_return')
    for _, r in bot_all.iterrows():
        print(f'  {r["net_return"]:>7.2%}  {r["symbol"]:12s} ({r.get("name","?"):8s})  {r["industry"]:10s}  [{r["regime"]:12s}]')

    print('\n盈利最大单笔交易（含股票名称）:')
    top_all = df.nlargest(20, 'net_return')
    for _, r in top_all.iterrows():
        print(f'  +{r["net_return"]:.2%}  {r["symbol"]:12s} ({r.get("name","?"):8s})  {r["industry"]:10s}  [{r["regime"]:12s}]')

except Exception as e:
    print(f'Tushare查询失败: {e}')
    print('跳过行业分析。')

print('\nDone!')
