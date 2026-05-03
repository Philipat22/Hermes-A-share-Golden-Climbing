import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import tushare as ts
pro = ts.pro_api('5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7')

ticker = '600396.SH'
print(f'=== {ticker} ===')
basic = pro.stock_basic(ts_code=ticker, fields='name,industry,main_business,list_date')
if len(basic) > 0:
    b = basic.iloc[0]
    print(f'股票名称: {b.get("name", "N/A")}')
    print(f'行业: {b.get("industry", "N/A")}')
    print(f'主营: {b.get("main_business", "N/A")}')
    print(f'上市日期: {b.get("list_date", "N/A")}')

# Recent price
daily = pro.daily(ts_code=ticker, start_date='20260423', end_date='20260430', fields='trade_date,close,pct_chg,vol')
if len(daily) > 0:
    print('\n近期行情:')
    for _, row in daily.sort_values('trade_date', ascending=False).iterrows():
        print(f"  {row['trade_date']}: 收盘 {row['close']} | 涨跌 {row['pct_chg']}%")

# Financials
fin = pro.fina_indicator(ts_code=ticker, start_date='20250331', fields='roe,net_profit_ratio,debt_to_assets,pe,pb')
if len(fin) > 0:
    f = fin.iloc[0]
    print(f'\n财务指标: ROE={f.get("roe","N/A")} | 净利率={f.get("net_profit_ratio","N/A")} | 负债率={f.get("debt_to_assets","N/A")} | PE={f.get("pe","N/A")} | PB={f.get("pb","N/A")}')
