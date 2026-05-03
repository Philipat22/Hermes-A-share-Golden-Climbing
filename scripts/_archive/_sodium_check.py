import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import tushare as ts
pro = ts.pro_api('5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7')

tickers = [('002245.SZ', '蔚蓝锂芯'), ('002074.SZ', '国轩高科')]
for ticker, name in tickers:
    print(f'=== {name} ({ticker}) ===')
    try:
        basic = pro.stock_basic(ts_code=ticker, fields='industry,main_business')
        if len(basic) > 0:
            print(f'  行业: {basic.iloc[0].get("industry","N/A")}')
            print(f'  主营: {basic.iloc[0].get("main_business","N/A")[:100]}')
        
        # Recent news
        news = pro.major_news(ts_code=ticker, start_date='20260401', end_date='20260430', fields='title,pub_date')
        if len(news) > 0:
            for _, row in news.head(5).iterrows():
                print(f'  [{row["pub_date"]}] {row["title"][:70]}')
        else:
            print('  (no recent news)')
    except Exception as e:
        print(f'  Error: {e}')
    print()
