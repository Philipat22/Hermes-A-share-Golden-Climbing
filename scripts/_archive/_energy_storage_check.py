import os, sys, json
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import tushare as ts
pro = ts.pro_api('5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7')

print('=== 储能核心标的近期表现 ===')

# 储能核心标的
tickers = [
    ('300274.SZ', '阳光电源'),
    ('300750.SZ', '宁德时代'),
    ('300014.SZ', '亿纬锂能'),
    ('300763.SZ', '锦浪科技'),
    ('002459.SZ', '晶澳科技'),
    ('002594.SZ', '比亚迪'),
]

for tk, name in tickers:
    try:
        d = pro.daily(ts_code=tk, start_date='20260421', end_date='20260429')
        if len(d) >= 5:
            p0 = d.iloc[0]['close']
            p4 = d.iloc[4]['close']
            chg5 = (p0 - p4) / p4 * 100
            # 20日
            d20 = pro.daily(ts_code=tk, start_date='20260401', end_date='20260429')
            if len(d20) >= 20:
                p20_0 = d20.iloc[0]['close']
                p20_19 = d20.iloc[19]['close']
                chg20 = (p20_0 - p20_19) / p20_19 * 100
            else:
                chg20 = None
            vol = d.iloc[0]['vol']
            print(f'{name}({tk}) 5日:{chg5:+.1f}% 20日:{chg20:+.1f}% 现价:{p0} 成交:{vol//10000:.0f}万手')
    except Exception as e:
        print(f'{name} error: {e}')

# 申万电力设备指数
try:
    df = pro.index_daily(ts_code='801732.SI', start_date='20260421', end_date='20260429')
    print(f'\n=== 申万电力设备指数({len(df)}条) ===')
    print(df[['trade_date','close','pct_chg']].head(10).to_string(index=False))
except Exception as e:
    print(f'指数: {e}')

# 北向资金
try:
    mf = pro.moneyflow_hsgt(start_date='20260421', end_date='20260429')
    print(f'\n=== 北向资金(HSGT) ===')
    print(mf[['trade_date','hsgt_netbuy','south_netvol']].head(10).to_string(index=False))
except Exception as e:
    print(f'北向: {e}')
