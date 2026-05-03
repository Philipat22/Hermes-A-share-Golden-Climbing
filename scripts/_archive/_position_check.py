import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import tushare as ts
pro = ts.pro_api('5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7')

tickers = [
    ('002245.SZ', '蔚蓝锂芯'),
    ('002074.SZ', '国轩高科'),
    ('603906.SH', '龙蟠科技'),
    ('600726.SH', '华电国际'),  # 暂用华电国际，需要确认
]

for tk, name in tickers:
    try:
        d5 = pro.daily(ts_code=tk, start_date='20260423', end_date='20260429')
        d20 = pro.daily(ts_code=tk, start_date='20260401', end_date='20260429')
        if len(d5) >= 5 and len(d20) >= 20:
            p0 = d5.iloc[0]['close']
            p4 = d5.iloc[4]['close']
            p19 = d20.iloc[19]['close']
            chg5 = (p0 - p4) / p4 * 100
            chg20 = (p0 - p19) / p19 * 100
            vol = d5.iloc[0]['vol']
            print(f'{name}({tk}) 现价:{p0} 5日:{chg5:+.1f}% 20日:{chg20:+.1f}% 成交:{vol//10000:.0f}万手')
        else:
            print(f'{name}({tk}): 数据不足5日{len(d5)}条 20日{len(d20)}条')
    except Exception as e:
        print(f'{name}({tk}) error: {e}')

# 检查华电辽能真实代码
try:
    basic = pro.stock_basic(ts_code='600726.SH')
    print(f'\n600726.SH: {basic.iloc[0]["name"] if len(basic)>0 else "not found"}')
except:
    pass

# 尝试找华电辽能
try:
    result = pro.stock_basic(name='华电')
    print(f'华电相关: {result[["ts_code","name","industry"]].to_string(index=False)}')
except:
    pass