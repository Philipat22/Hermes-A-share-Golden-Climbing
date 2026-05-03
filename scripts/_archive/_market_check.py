"""Check broader market context for sector analysis"""
import tushare as ts
import json
from datetime import datetime

TOKEN = '5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7'
pro = ts.pro_api(TOKEN)

# 大盘位置
indices = {'000016.SH': '上证50', '000300.SH': '沪深300', '000001.SH': '上证指数'}
for idx, name in indices.items():
    df = pro.index_daily(ts_code=idx, start_date='20260105', end_date='20260428')
    if df is not None and len(df) > 0:
        last5 = df.head(5)['pct_chg'].mean()
        ytd = df['pct_chg'].sum()
        close = df.iloc[0]['close']
        print(f'{name:8s}: 近5日均 {last5:+.2f}% | YTD {ytd:+.1f}% | 最新 {close:.0f}')

# 风格指数 - 大盘vs小盘
print()
try:
    for code, name in [('399303.SZ', '国证2000（小盘）')]:
        df = pro.index_daily(ts_code=code, start_date='20260421', end_date='20260428')
        if df is not None and len(df) > 0:
            print(f'{name}: 近5日均 {df["pct_chg"].head(5).mean():+.2f}%')
except Exception as e:
    print(f'风格指数: {e}')

# 行业指数（申万一级）- 近5日涨幅
print('\n=== 申万一级行业近5日表现 ===')
shenwan_codes = {
    '801780.SI': '银行', '801760.SI': '传媒', '801750.SI': '计算机',
    '801740.SI': '军工', '801730.SI': '电力设备', '801720.SI': '建筑',
    '801710.SI': '建筑材料', '801080.SI': '电子', '801160.SI': '有色金属',
    '801010.SI': '农林牧渔', '801880.SI': '汽车', '801890.SI': '机械设备',
    '801030.SI': '化工', '801120.SI': '食品饮料', '801150.SI': '医药'
}
for code, name in shenwan_codes.items():
    try:
        df = pro.index_daily(ts_code=code, start_date='20260421', end_date='20260428')
        if df is not None and len(df) > 0:
            last5 = df.head(min(5, len(df)))['pct_chg'].mean()
            print(f'  {name:8s}: {last5:+.2f}%')
    except:
        pass

# 北向资金
print('\n=== 北向资金 ===')
try:
    df = pro.moneyflow_hsgt(start_date='20260428', end_date='20260428')
    if df is not None and len(df) > 0:
        r = df.iloc[0]
        print(f'今日: 沪股通 {float(r["gt_sh"])/1e8:.1f}亿, 深股通 {float(r["gt_sz"])/1e8:.1f}亿')
    else:
        print('今日数据未出')
except Exception as e:
    print(f'北向: {e}')

# 近期北向累计
try:
    df = pro.moneyflow_hsgt(start_date='20260421', end_date='20260428')
    if df is not None and len(df) > 0:
        total_sh = df['gt_sh'].astype(float).sum()
        total_sz = df['gt_sz'].astype(float).sum()
        print(f'近5日累计: 沪股通 {total_sh/1e8:.1f}亿, 深股通 {total_sz/1e8:.1f}亿')
except Exception as e:
    print(f'北向累计: {e}')

print()
print(datetime.now().strftime('生成时间: %Y-%m-%d %H:%M'))
