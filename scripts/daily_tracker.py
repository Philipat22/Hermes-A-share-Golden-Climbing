"""
黄金坑策略每日追踪报告
用法: python daily_tracker.py
依赖: prices_full.pkl 需先运行 update_prices_today.py 更新
"""
import pickle, pandas as pd, numpy as np, warnings, time, urllib.request
from datetime import datetime, date
warnings.filterwarnings('ignore')

# ====== 配置 ======
PRICES_FILE = r"D:\AIHedgeFund\ai-hedge-fund-main\data\cache\prices_full.pkl"
CSI300_FILE = r"D:\AIHedgeFund\ai-hedge-fund-main\data\cache\csi300.pkl"
SIG_CACHE = r"D:\AIHedgeFund\ai-hedge-fund-main\data\cache\golden_pit_signals_all.pkl"

HOLDINGS = {
    '603906.SH': ('龙蟠科技', '电气设备', 30.16, '2026-05-12'),
    '603507.SH': ('振江股份', '电气设备', 44.82, '2026-05-12'),
    '000070.SZ': ('特发信息', '通信设备', 21.42, '2026-05-12'),
    '603026.SH': ('石大胜华', '化工原料', 104.04, '2026-05-12'),
}
EXIT_DATE = date(2026, 7, 6)  # 40个交易日 (5/12 + 40 trading days)

# 板块排除 (历史胜率<70%, 时间切分+分年验证)
EXCLUDE_SECTORS = {'银行','港口','证券','钢加工','旅游景点','火力发电','出版业','供气供热'}

MONITOR_SECTORS = ['半导体', '黄金']

# ====== 加载数据 ======
print("加载数据...")
with open(PRICES_FILE, 'rb') as f:
    prices = pickle.load(f)

# 数据新鲜度
sample_dates = []
for code in list(prices.keys())[:100]:
    df = prices[code]
    if hasattr(df, 'columns'):
        d = str(df['trade_date'].max()) if len(df) > 0 else ''
        if d: sample_dates.append(d)
latest_date = max(set(sample_dates), key=sample_dates.count) if sample_dates else '?'

# CSI300
csi = pd.read_pickle(CSI300_FILE)
csi = csi.sort_values('trade_date').reset_index(drop=True)
csi['ma60'] = csi['close'].rolling(60).mean()
csi_close = csi['close'].iloc[-1]
csi_ma60 = csi['ma60'].iloc[-1]
gate = 'OPEN' if csi_close > csi_ma60 else 'CLOSED'

# 板块信息
try:
    import tushare as ts
    pro = ts.pro_api('6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7')
    df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    sector_map = dict(zip(df_basic['ts_code'], df_basic['industry']))
    name_map = dict(zip(df_basic['ts_code'], df_basic['name']))
except:
    sector_map = {}; name_map = {}

# 历史信号路径参考
try:
    with open(SIG_CACHE, 'rb') as f:
        old_sig = pickle.load(f)
    old_sig['sector'] = old_sig['code'].map(sector_map)
    has_history = True
except:
    has_history = False

today = date.today()
days_held = (today - date(2026,5,12)).days
days_left = (EXIT_DATE - today).days

# ====== 报告 ======
print(f"\n{'='*60}")
print(f"  黄金坑策略 · 每日追踪")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}")

# 闸门
print(f"\n闸门: CSI300={csi_close:.0f} vs MA60={csi_ma60:.0f} → {gate}  |  数据: {latest_date}")
print(f"持仓第{days_held}天  |  到期日: {EXIT_DATE}  |  剩余{days_left}天")

# ====== 持仓跟踪 ======
print(f"\n{'─'*60}")
print(f"【持仓跟踪】")
print(f"{'─'*60}")

total_cost = 0; total_now = 0
for code, (name, sec, cost, entry_date) in HOLDINGS.items():
    if code not in prices: continue
    df = prices[code].sort_values('trade_date').reset_index(drop=True)
    c = df['close'].values; n = len(df); i = n-1
    price = c[i]
    pnl = (price/cost - 1)*100
    total_cost += cost; total_now += price
    
    # Trend check
    ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    score = sum([ma20[i]>ma60[i], ma60[i]>pd.Series(c).rolling(120).mean().values[i],
                 i>=5 and ma60[i]>ma60[i-5], c[i]>ma20[i], c[i]>ma60[i]])
    
    # Depth
    lb = min(20, i); pk = i-lb+np.argmax(c[i-lb:i+1])
    dd = (c[i]/c[pk]-1)*100
    
    # Status
    icon = '🔴' if pnl < -5 else ('🟡' if pnl < 0 else '🟢')
    status = f"趋势{score}/5 回撤{dd:+.1f}%"
    
    print(f"  {icon} {name:<6} ¥{price:>7.2f}  {pnl:>+5.1f}%  {status}")

total_pnl = (total_now/total_cost - 1)*100
print(f"  {'─'*45}")
print(f"  合计  ¥{total_now:>7.2f}  {total_pnl:>+5.1f}%  |  ¥{total_cost:.0f}→¥{total_now:.0f}")

# ====== 信号扫描 ======
print(f"\n{'─'*60}")
print(f"【今日信号】  (扫描中...)")
print(f"{'─'*60}")

results = []
for code, df_raw in prices.items():
    if not hasattr(df_raw, 'columns'): continue
    if '.BJ' in code or code.startswith('688'): continue
    if 'ST' in name_map.get(code, ''): continue
    
    df = df_raw.sort_values('trade_date').reset_index(drop=True)
    if len(df) < 120: continue
    c = df['close'].values; v = df['vol'].values; n = len(df); idx = n-1
    ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    ma120 = pd.Series(c).rolling(120).mean().values
    if np.isnan(ma60[idx]): continue
    score = sum([ma20[idx]>ma60[idx], ma60[idx]>ma120[idx], idx>=5 and ma60[idx]>ma60[idx-5],
                 c[idx]>ma20[idx], c[idx]>ma60[idx]])
    if score < 4: continue
    lb = min(20, idx); pk = idx-lb+np.argmax(c[idx-lb:idx+1])
    dd = (c[idx]/c[pk]-1)*100
    if dd > -10: continue
    spd = abs(dd)/max(idx-pk,1)
    if spd <= 0.5: continue
    vm = pd.Series(v).rolling(20).mean().values
    vr = v[idx]/vm[idx] if vm[idx]>0 else 99
    if vr >= 3: continue
    ret60 = (c[idx]/c[idx-60]-1)*100; ret120 = (c[idx]/c[idx-120]-1)*100
    if ret120 <= abs(dd)*2: continue
    if sector_map.get(code,'其他') in EXCLUDE_SECTORS: continue
    results.append({'code':code,'name':name_map.get(code,'?'),'sector':sector_map.get(code,'其他'),
        'close':c[idx],'trend':score,'speed':spd,'dd':dd,'ret60':ret60,'vol_ratio':vr})

# 排序: 回撤深度优先(跌越深越前) → 趋势分辅助
sig = pd.DataFrame(results).sort_values(['dd','trend'], ascending=[True,False])
print(f"  信号总数: {len(sig)}个\n")

# 集中板块
from collections import Counter
sector_cnt = Counter(s['sector'] for _,s in sig.iterrows())
for sec, cnt in sector_cnt.most_common(8):
    sub = sig[sig['sector']==sec]
    top = sub.iloc[0]
    held_marks = [name_map.get(c,'?') for c in HOLDINGS if HOLDINGS[c][1]==sec]
    held_str = f"  [{', '.join(held_marks)}]" if held_marks else ""
    print(f"  {sec:<8} {cnt}个  最强: {top['name']} 趋{int(top['trend'])} {top['speed']:.1f}%/d 深{top['dd']:+.1f}%{held_str}")

# ====== 历史对比 ======
if has_history:
    print(f"\n{'─'*60}")
    print(f"【持仓 vs 历史路径】")
    print(f"{'─'*60}")
    
    for code, (name, sec, cost, entry_date) in HOLDINGS.items():
        if code not in prices: continue
        df = prices[code].sort_values('trade_date').reset_index(drop=True)
        price = df['close'].iloc[-1]
        pnl = (price/cost - 1)*100
        
        # Sector history
        sec_sig = old_sig[(old_sig['sector']==sec) & (old_sig['trend']>=4) & (old_sig['dd']<=-10) & (old_sig['dd']>-12)]
        if len(sec_sig) > 10:
            sec_mean = sec_sig['r40'].mean()
            sec_win = (sec_sig['r40']>0).mean()*100
            sec_n = len(sec_sig)
            
            # Where does current PnL stand?
            # Use r5/r10 as proxy for early-stage path
            pnl_pctile = (pnl > sec_sig['r40']).mean()*100 if pnl < 0 else 100
            
            signal = '✓ 正常' if pnl > -8 else ('⚠️ 偏深' if pnl > -15 else '🔴 异常')
            print(f"  {name:<6} {pnl:>+5.1f}%  |  {sec}历史: {sec_n}笔 均值{sec_mean:+.1f}% 胜率{sec_win:.0f}%  |  {signal}")

# ====== 监控板块 ======
print(f"\n{'─'*60}")
print(f"【监控板块】")
print(f"{'─'*60}")

for mon_sec in MONITOR_SECTORS:
    mon_codes = [c for c, ind in sector_map.items() if ind == mon_sec]
    scores = []
    for code in mon_codes[:50]:  # sample
        if code not in prices: continue
        df = prices[code].sort_values('trade_date').reset_index(drop=True)
        if len(df) < 120: continue
        c = df['close'].values; n = len(df); i = n-1
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        ma120 = pd.Series(c).rolling(120).mean().values
        if np.isnan(ma60[i]): continue
        s = sum([ma20[i]>ma60[i], ma60[i]>ma120[i], i>=5 and ma60[i]>ma60[i-5], c[i]>ma20[i], c[i]>ma60[i]])
        scores.append(s)
    
    if scores:
        avg = np.mean(scores)
        high = sum(1 for s in scores if s >= 4)
        low = sum(1 for s in scores if s <= 2)
        trend_word = '强' if avg >= 4 else ('中' if avg >= 3 else '弱')
        print(f"  {mon_sec:<8} {len(scores)}只  均{avg:.1f}/5  ≥4分:{high}只  ≤2分:{low}只  → {trend_word}")

# ====== 底部 ======
print(f"\n{'='*60}")
print(f"  下次到期: {EXIT_DATE}  剩余{days_left}天")
print(f"  更新数据: python update_prices_today.py")
print(f"  重新扫盘: python daily_tracker.py")
print(f"{'='*60}")
