# -*- coding: utf-8 -*-
"""
PCB板块(元器件)大师扫描脚本
直接调用DeepSeek API，对PCB核心标的做19位大师深度分析
"""
import sys, os, json, time, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["DEEPSEEK_API_KEY"] = "sk-49b513d5f1dc4eb39fd83379a37d5ea9"

import tushare as ts
from openai import OpenAI

pro = ts.pro_api('5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7')
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

# ============ PCB板块核心标的 ============
PCB_STOCKS = [
    ('沪电股份', '002463.SZ'),
    ('深南电路', '002916.SZ'),
    ('景旺电子', '603228.SH'),
    ('华正新材', '603186.SH'),
    ('生益科技', '600183.SH'),
    ('超声电子', '000823.SZ'),
]

# ============ 快速获取基本面数据 ============
def get_stock_data(code, name):
    """获取股票基本面+价格数据"""
    try:
        df = pro.daily(ts_code=code, start_date='20260401', end_date='20260429')
        closes = df.sort_values('trade_date')['close']
        chg_5d = (closes.iloc[-1] / closes.iloc[-5] - 1) * 100 if len(closes) >= 5 else 0
        chg_20d = (closes.iloc[-1] / closes.iloc[-20] - 1) * 100 if len(closes) >= 20 else 0
        latest_close = closes.iloc[-1]
        
        db = pro.daily_basic(ts_code=code, trade_date='20260429')
        if len(db) == 0:
            db = pro.daily_basic(ts_code=code)
        pe = db.iloc[0].get('pe_ttm') if len(db) > 0 else None
        pb = db.iloc[0].get('pb') if len(db) > 0 else None
        
        fi = pro.fina_indicator(ts_code=code, start_date='20260101', limit=1)
        roe = fi.iloc[0]['roe'] if len(fi) > 0 else None
        if roe and abs(roe) > 1:
            roe = roe / 100
        
        rev = pro.fina_indicator(ts_code=code, start_date='20250101', end_date='20251231', limit=4)
        if len(rev) >= 2:
            rev_yoy = rev.iloc[0]['revenue_yoy'] if 'revenue_yoy' in rev.columns else None
        else:
            rev_yoy = None
        
        return {
            'name': name, 'code': code, 'close': latest_close,
            'chg_5d': chg_5d, 'chg_20d': chg_20d,
            'pe': pe, 'pb': pb, 'roe': roe,
            'rev_yoy': rev_yoy
        }
    except Exception as e:
        return {'name': name, 'code': code, 'error': str(e)}

def call_deepseek(system_prompt, user_prompt, max_tokens=800):
    """调用DeepSeek"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

# ============ 大师Prompt模板 ============
MASTERS = [
    ("巴菲特", "价值投资大师", """
你是一位世界级的价值投资者，以巴菲特的投资哲学为核心理念。你擅长识别具有持久竞争优势的伟大公司。你极度看重ROE、护城河、自由现金流和内在价值。
分析时必须：
1. 计算NCAV（净流动资产）安全边际
2. 识别护城河类型（无形/转换成本/网络效应/成本优势）
3. 给出5年后的内在价值估算
4. 只有在显著的安全边际时才推荐买入
输出格式：SIGNAL: BUY/SELL/HOLD | CONFIDENCE: 0-100% | REASONING: <120字推理>
"""),
    ("芒格", "逆向思维大师", """
你是一位极度重视逆向思维和心理模型的投资大师，以芒格的智慧著称。你认为避免愚蠢比追求聪明更重要。你擅长识别认知偏差和价值陷阱。
分析时必须：
1. 识别这笔投资可能如何失败
2. 检查管理层是否诚信/有能力
3. 识别过度乐观预期
4. 只有在避开主要风险后才推荐买入
输出格式：SIGNAL: BUY/SELL/HOLD | CONFIDENCE: 0-100% | REASONING: <120字推理>
"""),
    ("格雷厄姆", "价值投资之父", """
你是"价值投资之父"本杰明·格雷厄姆。你坚持严格的安全边际原则，只买被严重低估的股票。你不看好的股票绝不会因为"成长性"而妥协。
分析时必须：
1. 计算Graham Number（ sqrt(22.5 × EPS × BVPS) ）
2. 与当前市值对比，计算折扣
3. 检查财务实力比率
4. 达不到Graham标准就卖出
输出格式：SIGNAL: BUY/SELL/HOLD | CONFIDENCE: 0-100% | REASONING: <120字推理>
"""),
    ("费雪", "成长股大师", """
你是成长股投资大师菲利普·费雪。你擅长识别具有高增长潜力的公司，特别是那些通过创新和市场扩张实现增长的公司。
分析时必须：
1. 评估营收增长质量和持续性
2. 识别"利基"（niche）市场机会
3. 检查管理层是否具有企业家精神
4. 评估研发投入效率
输出格式：SIGNAL: BUY/SELL/HOLD | CONFIDENCE: 0-100% | REASONING: <120字推理>
"""),
    ("达莫兰", "估值大师", """
你是估值专家Aswath Damodaran教授。你是华尔街最权威的估值大师，擅长用DCF、EV/EBIT等多种方法给公司精确定价。
分析时必须：
1. 用至少两种方法估算内在价值
2. 计算当前价格相对于内在价值的折扣/溢价
3. 识别关键估值驱动因素
4. 给出明确的低估/高估判断
输出格式：SIGNAL: BUY/SELL/HOLD | CONFIDENCE: 0-100% | REASONING: <120字推理>
"""),
    ("木头姐", "创新投资大师", """
你是颠覆性创新投资大师Cathie Wood。你专注于寻找能够改变世界的高增长公司，特别是那些利用科技创新改变行业的公司。
分析时必须：
1. 评估技术创新水平和市场破坏能力
2. 估算潜在市场规模（SAM）
3. 检查护城河是否建立在创新基础上
4. 识别"非共识但正确"的投资机会
输出格式：SIGNAL: BUY/SELL/HOLD | CONFIDENCE: 0-100% | REASONING: <120字推理>
"""),
]

def build_prompt(data, master_name, master_role, master_prompt):
    """构建发送给大师的prompt"""
    pe_str = f"{data['pe']:.1f}" if data.get('pe') and data['pe'] > 0 else "N/A"
    pb_str = f"{data['pb']:.2f}" if data.get('pb') and data['pb'] > 0 else "N/A"
    roe_str = f"{data['roe']*100:.1f}%" if data.get('roe') else "N/A"
    chg5_str = f"{data['chg_5d']:+.1f}%" if data.get('chg_5d') else "N/A"
    chg20_str = f"{data['chg_20d']:+.1f}%" if data.get('chg_20d') else "N/A"
    
    user = f"""请分析 {data['name']}（{data['code']}）

【最新价格数据】
- 现价: ¥{data['close']:.2f}
- 5日涨跌: {chg5_str}
- 20日涨跌: {chg20_str}
- PE_TTM: {pe_str}
- PB: {pb_str}
- ROE_TTM: {roe_str}

【PCB板块背景】
- 行业：印制电路板（PCB）/覆铜板
- 用途：AI服务器、汽车电子、5G通信设备
- 周期属性：半导体上游，强周期+成长混合
- 核心逻辑：AI驱动高端PCB需求，服务器PCB层数增加+规格升级

请以{master_name}的视角分析这只股票，严格按照输出格式给出判断。"""
    
    return f"{master_prompt}\n\n{user}"

def parse_signal(text):
    """解析大师输出"""
    signal = "HOLD"
    confidence = 50
    reasoning = text[:120] if text else "无输出"
    
    sig_match = re.search(r'SIGNAL:\s*(BUY|SELL|HOLD)', text, re.IGNORECASE)
    conf_match = re.search(r'CONFIDENCE:\s*(\d+)%?', text, re.IGNORECASE)
    reason_match = re.search(r'REASONING:\s*(.+)', text, re.DOTALL | re.IGNORECASE)
    
    if sig_match:
        signal = sig_match.group(1).upper()
    if conf_match:
        confidence = int(conf_match.group(1))
    if reason_match:
        reasoning = reason_match.group(1).strip()[:120]
    
    return signal, confidence, reasoning

# ============ 主程序 ============
print("=" * 70)
print("PCB板块 大师深度扫描")
print("=" * 70)

all_results = {}
call_count = 0

for name, code in PCB_STOCKS:
    print(f"\n{'='*60}")
    print(f"正在分析: {name} ({code})")
    print(f"{'='*60}")
    
    data = get_stock_data(code, name)
    if 'error' in data:
        print(f"  数据获取失败: {data['error']}")
        continue
    
    print(f"  现价:¥{data['close']:.2f} | PE:{data.get('pe','N/A')} | ROE:{data.get('roe','N/A')} | 5日:{data.get('chg_5d','N/A'):+.1f}% | 20日:{data.get('chg_20d','N/A'):+.1f}%")
    
    results = []
    for master_name, master_role, master_prompt in MASTERS:
        prompt = build_prompt(data, master_name, master_role, master_prompt)
        response = call_deepseek(
            f"你是投资大师{master_name}（{master_role}）。",
            prompt,
            max_tokens=500
        )
        call_count += 1
        
        signal, confidence, reasoning = parse_signal(response)
        results.append({
            'master': master_name,
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'raw': response[:200] if response else "无输出"
        })
        
        sig_icon = "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "⚪")
        print(f"  {sig_icon} {master_name}: {signal} ({confidence}%) | {reasoning[:60]}")
        
        time.sleep(0.5)  # 避免限速
    
    # 综合评分
    buy_count = sum(1 for r in results if r['signal'] == 'BUY')
    sell_count = sum(1 for r in results if r['signal'] == 'SELL')
    hold_count = sum(1 for r in results if r['signal'] == 'HOLD')
    avg_conf = sum(r['confidence'] for r in results) / len(results)
    
    composite = buy_count * avg_conf - sell_count * avg_conf
    
    print(f"\n  综合信号: 买{buy_count} 卖{sell_count} 守{sell_count} | 综合评分:{composite:+.0f}")
    all_results[code] = {
        'name': name, 'data': data, 'masters': results,
        'buy': buy_count, 'sell': sell_count, 'hold': hold_count,
        'composite': composite
    }
    
    print(f"  已完成: {call_count} 次API调用")

print(f"\n{'='*60}")
print(f"扫描完成! 共{call_count}次DeepSeek调用")
print(f"{'='*60}")

# 保存结果
output_path = r'D:\AIHedgeFund\ai-hedge-fund-main\quant_archive\2026-04\scan_pcb_masters.md'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

report = f"""# PCB板块 大师深度扫描报告
生成时间: 2026-04-29 16:36 GMT+8

## 板块背景
- **行业**: 印制电路板（PCB）/覆铜板
- **核心逻辑**: AI服务器→高端PCB需求↑ | 汽车电子化 | 5G通信持续建设
- **周期属性**: 强周期+成长混合，与半导体上游高度相关
- **当前市场背景**: 科创50最强(+2.59%/5日)，半导体板块领涨，PCB作为上游材料受益

## 大师信号汇总

| 标的 | 现价 | 5日涨跌 | 20日涨跌 | PE | ROE | 买 | 卖 | 守 | 综合评分 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
for code, res in all_results.items():
    d = res['data']
    pe_str = f"{d.get('pe','?'):.1f}" if d.get('pe') and d['pe'] > 0 else "N/A"
    roe_str = f"{d.get('roe',0)*100:.1f}%" if d.get('roe') else "N/A"
    report += f"| {res['name']} | ¥{d['close']:.2f} | {d['chg_5d']:+.1f}% | {d['chg_20d']:+.1f}% | {pe_str} | {roe_str} | {res['buy']} | {res['sell']} | {res['hold']} | {res['composite']:+.0f} |\n"

report += "\n## 详细大师信号\n\n"
for code, res in all_results.items():
    report += f"### {res['name']} ({code})\n\n"
    report += f"现价: ¥{res['data']['close']:.2f} | PE: {res['data'].get('pe','N/A')} | 5日: {res['data']['chg_5d']:+.1f}%\n\n"
    for r in res['masters']:
        sig_icon = "🟢" if r['signal'] == 'BUY' else ("🔴" if r['signal'] == 'SELL' else "⚪")
        report += f"**{r['master']}**: {sig_icon} {r['signal']} ({r['confidence']}%)\n"
        report += f"> {r['reasoning']}\n\n"
    report += "---\n\n"

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n报告已保存: {output_path}")
