"""
19位大师 × 4只持仓股 全维度分析
"""
import os, json, sys, requests
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import tushare as ts
pro = ts.pro_api('5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7')

# ============ Data Functions ============
def get_financial_summary(ticker):
    try:
        # 基本信息
        info = pro.stock_basic(ts_code=ticker, fields='name,industry,market,pe,pb')
        name = info.iloc[0]['name'] if len(info) > 0 else ticker
        
        # 价格数据
        prices = pro.daily(ts_code=ticker, start_date='20260401', end_date='20260429')
        
        # 财务指标
        fin_df = pro.fina_indicator(ts_code=ticker, start_date='20250101', fields='ts_code,roe,roic,debt_to_equity,eps_yoy,tr_yoy,netprofit_yoy,gross_profit_margin,net_profit_margin')
        
        # 估值
        daily_basic = pro.daily_basic(ts_code=ticker, start_date='20260428', end_date='20260429')
        
        result = {'ticker': ticker, 'name': name}
        
        if len(prices) >= 1:
            p0 = prices.iloc[0]
            result['price'] = p0['close']
            result['pct_5d'] = round((p0['close'] - prices.iloc[min(4,len(prices)-1)]['close']) / prices.iloc[min(4,len(prices)-1)]['close'] * 100, 2) if len(prices) >= 5 else 0
            result['pct_20d'] = round((p0['close'] - prices.iloc[min(19,len(prices)-1)]['close']) / prices.iloc[min(19,len(prices)-1)]['close'] * 100, 2) if len(prices) >= 20 else 0
            result['vol_5d_avg'] = round(prices.head(5)['vol'].mean(), 0)
        
        if len(fin_df) >= 1:
            f = fin_df.iloc[0]
            for k in ['roe', 'roic', 'eps_yoy', 'tr_yoy', 'netprofit_yoy', 'gross_profit_margin', 'net_profit_margin']:
                v = f.get(k)
                if v is not None and abs(v) > 100:
                    v = v / 100
                result[k] = round(v, 2) if v is not None else None
        
        if len(daily_basic) >= 1:
            db = daily_basic.iloc[0]
            result['pe'] = round(db.get('pe', 0), 2) if db.get('pe') else None
            result['pb'] = round(db.get('pb', 0), 2) if db.get('pb') else None
            result['market_cap'] = round(db.get('total_market_cap', 0) / 10000, 2) if db.get('total_market_cap') else None
        
        return result
    except Exception as e:
        return {'ticker': ticker, 'error': str(e)}

# ============ Master Prompts ============
SYSTEM_PROMPTS = {
    ' Buffett': '你是Warren Buffett风格价值投资大师。极度关注护城河（品牌/定价权/网络效应/成本优势）、长期ROE稳定性和资本配置效率。只买ROE>15%、PE<30、有宽护城河的公司。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Munger': '你是Charlie Munger风格逆向投资大师。极度挑剔，擅长发现公司致命缺陷（多元化折价、激励机制失效、行业内卷、客户集中度过高）。只认可"极好+极便宜"的机会。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Graham': '你是Benjamin Graham式价值投资之父。严格遵守NCAV（净流动资产）安全边际理念。只在PE<20且PB<2时买入，极端情况下才重仓。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Fisher': '你是Philip Fisher式成长股投资大师。关注公司的技术创新能力、管理层质量和"沉默知识"（内部人持股、研发投入强度）。5-10年视角。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Lynch': '你是Peter Lynch式投资大师。"投资你了解的领域"——关注业务可理解性、PEG<1、管理层诚信、门店/客户扩张。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Dalas': '你是达莫达兰式估值大师。使用DCF、SOTP、EV/EBIT给公司估值。关注成长率假设的合理性（拒绝过度乐观）。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Wood': '你是Cathie Wood式成长投资大师。专注颠覆性创新（AI/新能源/生物科技），用5年+的长期视野评估，对短期估值容忍度高。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Cramer': '你是Jim Cramer式实战派。技术面+消息面+板块联动，短期动量交易为主，关注支撑/阻力位。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Burry': '你是Michael Burry式逆向大师。识别市场共识错误，宏观择时能力强，对泡沫极度敏感，愿意左侧交易。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Howie': '你是Howie价值投资专家。纯财务驱动：ROE>10%、PE<40、无商誉减值、现金流健康。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Psych': '你是市场情绪专家。识别贪婪/恐惧、机构持仓集中度、分析师共识情绪、散户行为偏差。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Quant': '你是量化分析师。多因子打分：价值/成长/动量/质量因子均衡评估，寻找统计套利机会。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
    ' Macro': '你是宏观策略师。结合全球经济周期、政策导向、汇率对A股出口影响、新能源补贴政策、人民币汇率评估行业。输出JSON：{"signal":"BUY/SELL/HOLD","confidence":0-100,"target_price":"元","stop_loss":"元","reasoning":"60字","holding_period":"短期/中期/长期","risk_level":"低/中/高"}',
}

MASTERS = list(SYSTEM_PROMPTS.keys())

def ask_master(master_name, system, fin_data):
    api_key = os.environ.get('DEEPSEEK_API_KEY', '')
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get('DEEPSEEK_API_KEY', '')
    
    fin_str = json.dumps(fin_data, ensure_ascii=False)
    
    user_prompt = f"""分析 {fin_data.get('name','')} ({fin_data.get('ticker','')})

数据摘要: 价格={fin_data.get('price')} | PE={fin_data.get('pe')} | PB={fin_data.get('pb')} | ROE={fin_data.get('roe')} | 营收增长={fin_data.get('tr_yoy')} | 利润增长={fin_data.get('netprofit_yoy')} | 5日涨跌={fin_data.get('pct_5d')}% | 20日涨跌={fin_data.get('pct_20d')}%

市场背景: 
- 当前A股主线: 半导体/AI/机器人
- 锂电/新能源: 产能过剩+价格战，等待需求拐点
- 宏观: 稳增长政策持续，新能源汽车补贴政策不明朗

请严格输出JSON格式结论："""

    try:
        resp = requests.post(
            'https://api.deepseek.com/chat/completions',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json={'model': 'deepseek-chat', 'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user_prompt}
            ], 'max_tokens': 350, 'temperature': 0.3},
            timeout=35
        )
        
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            # Extract JSON
            json_str = content
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                json_str = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(json_str.strip())
            result['master'] = master_name.strip()
            return result
        else:
            return {'master': master_name.strip(), 'signal': 'ERROR', 'confidence': 0, 'reasoning': f'API {resp.status_code}'}
    except json.JSONDecodeError as e:
        return {'master': master_name.strip(), 'signal': 'HOLD', 'confidence': 40, 'reasoning': f'JSON error: {str(e)[:50]}'}
    except Exception as e:
        return {'master': master_name.strip(), 'signal': 'ERROR', 'confidence': 0, 'reasoning': str(e)[:80]}

def main():
    print('=' * 70)
    print('  19-Master Portfolio Analysis  (13 Masters x 4 Positions)')
    print('=' * 70)
    
    TICKERS = [
        ('002245.SZ', '蔚蓝锂芯'),
        ('002074.SZ', '国轩高科'),
        ('603906.SH', '龙蟠科技'),
        ('601016.SH', '节能环境'),  # 华电辽能候选
    ]
    
    all_results = {}
    
    for ticker, stock_name in TICKERS:
        print(f'\n>>> {stock_name} ({ticker}) ...')
        fin_data = get_financial_summary(ticker)
        print(f'    Price={fin_data.get("price")} PE={fin_data.get("pe")} ROE={fin_data.get("roe")} 5d={fin_data.get("pct_5d")}% 20d={fin_data.get("pct_20d")}%')
        
        results = []
        for master_name, system in SYSTEM_PROMPTS.items():
            r = ask_master(master_name, system, fin_data)
            results.append(r)
            sig = r.get('signal', '?')
            conf = r.get('confidence', 0)
            print(f'    [{master_name.strip():8s}] {sig} (conf={conf}%)')
        
        all_results[f'{stock_name}({ticker})'] = results
    
    # Summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    
    for ticker_key, results in all_results.items():
        buy = [r for r in results if r.get('signal') == 'BUY']
        sell = [r for r in results if r.get('signal') == 'SELL']
        hold = [r for r in results if r.get('signal') == 'HOLD']
        err = [r for r in results if r.get('signal') == 'ERROR']
        
        valid = [r for r in results if r.get('signal') != 'ERROR']
        avg_conf = sum(r.get('confidence', 0) for r in valid) / max(len(valid), 1)
        
        print(f'\n{ticker_key}:')
        print(f'  BUY={len(buy)}  SELL={len(sell)}  HOLD={len(hold)}  ERR={len(err)}  AvgConf={avg_conf:.0f}%')
        
        if buy:
            for r in buy:
                print(f'  BUY [{r["confidence"]}%]: {r.get("reasoning","")[:60]}')
        if sell:
            for r in sell:
                print(f'  SELL [{r["confidence"]}%]: {r.get("reasoning","")[:60]}')
        if hold:
            for r in hold[:3]:
                print(f'  HOLD [{r["confidence"]}%]: {r.get("reasoning","")[:60]}')
    
    # Overall
    total_buy = sum(len([r for r in v if r.get('signal')=='BUY']) for v in all_results.values())
    total_sell = sum(len([r for r in v if r.get('signal')=='SELL']) for v in all_results.values())
    total_hold = sum(len([r for r in v if r.get('signal')=='HOLD']) for v in all_results.values())
    
    print(f'\nOVERALL: BUY={total_buy} | SELL={total_sell} | HOLD={total_hold}')
    
    if total_sell >= total_buy + 3:
        print('>> ROTATE RECOMMENDED: Reduce lithium/energy and rotate to semiconductors')
    elif total_buy > total_sell:
        print('>> HOLD: Fundamentals supported')
    else:
        print('>> MIXED: Hold with caution, monitor semiconductor rotation')
    
    # Save report
    out_path = r'D:\AIHedgeFund\ai-hedge-fund-main\quant_archive\2026-04\portfolio_19masters_analysis.md'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    lines = ['# Portfolio 19-Master Analysis Report\n', f'**Generated**: 2026-04-30 09:16\n\n']
    
    for ticker_key, results in all_results.items():
        buy = [r for r in results if r.get('signal') == 'BUY']
        sell = [r for r in results if r.get('signal') == 'SELL']
        hold = [r for r in results if r.get('signal') == 'HOLD']
        valid = [r for r in results if r.get('signal') not in ('ERROR',)]
        avg_conf = sum(r.get('confidence', 0) for r in valid) / max(len(valid), 1)
        
        lines.append(f'## {ticker_key}\n')
        lines.append(f'| Signal | Count | Avg Conf |\n|---|---|---|\n')
        lines.append(f'| BUY | {len(buy)} | {sum(r["confidence"] for r in buy)/max(len(buy),1):.0f}% |\n')
        lines.append(f'| SELL | {len(sell)} | {sum(r["confidence"] for r in sell)/max(len(sell),1):.0f}% |\n')
        lines.append(f'| HOLD | {len(hold)} | {sum(r["confidence"] for r in hold)/max(len(hold),1):.0f}% |\n\n')
        
        lines.append('### BUY Signals\n')
        for r in buy:
            lines.append(f'- **{r["master"]}** [conf={r["confidence"]}%]: {r.get("reasoning","")[:100]}\n')
            if r.get('target_price'):
                lines.append(f'  -> Target: {r["target_price"]} | Stop: {r.get("stop_loss","")} | Period: {r.get("holding_period","")} | Risk: {r.get("risk_level","")}\n')
        
        lines.append('\n### SELL Signals\n')
        for r in sell:
            lines.append(f'- **{r["master"]}** [conf={r["confidence"]}%]: {r.get("reasoning","")[:100]}\n')
            if r.get('target_price'):
                lines.append(f'  -> Target: {r["target_price"]} | Stop: {r.get("stop_loss","")} | Period: {r.get("holding_period","")} | Risk: {r.get("risk_level","")}\n')
        
        lines.append('\n---\n')
    
    lines.append(f'\n## Overall\n')
    lines.append(f'| Direction | Count |\n|---|---|\n')
    lines.append(f'| BUY | {total_buy} |\n| SELL | {total_sell} |\n| HOLD | {total_hold} |\n\n')
    
    if total_sell >= total_buy + 3:
        lines.append('**Recommendation**: ROTATE OUT of lithium/energy stocks. Consider moving allocation to semiconductors (韦尔股份/北方华创).\n')
    elif total_buy > total_sell:
        lines.append('**Recommendation**: HOLD current positions. Fundamentals are supported.\n')
    else:
        lines.append('**Recommendation**: MIXED signals. Hold with caution, monitor semiconductor rotation opportunity.\n')
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))
    
    print(f'\nReport saved: {out_path}')

if __name__ == '__main__':
    main()