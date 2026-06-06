"""
A股宏观状态监控器 v1.2
分层: 流动性五层 + 实体定价三层
  流动性: 央行→银行→外资→散户→M1M2
  实体定价: 铜油比(工业需求) + 金铜比(风险偏好) + ERP(估值锚)
频率: 北向(日)/SHIBOR(日)/融资(日)/社融(月)/M1M2(月)/期货(日)/ERP(日)
用法:
    from macro_monitor import MacroMonitor
    m = MacroMonitor()
    state = m.get_state()  # 当前四层流动性状态
    print(state['summary'])  # 人类可读摘要
"""
import akshare as ak, pandas as pd, numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle, json, warnings
warnings.filterwarnings('ignore')


class MacroMonitor:
    """
    A股四层流动性监控。
    每次调用 get_state() 自动检查数据新鲜度，过期则重新拉取。
    
    数据新鲜度要求:
      北向资金: 1个交易日
      SHIBOR: 1个交易日
      融资余额: 1个交易日  
      社融: 30天（月度发布）
      M1/M2: 30天（月度发布）
    """
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = Path.home() / '.qclaw' / 'workspace-agent-b9c8dcea' / 'data' / 'macro'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._state = None
    
    # ---- 实体定价层 (v1.2) ----
    
    def get_gold_copper(self):
        """金铜比 (AU0/CU0) - 风险偏好反向指标 (后120天r=+0.501, diff=16.8%)"""
        if self._is_fresh('gold_copper', 24):
            return self._cache_get('gold_copper')
        try:
            au = ak.futures_main_sina(symbol='AU0')
            cu = ak.futures_main_sina(symbol='CU0')
            if au is not None and cu is not None:
                au = au.iloc[:, [0, 1]].copy(); au.columns = ['date', 'price']
                au['date'] = pd.to_datetime(au['date']); au = au.set_index('date')['price']
                cu = cu.iloc[:, [0, 1]].copy(); cu.columns = ['date', 'price']
                cu['date'] = pd.to_datetime(cu['date']); cu = cu.set_index('date')['price']
                merged = pd.DataFrame({'au': au, 'cu': cu}).dropna()
                merged['ratio'] = merged['au'] / merged['cu']
                merged['pct_1y'] = merged['ratio'].rolling(252, min_periods=100).rank(pct=True)
                latest = merged.iloc[-1]
                s = '极端恐慌-买入' if latest['pct_1y'] > 0.7 else ('极端贪婪-警惕' if latest['pct_1y'] < 0.3 else '正常')
                result = {'ratio': float(latest['ratio']), 'pct_1y': float(latest['pct_1y']),
                          'date': merged.index[-1].strftime('%Y-%m-%d'), 'signal': s}
                self._cache_set('gold_copper', result)
                return result
        except Exception as e:
            return {'error': str(e)}
        return {'error': '金铜比获取失败'}
    
    def get_cu_oil(self):
        """铜油比 (CU0/SC0) - 工业需求vs能源成本 (后120天r=+0.172, diff=12.2%)"""
        if self._is_fresh('cu_oil', 24):
            return self._cache_get('cu_oil')
        try:
            cu = ak.futures_main_sina(symbol='CU0')
            oil = ak.futures_main_sina(symbol='SC0')
            if cu is not None and oil is not None:
                cu = cu.iloc[:, [0, 1]].copy(); cu.columns = ['date', 'price']
                cu['date'] = pd.to_datetime(cu['date']); cu = cu.set_index('date')['price']
                oil = oil.iloc[:, [0, 1]].copy(); oil.columns = ['date', 'price']
                oil['date'] = pd.to_datetime(oil['date']); oil = oil.set_index('date')['price']
                merged = pd.DataFrame({'cu': cu, 'oil': oil}).dropna()
                merged['ratio'] = merged['cu'] / merged['oil']
                merged['pct_1y'] = merged['ratio'].rolling(252, min_periods=100).rank(pct=True)
                latest = merged.iloc[-1]
                s = '强劲-做多' if latest['pct_1y'] > 0.7 else ('走弱-防御' if latest['pct_1y'] < 0.3 else '正常')
                result = {'ratio': float(latest['ratio']), 'pct_1y': float(latest['pct_1y']),
                          'date': merged.index[-1].strftime('%Y-%m-%d'), 'signal': s}
                self._cache_set('cu_oil', result)
                return result
        except Exception as e:
            return {'error': str(e)}
        return {'error': '铜油比获取失败'}
    
    def get_erp(self):
        """股债性价比 ERP = 1/PE - SHIBOR (后120天r=+0.398, diff=9.9%)"""
        if self._is_fresh('erp', 24):
            return self._cache_get('erp')
        try:
            import tushare as ts
            pro = ts.pro_api('6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7')
            pe = pro.index_dailybasic(ts_code='000300.SH', start_date='20240101',
                                       end_date=datetime.now().strftime('%Y%m%d'), fields='trade_date,pe')
            if pe is not None and len(pe) > 0:
                pe['trade_date'] = pd.to_datetime(pe['trade_date'])
                pe = pe.set_index('trade_date')['pe'].sort_index()
                shibor = ak.macro_china_shibor_all()
                shibor = shibor.iloc[:, [0, 1]].copy(); shibor.columns = ['date', 'rate']
                shibor['date'] = pd.to_datetime(shibor['date'])
                shibor = shibor.set_index('date')['rate'].resample('D').ffill()
                merged = pd.DataFrame({'erp_raw': 1/pe * 100, 'shibor': shibor}).dropna()
                merged['erp'] = merged['erp_raw'] - merged['shibor']
                merged['pct_1y'] = merged['erp'].rolling(252, min_periods=100).rank(pct=True)
                latest = merged.iloc[-1]
                s = '极度便宜-重仓' if latest['pct_1y'] > 0.7 else ('偏贵-减仓' if latest['pct_1y'] < 0.3 else '正常')
                result = {'erp': float(latest['erp']), 'pct_1y': float(latest['pct_1y']),
                          'date': merged.index[-1].strftime('%Y-%m-%d'), 'signal': s}
                self._cache_set('erp', result)
                return result
        except Exception as e:
            return {'error': str(e)}
        return {'error': 'ERP获取失败'}
    
    # ================================================================
    # 数据获取
    # ================================================================
    
    def _is_fresh(self, key: str, max_age_hours: int) -> bool:
        cache_file = self.cache_dir / f'{key}.pkl'
        if not cache_file.exists():
            return False
        age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime))
        return age.total_seconds() / 3600 < max_age_hours
    
    def _cache_get(self, key: str):
        cache_file = self.cache_dir / f'{key}.pkl'
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _cache_set(self, key: str, data):
        with open(self.cache_dir / f'{key}.pkl', 'wb') as f:
            pickle.dump(data, f)
    
    # ---- 第一层: 央行的钱 ----
    def get_shibor(self):
        """SHIBOR隔夜利率 - 银行间流动性"""
        if self._is_fresh('shibor', 24):
            return self._cache_get('shibor')
        try:
            df = ak.macro_china_shibor_all()
            if df is not None and len(df) > 0:
                # 列0=日期, 列1=隔夜
                df = df.iloc[:, [0, 1]].copy()
                df.columns = ['date', 'rate']
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna(subset=['rate']).sort_values('date')
                latest = df.iloc[-1]
                avg_20d = df['rate'].tail(20).mean()
                result = {
                    'current': float(latest['rate']),
                    'date': latest['date'].strftime('%Y-%m-%d'),
                    'trend_30d': '下降' if latest['rate'] < avg_20d else '上升',
                    'level': '充裕' if latest['rate'] < 1.5 else ('正常' if latest['rate'] < 2.5 else '偏紧')
                }
                self._cache_set('shibor', result)
                return result
        except Exception as e:
            return {'error': str(e)}
    
    # ---- 第二层: 银行的钱 (社融) ----
    def get_social_financing(self):
        """社融增量 - 实体经济信贷"""
        if self._is_fresh('sf', 720):  # 月度，30天
            return self._cache_get('sf')
        try:
            df = ak.macro_china_shrzgm()
            df = df.iloc[:, [0, 1]].copy()
            df.columns = ['month', 'sf_flow']
            df['date'] = pd.to_datetime(df['month'].astype(str) + '01', format='%Y%m%d')
            df = df.sort_values('date')
            df['sf_12m'] = df['sf_flow'].rolling(12).sum()
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            result = {
                'current_flow': float(latest['sf_flow']),
                'current_12m': float(latest['sf_12m']),
                'month': latest['date'].strftime('%Y-%m'),
                'mom_change': '增加' if latest['sf_flow'] > prev['sf_flow'] else '减少',
                'level': '扩张' if latest['sf_12m'] > df['sf_12m'].iloc[-13:-1].mean() else '收缩'
            }
            self._cache_set('sf', result)
            return result
        except Exception as e:
            return {'error': str(e)}
    
    # ---- 第三层: 外资的钱 (北向) ----
    def get_northbound(self):
        """北向资金 - 外资情绪 (Tushare)"""
        if self._is_fresh('northbound', 24):
            return self._cache_get('northbound')
        try:
            import tushare as ts
            pro = ts.pro_api('6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7')
            df = pro.moneyflow_hsgt(
                start_date=(datetime.now()-timedelta(days=30)).strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d')
            )
            if df is not None and len(df) > 0:
                # north_money = 北向资金日净流入(万元) - 字符串需转float
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df['north_money'] = df['north_money'].astype(float)
                
                today_flow = float(df['north_money'].iloc[-1]) / 10000  # 万元→亿元
                flow_5d = float(df['north_money'].tail(5).sum()) / 10000
                flow_20d = float(df['north_money'].tail(20).sum()) / 10000
                
                result = {
                    'today_flow': float(today_flow),
                    'flow_5d': float(flow_5d),
                    'flow_20d': float(flow_20d),
                    'date': df['trade_date'].iloc[-1].strftime('%Y-%m-%d'),
                    'signal': '流入' if flow_5d > 0 else '流出',
                    'level': '强力流入' if flow_5d > 100 else ('持续流出' if flow_5d < -100 else '正常')
                }
                self._cache_set('northbound', result)
                return result
        except Exception as e:
            return {'error': str(e)}
        return {'error': '北向数据获取失败'}
    
    def get_margin(self):
        """融资余额 - 散户杠杆情绪 (Tushare)"""
        if self._is_fresh('margin', 24):
            return self._cache_get('margin')
        try:
            import tushare as ts
            pro = ts.pro_api('6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7')
            # Tushare margin接口
            df = pro.margin(start_date=(datetime.now()-timedelta(days=30)).strftime('%Y%m%d'),
                           end_date=datetime.now().strftime('%Y%m%d'))
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                latest_date = df['trade_date'].iloc[-1]
                # 沪深两市合计
                latest_data = df[df['trade_date'] == latest_date]
                total_margin = float(latest_data['rzye'].sum()) / 1e8  # 元→亿
                
                # 趋势判断: 和5天前比
                prev_date = df['trade_date'].unique()[-6] if len(df['trade_date'].unique()) >= 6 else df['trade_date'].iloc[0]
                prev_data = df[df['trade_date'] == prev_date]
                prev_margin = float(prev_data['rzye'].sum()) / 1e8 if len(prev_data) > 0 else total_margin
                
                result = {
                    'margin_balance': total_margin,
                    'date': latest_date.strftime('%Y-%m-%d'),
                    'trend': '上升' if total_margin > prev_margin else '下降',
                }
                self._cache_set('margin', result)
                return result
        except Exception as e:
            return {'error': str(e)}
        return {'error': '融资数据获取失败'}
    
    # ---- M1-M2 (洪灏框架) ----
    def get_m1m2(self):
        """M1-M2剪刀差 - 企业活期化程度"""
        if self._is_fresh('m1m2', 720):
            return self._cache_get('m1m2')
        try:
            import tushare as ts
            pro = ts.pro_api('6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7')
            df = pro.cn_m(start_date='20240101', end_date=datetime.now().strftime('%Y%m%d'))
            if df is not None and len(df) > 0:
                # month格式: '202604' → 需要特殊处理
                df = df.copy()
                df['month_str'] = df['month'].astype(str)
                df = df[df['month_str'].str.match(r'^\d{6}$')]  # 只保留有效月份
                df = df.sort_values('month_str')
                latest = df.iloc[-1]
                m1_yoy = float(latest['m1_yoy'])
                m2_yoy = float(latest['m2_yoy'])
                scissors = m1_yoy - m2_yoy
                result = {
                    'm1_yoy': m1_yoy,
                    'm2_yoy': m2_yoy,
                    'scissors': scissors,
                    'month': latest['month_str'],
                    'signal': '活化' if scissors > 0 else '窖藏'
                }
                self._cache_set('m1m2', result)
                return result
        except Exception as e:
            return {'error': str(e)}
        return {'error': 'M1/M2数据获取失败'}
    
    # ================================================================
    # 综合判断
    # ================================================================
    
    def get_state(self) -> dict:
        """获取当前宏观状态摘要"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'layers': {},
            'signals': {'positive': 0, 'negative': 0, 'neutral': 0},
            'summary': '',
            'action': ''
        }
        
        # 第一层: 央行
        shibor = self.get_shibor()
        state['layers']['central_bank'] = shibor
        if 'error' not in shibor:
            if shibor['level'] in ('充裕', '正常'):
                state['signals']['positive'] += 1
            else:
                state['signals']['negative'] += 1
        
        # 第二层: 银行
        sf = self.get_social_financing()
        state['layers']['bank_credit'] = sf
        if 'error' not in sf:
            if sf['level'] == '扩张':
                state['signals']['positive'] += 1
            else:
                state['signals']['negative'] += 1
        
        # 第三层: 外资
        nb = self.get_northbound()
        state['layers']['foreign'] = nb
        if 'error' not in nb:
            if nb['signal'] == '流入':
                state['signals']['positive'] += 1
            else:
                state['signals']['negative'] += 1
        
        # 第四层: 散户
        margin = self.get_margin()
        state['layers']['retail'] = margin
        if 'error' not in margin and 'margin_balance' in margin:
            if margin.get('trend') == '上升':
                state['signals']['positive'] += 1
            else:
                state['signals']['neutral'] += 1
        
        # 附加: M1M2剪刀差
        m1m2 = self.get_m1m2()
        state['layers']['m1m2'] = m1m2
        if 'error' not in m1m2 and 'signal' in m1m2:
            if m1m2['signal'] == '活化':
                state['signals']['positive'] += 1
            else:
                state['signals']['negative'] += 1
        
        # Layer 6: 铜油比 - 工业需求
        cu_oil = self.get_cu_oil()
        state['layers']['cu_oil'] = cu_oil
        if 'error' not in cu_oil and cu_oil['pct_1y'] > 0.5:
            state['signals']['positive'] += 1
        elif 'error' not in cu_oil:
            state['signals']['negative'] += 1
        
        # Layer 7: 金铜比 - 风险偏好 (反向加权x2)
        gold_cu = self.get_gold_copper()
        state['layers']['gold_copper'] = gold_cu
        if 'error' not in gold_cu:
            if gold_cu['pct_1y'] > 0.7:
                state['signals']['positive'] += 2
            elif gold_cu['pct_1y'] < 0.3:
                state['signals']['negative'] += 2
        
        # Layer 8: ERP - 估值锚
        erp = self.get_erp()
        state['layers']['erp'] = erp
        if 'error' not in erp and erp['pct_1y'] > 0.5:
            state['signals']['positive'] += 1
        elif 'error' not in erp:
            state['signals']['negative'] += 1
        
        # 综合判断
        pos = state['signals']['positive']
        neg = state['signals']['negative']
        net = pos - neg
        
        if net >= 2:
            state['action'] = 'AGGRESSIVE'
            state['summary'] = f'流动性全面宽松({pos}正{neg}负): 钱在央行→银行→外资→A股全线流动。重仓窗口。'
        elif net == 1:
            state['action'] = 'MODERATE'
            state['summary'] = f'流动性偏松({pos}正{neg}负): 部分层级在流动，选择性参与。'
        elif net == 0:
            state['action'] = 'CAUTIOUS'
            state['summary'] = f'流动性中性({pos}正{neg}负): 水没放出来，维持仓位等待。'
        elif net == -1:
            state['action'] = 'DEFENSIVE'
            state['summary'] = f'流动性偏紧({pos}正{neg}负): 至少一层在收缩，减仓。'
        else:
            state['action'] = 'BEARISH'
            state['summary'] = f'流动性全面紧缩({pos}正{neg}负): 钱在退出A股，清仓防御。'
        
        self._state = state
        return state
    
    def print_state(self):
        """打印人类可读的宏观状态"""
        state = self.get_state()
        print(f"\n{'='*60}")
        print(f"A股宏观流动性状态  {state['timestamp'][:19]}")
        print(f"{'='*60}")
        for layer, data in state['layers'].items():
            layer_names = {'central_bank':'央行(银行间)', 'bank_credit':'银行(实体)', 
                          'foreign':'外资(A股)', 'retail':'散户(杠杆)', 'm1m2':'M1-M2剪刀差',
                          'cu_oil':'铜油比(工业)', 'gold_copper':'金铜比(情绪)', 'erp':'股债性价比(估值)'}
            if 'error' in data:
                print(f"  {layer_names.get(layer, layer):12s}: 数据不可用")
            else:
                print(f"  {layer_names.get(layer, layer):12s}: {json.dumps(data, ensure_ascii=False, default=str)}")
        print(f"\n  综合: {state['summary']}")
        print(f"  建议: {state['action']}")
        print(f"{'='*60}\n")


# ================================================================
# CLI
# ================================================================
if __name__ == '__main__':
    m = MacroMonitor()
    m.print_state()
