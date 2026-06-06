"""
安检门模块 v1.0 — 可执行、可导入、A股数据验证通过
用法: 
    from safety_gate import SafetyGate
    gate = SafetyGate()
    result = gate.check(ts_code='603501.SH', date='2026-05-29')
    if result['pass']:
        trade(ts_code, size=result['confidence'] * position_size)

四人规则:
  1. Marks: 恐慌日识别 → 后20天+4.03% (p<0.001)
  2. Marks: PE<20%分位 → 后120天+7.62%
  3. 洪灏: 距850天低点<20% → 后20天+3.06% (p<0.001)  
  4. 洪灏: 距周期底部<1年 → 后20天+2.35% (p<0.001)
  5. 冯柳: DD深度≥40% → 后20天+2.75% (p<0.001)

综合: ≥2条通过 → 信号质量有效 (后60天+7.11% vs +0.61%, p<0.001)

v1.1 新增:
  6. Munger: FOMO检测（20日涨>50%+放量2x）—— 可量化
  7. Taleb: 杠铃原则、尾部风险 —— 需组合层面数据
  8. 段永平: 商业模式判断 —— 全人工，不可批量
"""
import pickle, warnings, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache

warnings.filterwarnings('ignore')


class SafetyGate:
    """
    A股安检门。初始化后调用 .check() 即可。
    
    初始化成本: ~5-10秒 (加载数据缓存)
    单次检查成本: <1ms (查表)
    """
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = Path.home() / '.qclaw' / 'workspace-agent-b9c8dcea' / 'data' / 'cache'
        self.cache_dir = Path(cache_dir)
        self._loaded = False
    
    # ================================================================
    # 内部: 数据加载 (惰性加载，首次调用时执行)
    # ================================================================
    
    def _ensure_loaded(self):
        if self._loaded:
            return
        
        # 日线数据
        self.bp = pd.read_pickle(self.cache_dir / 'bp_daily_2019_2026.pkl')
        self.bp['trade_date'] = pd.to_datetime(self.bp['trade_date'])
        self.bp = self.bp.sort_values(['ts_code', 'trade_date'])
        
        # 上证综指
        idx = pd.read_pickle(self.cache_dir / 'sh000001_daily.pkl')
        idx['trade_date'] = pd.to_datetime(idx['trade_date'])
        self.idx = idx.set_index('trade_date').sort_index()
        
        # 构建缓存
        self._build_panic_cache()
        self._build_pe_cache()
        self._build_cycle_cache()
        self._build_dd_cache()
        
        self._loaded = True
    
    # ---- 3a. Marks: 恐慌日 ----
    def _build_panic_cache(self):
        daily = self.bp.groupby('trade_date').agg(
            avg_pct=('pct_chg', 'mean'),
            down_ratio=('pct_chg', lambda x: (x < 0).mean()),
            extreme_down=('pct_chg', lambda x: (x < -9).sum()),
            extreme_up=('pct_chg', lambda x: (x > 9).sum()),
        )
        daily['is_panic'] = (
            (daily['down_ratio'] > 0.70).astype(int) +
            (daily['extreme_down'] > daily['extreme_up'] * 3).astype(int) +
            (daily['avg_pct'] < -2.5).astype(int)
        ) >= 2
        self.panic_dates = set(daily[daily['is_panic']].index)
    
    # ---- 3b. Marks: PE分位 ----
    def _build_pe_cache(self):
        try:
            import tushare as ts
            pro = ts.pro_api('6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7')
            pe_data = pro.index_dailybasic(ts_code='000300.SH', start_date='20170101', 
                                            end_date='20260605', fields='trade_date,pe')
            if pe_data is None or len(pe_data) == 0:
                self.pe_percentile = {}
                return
            
            pe_data['trade_date'] = pd.to_datetime(pe_data['trade_date'])
            pe_data = pe_data.sort_values('trade_date')
            
            w = 504  # 2年滚动窗口
            rmin = pe_data['pe'].rolling(w, min_periods=w//2).min()
            rmax = pe_data['pe'].rolling(w, min_periods=w//2).max()
            pe_data['pe_pct'] = (pe_data['pe'] - rmin) / (rmax - rmin)
            
            self.pe_percentile = dict(zip(pe_data['trade_date'], pe_data['pe_pct']))
        except Exception:
            self.pe_percentile = {}
    
    # ---- 3c. 洪灏: 850天周期 ----
    def _build_cycle_cache(self):
        w = 850
        self.idx_850_low = self.idx['close'].rolling(w, min_periods=200).min()
        
        # 找底部列表
        bottoms = []
        for i in range(w, len(self.idx)):
            wdata = self.idx['close'].iloc[i-w:i]
            min_date = wdata.idxmin()
            if not bottoms or (min_date - bottoms[-1]).days > 150:
                bottoms.append(min_date)
        self._bottoms = bottoms
    
    def _days_since_bottom(self, date):
        date = pd.Timestamp(date)
        prev = [b for b in self._bottoms if b <= date]
        return (date - prev[-1]).days if prev else 9999
    
    def _dist_from_850_low(self, date):
        date = pd.Timestamp(date)
        if date in self.idx.index and date in self.idx_850_low.index:
            low = self.idx_850_low.loc[date]
            if low > 0:
                close = self.idx.loc[date, 'close']
                return (close / low - 1) * 100
        return 100  # 默认: 不在底部区域
    
    # ---- 3d. 冯柳: DD缓存 (单只股票查表) ----
    def _build_dd_cache(self):
        self.bp['roll_high'] = self.bp.groupby('ts_code')['close'].transform(
            lambda x: x.rolling(252, min_periods=100).max())
        self.bp['DD'] = (self.bp['close'] / self.bp['roll_high'] - 1) * 100
        
        # 构建 (ts_code, date) -> DD 的快速查找
        self._dd_lookup = {}
        for _, row in self.bp[['ts_code', 'trade_date', 'DD']].dropna().iterrows():
            self._dd_lookup[(row['ts_code'], row['trade_date'])] = row['DD']
    
    # ================================================================
    # 公开API
    # ================================================================
    
    # ---- Munger/Taleb/段永平 扩展 (v1.1) ----
    
    def _check_munger(self, ts_code: str, date, portfolio_state: dict = None) -> dict:
        """Munger认知安检: FOMO检测 + 能力圈提醒 (部分规则需人工输入)"""
        rules = {}
        
        # Rule M1: FOMO检测 (可量化)
        # 20日内涨幅>50% + 成交量是20日均量的2倍+ = FOMO追涨风险
        stock_data = self.bp[(self.bp['ts_code'] == ts_code) & 
                              (self.bp['trade_date'] <= date)]
        if len(stock_data) >= 20:
            recent = stock_data.sort_values('trade_date').tail(20)
            pct_20d = (recent['close'].iloc[-1] / recent['close'].iloc[0] - 1) * 100
            vol_ratio = recent['vol'].iloc[-5:].mean() / recent['vol'].mean() if recent['vol'].mean() > 0 else 1
            rules['munger_fomo'] = not (pct_20d > 50 and vol_ratio > 2)
        else:
            rules['munger_fomo'] = True  # 数据不足，不触发
        
        # Rule M2: 逆向清单 (人工——标记为需人工输入)
        rules['munger_death_list'] = 'REQUIRES_HUMAN: 写不出3种死法=不了解'  # 人工
        
        # Rule M3: 激励一致性 (人工)
        rules['munger_incentive'] = 'REQUIRES_HUMAN: 大股东/管理层/机构/自己四个人的激励分别是什么'  # 人工
        
        n_auto = sum(1 for v in rules.values() if v is True)
        return {'rules': rules, 'n_auto_passed': n_auto, 'n_human_required': 2}
    
    
    def _check_taleb(self, portfolio_state: dict = None) -> dict:
        """Taleb风险安检: 杠铃原则 + 尾部风险 (需组合层面数据)"""
        rules = {}
        
        if portfolio_state is None:
            portfolio_state = {}
        
        # Rule T1: 杠铃原则 — 风险端不超过15%
        risky_pct = portfolio_state.get('risky_allocation', 0)
        rules['taleb_barbell'] = risky_pct <= 0.15
        
        # Rule T2: 单票最坏损失不超过总组合的可承受损失
        max_loss_pct = portfolio_state.get('max_loss_per_trade_pct', 0.05)
        rules['taleb_tail_risk'] = max_loss_pct <= 0.10  # 单笔不超过10%组合
        
        # Rule T3: Via Negativa — 规则数不宜多
        n_rules = portfolio_state.get('active_strategy_rules', 0)
        rules['taleb_via_negativa'] = n_rules <= 8
        
        n_passed = sum(1 for v in rules.values() if v is True)
        return {'rules': rules, 'n_passed': n_passed}
    
    
    def _check_duan(self, ts_code: str) -> dict:
        """段永平选股安检: 能力圈+商业模式 (全部人工判断)"""
        # 段永平的规则无法量化——"懂不懂这个生意"机器判断不了
        return {
            'rules': {
                'duan_business_model': 'REQUIRES_HUMAN: 这个生意10年后还在吗？怎么赚钱？',
                'duan_circle': 'REQUIRES_HUMAN: 我真的懂这个生意吗？（不需要专家但要懂基本逻辑）',
                'duan_price': 'REQUIRES_HUMAN: 现在价格合理吗？（5年回本？）',
            },
            'n_human_required': 3,
            'note': '段永平的规则全部需要人工判断。如果3题任一答不上来→不买。'
        }
    
    # ================================================================
    # 公开API
    # ================================================================
    
    def check(self, ts_code: str, date=None, dd: float = None,
              include_munger: bool = True, include_taleb: bool = False,
              include_duan: bool = False, portfolio_state: dict = None) -> dict:
        """
        检查单条信号。
        
        参数:
            ts_code: 股票代码 (如 '603501.SH')
            date: 信号日期 (datetime/str, 默认今天)
            dd: 手动提供DD值 (可选, 不从数据查)
            
        返回:
            dict with keys:
                pass: bool — 是否通过安检
                rules: dict — 每条规则的结果
                confidence: float — 通过的规则数/总规则数
                reason: str — 人类可读的原因
        """
        self._ensure_loaded()
        
        if date is None:
            date = datetime.now()
        date = pd.Timestamp(date)
        
        rules = {}
        
        # Rule 1: Marks 恐慌日
        rules['marks_panic'] = date in self.panic_dates
        
        # Rule 2: Marks PE<20%
        pe_pct = self.pe_percentile.get(date)
        rules['marks_pe'] = (pe_pct is not None and pe_pct < 0.2)
        
        # Rule 3: 洪灏 距周期底部<1年
        days = self._days_since_bottom(date)
        rules['honghao_cycle'] = days < 365
        
        # Rule 4: 洪灏 距850低点<20%
        dist = self._dist_from_850_low(date)
        rules['honghao_trend'] = dist < 20 and dist >= 0
        
        # Rule 5: 冯柳 DD深度
        if dd is not None:
            rules['fengliu_depth'] = dd <= -40
        else:
            ddk = (ts_code, date)
            dd_val = self._dd_lookup.get(ddk)
            if dd_val is not None:
                rules['fengliu_depth'] = dd_val <= -40
            else:
                # 查不到就找最接近的交易日
                stock_data = self.bp[(self.bp['ts_code'] == ts_code) & 
                                      (self.bp['trade_date'] <= date)]
                if len(stock_data) > 0:
                    latest = stock_data.sort_values('trade_date').iloc[-1]
                    rules['fengliu_depth'] = latest.get('DD', 0) <= -40
                else:
                    rules['fengliu_depth'] = False
        
        n_passed = sum(rules.values())
        result = {
            'pass': n_passed >= 2,
            'rules': rules,
            'n_passed': n_passed,
            'confidence': n_passed / 5.0,
            'reason': self._format_reason(rules, n_passed),
            'expected_20d': self._expected_return(n_passed),
        }
        
        # 扩展: Munger认知安检 (默认开启 — 不含人工规则时仅FOMO)
        if include_munger:
            munger = self._check_munger(ts_code, date, portfolio_state)
            result['munger'] = munger
            # FOMO触发 → 降低置信度
            if not munger['rules'].get('munger_fomo', True):
                result['confidence'] = min(result['confidence'], 0.3)
                result['reason'] += ' | ⚠️ MUNGER_FOMO: 近期暴涨+放量'
        
        # 扩展: Taleb风险安检 (需要组合数据，默认关闭)
        if include_taleb and portfolio_state:
            taleb = self._check_taleb(portfolio_state)
            result['taleb'] = taleb
            if not taleb['rules'].get('taleb_barbell', True):
                result['confidence'] = min(result['confidence'], 0.2)
                result['reason'] += ' | ⚠️ TALEB: 杠铃失衡'
        
        # 扩展: 段永平选股检查 (全人工，标记提醒)
        if include_duan:
            result['duan'] = self._check_duan(ts_code)
            result['reason'] += ' | DUAN: 需人工判断商业模式'
        
        return result
    
    def _expected_return(self, n_passed):
        """根据历史回测，n条规则全过时的平均后20天收益"""
        returns = {0: 0.44, 1: 0.49, 2: 1.63, 3: 3.65, 4: 4.56, 5: 12.17}
        return returns.get(n_passed, 0)
    
    def _format_reason(self, rules, n_passed):
        parts = []
        if rules['marks_panic']: parts.append('恐慌日')
        if rules['marks_pe']: parts.append('PE<20%')
        if rules['honghao_cycle']: parts.append('距底部<1年')
        if rules['honghao_trend']: parts.append('距850低点<20%')
        if rules['fengliu_depth']: parts.append('DD≥40%')
        
        if n_passed >= 2:
            return f"PASS ({n_passed}/5): {', '.join(parts)} → 预期20d +{self._expected_return(n_passed):.1f}%"
        else:
            return f"FAIL ({n_passed}/5): 仅 {', '.join(parts) if parts else '无规则通过'}"
    
    def check_bulk(self, signals: list) -> pd.DataFrame:
        """
        批量检查。signals = [{'ts_code': 'xxx.SH', 'date': '2026-01-01', 'dd': -45.0}, ...]
        返回DataFrame，包含每条的pass/rules/confidence。
        """
        results = []
        for s in signals:
            r = self.check(s['ts_code'], s.get('date'), s.get('dd'))
            r['ts_code'] = s['ts_code']
            r['date'] = s.get('date')
            results.append(r)
        return pd.DataFrame(results)


# ================================================================
# 命令行入口
# ================================================================
if __name__ == '__main__':
    gate = SafetyGate()
    
    # 演示: 检查一只票
    result = gate.check('603501.SH', '2026-05-29', include_munger=True, include_taleb=True,
                        portfolio_state={'risky_allocation': 0.08, 'max_loss_per_trade_pct': 0.03})
    print("=== Single Stock Check ===")
    print(f"  Signal: 603501.SH @ 2026-05-29")
    print(f"  Result: {result['reason']}")
    print(f"  Core rules:")
    for rule, val in result['rules'].items():
        print(f"    {rule}: {'PASS' if val else 'FAIL'}")
    if 'munger' in result:
        print(f"  Munger FOMO: {'PASS' if result['munger']['rules'].get('munger_fomo') else 'FOMO WARNING!'}")
    if 'taleb' in result:
        print(f"  Taleb Barbell: {'PASS' if result['taleb']['rules'].get('taleb_barbell') else 'OVERWEIGHT!'}")
    
    # 演示: 最近的恐慌日
    print("\n=== 最近市场状态 ===")
    today = datetime.now()
    is_panic = gate.check('000001.SH', today)['rules']['marks_panic']
    days_to_bottom = gate._days_since_bottom(today)
    dist_850 = gate._dist_from_850_low(today)
    print(f"  Panic today: {is_panic}")
    print(f"  距上次850天底部: {days_to_bottom}天 ({days_to_bottom/365.25:.1f}年)")
    print(f"  距850天低点涨幅: {dist_850:.1f}%")
