#!/usr/bin/env python3
"""市场状态分类器

基于CSI300指数判定当前市场处于什么状态：
  - BULL:  多头趋势，动量向上
  - BEAR:  空头趋势，动量向下
  - SIDEWAYS: 横盘震荡
  - VOLATILE: 高波动（可叠加在其他状态上）

用法:
  from src.surge.regime_classifier import RegimeClassifier
  rc = RegimeClassifier()
  regime = rc.classify(date='2026-04-29')
  # -> {'regime': 'BULL', 'confidence': 0.85, ...}
"""
import os, sys
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, '.env'))


class RegimeClassifier:
    """CSI300 市场状态分类器"""

    def __init__(self):
        self._csi300: Optional[pd.DataFrame] = None

    # ══════════════════════════════════════════════════════
    # 数据获取
    # ══════════════════════════════════════════════════════

    def _fetch_csi300(self, start_date: str = '20180101') -> pd.DataFrame:
        """从 Tushare 获取 CSI300 日线数据"""
        import tushare as ts
        token = os.getenv('TUSHARE_PRO_TOKEN', '')
        if not token:
            raise RuntimeError("TUSHARE_PRO_TOKEN 未设置，请在 .env 中配置或设置环境变量")
        pro = ts.pro_api(token)

        df = pro.index_daily(
            ts_code='000300.SH',
            start_date=start_date,
            end_date=datetime.now().strftime('%Y%m%d'),
            fields='trade_date,open,high,low,close,volume,pct_chg'
        )
        if df is not None and len(df) > 0:
            df = df.sort_values('trade_date').reset_index(drop=True)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            return df
        return pd.DataFrame()

    @property
    def csi300(self) -> pd.DataFrame:
        if self._csi300 is None:
            self._csi300 = self._fetch_csi300()
        return self._csi300

    def _load_cached(self, cache_path: str = 'data/cache/csi300.pkl') -> pd.DataFrame:
        """加载缓存的 CSI300 数据"""
        full_path = os.path.join(ROOT, cache_path)
        if os.path.exists(full_path):
            self._csi300 = pd.read_pickle(full_path)
        else:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            self._csi300 = self._fetch_csi300()
            self._csi300.to_pickle(full_path)
        return self._csi300

    # ══════════════════════════════════════════════════════
    # 特征计算
    # ══════════════════════════════════════════════════════

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指数上的技术指标"""
        df = df.copy().sort_values('trade_date').reset_index(drop=True)
        closes = df['close'].values

        # 均线
        df['ma20'] = pd.Series(closes).rolling(20).mean().values
        df['ma50'] = pd.Series(closes).rolling(50).mean().values
        df['ma200'] = pd.Series(closes).rolling(200).mean().values

        # 动量
        df['ret_5'] = pd.Series(closes).pct_change(5).values
        df['ret_10'] = pd.Series(closes).pct_change(10).values
        df['ret_20'] = pd.Series(closes).pct_change(20).values
        df['ret_60'] = pd.Series(closes).pct_change(60).values

        # 波动率 (20日年化)
        daily_ret = pd.Series(closes).pct_change()
        df['vol_20'] = daily_ret.rolling(20).std().values * np.sqrt(252)

        # 关键技术条件
        # 1. 趋势方向: 收盘价相对 MA200 的位置
        with np.errstate(invalid='ignore'):
            df['price_vs_ma200'] = (closes - df['ma200'].values) / df['ma200'].values

        # 2. 均线排列
        df['ma5_above_ma20'] = (pd.Series(closes).rolling(5).mean().values >
                                df['ma20'].values).astype(int)
        df['ma20_above_ma50'] = (df['ma20'].values > df['ma50'].values).astype(int)
        df['ma50_above_ma200'] = (df['ma50'].values > df['ma200'].values).astype(int)

        # 3. 均线距离
        with np.errstate(invalid='ignore'):
            df['ma_50_200_spread'] = (df['ma50'].values - df['ma200'].values) / df['ma200'].values
            df['ma_20_50_spread'] = (df['ma20'].values - df['ma50'].values) / df['ma50'].values

        # 4. ADX 趋势强度 (简化版)
        # 用 +DI 和 -DI 的比值作为趋势强度
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1)),
        }).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean().values
        df['atr_pct'] = (df['atr_14'] / closes * 100).values

        return df

    # ══════════════════════════════════════════════════════
    # 核心分类逻辑
    # ══════════════════════════════════════════════════════

    def classify(self,
                 date: Optional[str] = None,
                 lookback_days: int = 250,
                 return_details: bool = True) -> dict:
        """分类当前(或指定日期)的市场状态

        参数:
          date: 目标日期 (YYYY-MM-DD 或 YYYYMMDD), 默认最新数据
          lookback_days: 需要的历史天数
          return_details: 是否返回详细特征

        返回:
          {'regime': str, 
           'regime_id': int,     # 0=BULL, 1=BEAR, 2=SIDEWAYS
           'confidence': float,   # 置信度 0-1
           'features': {...}}     # (optional) 全部特征
        """
        df = self._load_cached()

        if len(df) < 200:
            return {'regime': 'SIDEWAYS', 'regime_id': 2,
                    'confidence': 0.5, 'error': 'insufficient_data'}

        df = self._compute_features(df)

        if date:
            try:
                target = pd.to_datetime(date)
            except ValueError:
                target = pd.to_datetime(date, format='%Y%m%d')
            # 找到最接近的交易日
            mask = df['trade_date'] <= target
            if mask.sum() == 0:
                row = df.iloc[-1]
            else:
                row = df[mask].iloc[-1]
        else:
            row = df.iloc[-1]

        features = row.to_dict()
        for k, v in features.items():
            if isinstance(v, (np.floating,)):
                features[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, (np.integer,)):
                features[k] = int(v)

        # ── 分类规则 ──
        # 读取关键特征
        price_vs_ma200 = features.get('price_vs_ma200') or 0
        ma_50_200_spread = features.get('ma_50_200_spread') or 0
        ret_20 = features.get('ret_20') or 0
        ret_60 = features.get('ret_60') or 0
        ma50_above_ma200 = features.get('ma50_above_ma200', 0)
        vol_20 = features.get('vol_20') or 0.2

        # 评分体系 (各因素加权)
        bull_score = 0.0
        bear_score = 0.0

        # 趋势方向 (最大 ±0.30)
        if ma50_above_ma200 == 1:
            bull_score += 0.15
        else:
            bear_score += 0.15

        # 价格 vs MA200 (最大 ±0.25)
        if price_vs_ma200 > 0.10:
            bull_score += 0.25
        elif price_vs_ma200 > 0.05:
            bull_score += 0.15
        elif price_vs_ma200 < -0.10:
            bear_score += 0.25
        elif price_vs_ma200 < -0.05:
            bear_score += 0.15

        # 均线距离 MA50-MA200 (最大 ±0.20)
        if ma_50_200_spread > 0.08:
            bull_score += 0.20
        elif ma_50_200_spread > 0.03:
            bull_score += 0.10
        elif ma_50_200_spread < -0.08:
            bear_score += 0.20
        elif ma_50_200_spread < -0.03:
            bear_score += 0.10

        # 短期动量 (最大 ±0.25)
        if ret_20 > 0.05:
            bull_score += 0.25
        elif ret_20 > 0.02:
            bull_score += 0.10
        elif ret_20 < -0.05:
            bear_score += 0.25
        elif ret_20 < -0.02:
            bear_score += 0.10

        # 中期动量 (最大 ±0.15)
        if ret_60 > 0.10:
            bull_score += 0.15
        elif ret_60 > 0.05:
            bull_score += 0.08
        elif ret_60 < -0.10:
            bear_score += 0.15
        elif ret_60 < -0.05:
            bear_score += 0.08

        # 波动率修正 (高波动 = 降低所有分数)
        vol_factor = max(0.5, min(1.0, 1.0 - (vol_20 - 0.15)))
        bull_score *= vol_factor
        bear_score *= vol_factor

        total = bull_score + bear_score
        if total == 0:
            return {'regime': 'SIDEWAYS', 'regime_id': 2,
                    'confidence': 0.5, **({'features': features} if return_details else {})}

        # 判定
        bull_ratio = bull_score / total if total > 0 else 0.5
        confidence = abs(bull_ratio - 0.5) * 2  # 0~1

        if bull_ratio > 0.65:
            regime = 'BULL'
            regime_id = 0
        elif bull_ratio < 0.35:
            regime = 'BEAR'
            regime_id = 1
        else:
            regime = 'SIDEWAYS'
            regime_id = 2

        result = {
            'regime': regime,
            'regime_id': regime_id,
            'confidence': round(confidence, 4),
            'bull_score': round(bull_score, 4),
            'bear_score': round(bear_score, 4),
            'bull_ratio': round(bull_ratio, 4),
        }

        if return_details:
            result['features'] = features

        return result

    # ══════════════════════════════════════════════════════
    # 批量分类 (用于回测)
    # ══════════════════════════════════════════════════════

    def classify_bulk(self) -> pd.DataFrame:
        """对全部历史数据做逐日分类

        返回:
          DataFrame: trade_date, regime, regime_id, confidence
        """
        df = self._load_cached()
        if len(df) < 200:
            return pd.DataFrame()

        df = self._compute_features(df)

        # 对每一行分类
        regimes = []
        for i in range(len(df)):
            row = df.iloc[i]
            features = row.to_dict()

            price_vs_ma200 = features.get('price_vs_ma200') or 0
            ma_50_200_spread = features.get('ma_50_200_spread') or 0
            ret_20 = features.get('ret_20') or 0
            ret_60 = features.get('ret_60') or 0
            ma50_above_ma200 = features.get('ma50_above_ma200', 0)
            vol_20 = features.get('vol_20') or 0.2

            bull_score, bear_score = 0.0, 0.0

            if ma50_above_ma200 == 1:
                bull_score += 0.15
            else:
                bear_score += 0.15

            if price_vs_ma200 > 0.10:
                bull_score += 0.25
            elif price_vs_ma200 > 0.05:
                bull_score += 0.15
            elif price_vs_ma200 < -0.10:
                bear_score += 0.25
            elif price_vs_ma200 < -0.05:
                bear_score += 0.15

            if ma_50_200_spread > 0.08:
                bull_score += 0.20
            elif ma_50_200_spread > 0.03:
                bull_score += 0.10
            elif ma_50_200_spread < -0.08:
                bear_score += 0.20
            elif ma_50_200_spread < -0.03:
                bear_score += 0.10

            if ret_20 > 0.05:
                bull_score += 0.25
            elif ret_20 > 0.02:
                bull_score += 0.10
            elif ret_20 < -0.05:
                bear_score += 0.25
            elif ret_20 < -0.02:
                bear_score += 0.10

            if ret_60 > 0.10:
                bull_score += 0.15
            elif ret_60 > 0.05:
                bull_score += 0.08
            elif ret_60 < -0.10:
                bear_score += 0.15
            elif ret_60 < -0.05:
                bear_score += 0.08

            vol_factor = max(0.5, min(1.0, 1.0 - (vol_20 - 0.15)))
            bull_score *= vol_factor
            bear_score *= vol_factor

            total = bull_score + bear_score
            if total == 0:
                regimes.append((features.get('trade_date'), 'SIDEWAYS', 2, 0.5))
                continue

            bull_ratio = bull_score / total
            confidence = abs(bull_ratio - 0.5) * 2

            if bull_ratio > 0.65:
                regime, rid = 'BULL', 0
            elif bull_ratio < 0.35:
                regime, rid = 'BEAR', 1
            else:
                regime, rid = 'SIDEWAYS', 2

            regimes.append((features.get('trade_date'), regime, rid, round(confidence, 4)))

        result = pd.DataFrame(regimes, columns=['trade_date', 'regime', 'regime_id', 'confidence'])
        result = result.dropna(subset=['trade_date'])
        return result


# ══════════════════════════════════════════════════════
# 快速测试
# ══════════════════════════════════════════════════════

if __name__ == '__main__':
    import json

    rc = RegimeClassifier()

    # 当前状态
    current = rc.classify(return_details=True)
    print('Current Regime:')
    print(json.dumps({k: v for k, v in current.items() if k != 'features'},
                     indent=2, ensure_ascii=False))
    if 'features' in current:
        f = current['features']
        print(f'  CSI300 Close: {f.get("close"):.0f}')
        print(f'  MA50: {f.get("ma50"):.0f}  MA200: {f.get("ma200"):.0f}')
        print(f'  20d Ret: {f.get("ret_20")*100:.1f}%  60d Ret: {f.get("ret_60")*100:.1f}%')
        print(f'  20d Vol: {f.get("vol_20")*100:.1f}%')

    # 历史状态分布
    bulk = rc.classify_bulk()
    if len(bulk) > 0:
        print(f'\nHistorical Regime Distribution ({len(bulk)} days):')
        dist = bulk['regime'].value_counts()
        for r in ['BULL', 'BEAR', 'SIDEWAYS']:
            cnt = dist.get(r, 0)
            print(f'  {r}: {cnt} ({cnt/len(bulk)*100:.1f}%)')

        # 按年份
        bulk['year'] = pd.to_datetime(bulk['trade_date']).dt.year
        yr_dist = bulk.groupby('year')['regime'].value_counts().unstack(fill_value=0)
        print(f'\n  By Year:')
        print(f'  {"Year":>6} | {"BULL":>6} | {"BEAR":>6} | {"SIDE":>6}')
        for yr, row in yr_dist.iterrows():
            if yr < 2019:
                continue
            total = row.sum()
            print(f'  {int(yr):>6} | {row.get("BULL",0):>5} ({row.get("BULL",0)/total*100:.0f}%)'
                  f' | {row.get("BEAR",0):>5} ({row.get("BEAR",0)/total*100:.0f}%)'
                  f' | {row.get("SIDEWAYS",0):>5} ({row.get("SIDEWAYS",0)/total*100:.0f}%)')
