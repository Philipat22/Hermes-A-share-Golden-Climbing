#!/usr/bin/env python3
"""
每日主升浪信号生成器 v5.2 — 数据驱动极简版

选股: ★★★★(4-7票) + 暴跌<-20% + 缩量<0.8x + 不独跌(<15%)
排名: (0.8-量比)×2 + abs(跌幅)×1.5
仓位: TOP4, 月度调仓, 同行业≤2
出场: 20天底仓 → RSI>70+放量>1.5x → 否则60天到期
环境: CSI300<MA60 → 减到2只

执行: python src/signals/daily_generator.py [--max-stocks 2000] [--no-emotion]
"""
import os, sys, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

ARCHIVE = os.path.join(ROOT, "quant_archive")
CACHE = os.path.join(ROOT, "data", "cache")
os.makedirs(ARCHIVE, exist_ok=True)

MAX_WORKERS = 2  # tushare免费版限1500次/分钟, 2并发刚好
MIN_TRADING_DAYS = 60
FETCH_DAYS = 200


# ══════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════

def load_fundamentals() -> dict[str, dict]:
    """加载基本面缓存, 返回 {ts_code: {pe, pb, pe_ttm, total_mv, roe, debt_to_assets, ...}}"""
    path = os.path.join(CACHE, "fundamentals_daily.pkl")
    result = {}
    has_fund = False
    
    if os.path.exists(path):
        df = pd.read_pickle(path)
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
        latest = df.sort_values('trade_date').groupby('ts_code').last()
        has_fund = True
        
        for code, row in latest.iterrows():
            result[code] = {
                'pe': float(row.get('pe', 999)) if pd.notna(row.get('pe')) else 999,
                'pb': float(row.get('pb', 999)) if pd.notna(row.get('pb')) else 999,
                'pe_ttm': float(row.get('pe_ttm', 999)) if pd.notna(row.get('pe_ttm')) else 999,
                'total_mv': float(row.get('total_mv', 0)) if pd.notna(row.get('total_mv')) else 0,
                'circ_mv': float(row.get('circ_mv', 0)) if pd.notna(row.get('circ_mv')) else 0,
            }
    
    # ── 合并 ROE/负债/毛利/净利率 (优先 financials.pkl, 回退 roe_data.pkl) ──
    roe_paths = [os.path.join(CACHE, p) for p in ["financials.pkl", "roe_data.pkl"]]
    found_roe = False
    for roe_path in roe_paths:
        if not os.path.exists(roe_path):
            continue
        roe_df = pd.read_pickle(roe_path)
        roe_latest = roe_df.sort_values('end_date').groupby('ts_code').last()
        merged = 0
        roe_fields = ['roe', 'roa', 'roe_yearly', 'roic',
                      'debt_to_assets', 'debt_to_eqt', 'current_ratio', 'quick_ratio',
                      'grossprofit_margin', 'netprofit_margin',
                      'or_yoy', 'op_yoy', 'netprofit_yoy', 'equity_yoy', 'bps_yoy', 'basic_eps_yoy',
                      'q_ocf_to_sales', 'profit_dedt',
                      'eps', 'bps', 'ocfps', 'assets_turn']
        for code in roe_latest.index:
            row = roe_latest.loc[code]
            if code not in result:
                result[code] = {'pe': 999, 'pb': 999, 'pe_ttm': 999, 'total_mv': 0, 'circ_mv': 0}
            for col in roe_fields:
                val = row.get(col) if hasattr(row, 'get') else None
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    result[code][col] = float(val)
            merged += 1
        has_fund = True
        found_roe = True
        src_name = os.path.basename(roe_path)
        print(f"[基本面] fundamentals: {len([k for k in result if result[k].get('pe_ttm',999)!=999])}只 | {src_name}: {merged}只")
        break  # 只用第一个找到的文件
    
    if found_roe:
        pass  # 已打印
    elif has_fund:
        print(f"[基本面] 加载 {len(result)} 只股票 (无ROE数据)")
    else:
        print("[基本面] 无缓存, 价值/成长大师将降级为纯量价模式")
    
    return result


# ══════════════════════════════════════════════════════
# Layer 0: 北向闸门
# ══════════════════════════════════════════════════════

def check_north_gate() -> dict:
    try:
        from src.surge.north_gate import NorthGateStrategy
        path = os.path.join(CACHE, "macro_north_flow.pkl")
        if not os.path.exists(path):
            return {'open': True, 'level': 'UNKNOWN', 'reason': '无北向数据'}
        
        north = pd.read_pickle(path)
        north['date'] = pd.to_datetime(north['trade_date'].astype(str), format='%Y%m%d')
        north['north_money'] = pd.to_numeric(north['north_money'], errors='coerce')
        
        gate = NorthGateStrategy(north)
        pos = gate.position(pd.Timestamp.now().normalize())
        
        if pos >= 1.0: return {'open': True, 'level': 'FULL', 'position': 1.0}
        elif pos >= 0.5: return {'open': True, 'level': 'HALF', 'position': 0.5}
        else: return {'open': False, 'level': 'CLOSED', 'position': 0.0}
    except Exception as e:
        return {'open': True, 'level': 'ERROR', 'reason': str(e)}


# ══════════════════════════════════════════════════════
# Quality Gate — 在交给大师之前过滤垃圾股
# ══════════════════════════════════════════════════════

QUALITY_BLACKLIST = {'ST', '*ST', 'SST', 'S*ST'}

def check_quality(code: str, df: pd.DataFrame, fundamentals: dict) -> tuple[bool, str]:
    """质量闸门 — 只拦真正有问题的"""
    name = _get_stock_name(code)
    close = float(df['close'].iloc[-1])
    vol_col = 'vol' if 'vol' in df.columns else ('volume' if 'volume' in df.columns else None)
    if vol_col:
        vol = float(df[vol_col].iloc[-1])
        vols = df[vol_col].values
    else:
        vol = 0; vols = np.ones(1)
    fund = fundamentals.get(code, {})
    closes = df['close'].values; highs = df['high'].values; lows = df['low'].values

    # 1. ST
    if any(tag in name for tag in QUALITY_BLACKLIST):
        return False, 'ST/风险警示'

    # 2. 壳公司: PB>10 且 PE<0 (既不赚钱又没资产)
    pb = fund.get('pb', 999)
    pe = fund.get('pe_ttm', fund.get('pe', 999))
    if isinstance(pb,(int,float)) and not np.isnan(pb) and pb>10:
        if isinstance(pe,(int,float)) and not np.isnan(pe) and pe<0:
            return False, f'壳公司(PB{pb:.0f}>10 且 亏损)'

    # 3. 市值<30亿 (流动性太差)
    mcap = fund.get('total_mv', 0)
    if isinstance(mcap,(int,float)) and 0 < mcap < 300000:
        return False, f'市值{mcap/10000:.0f}亿<30亿'

    # 4. 日成交额<3000万
    if vol * close < 3e7:
        return False, f'日成交{vol*close/1e4:.0f}万<3000万'

    # 5. 僵尸股: 52周低位 + 无跳空
    if len(closes) >= 200:
        high_52w = np.max(highs[-200:]); low_52w = np.min(lows[-200:])
        pct = (close - low_52w) / (high_52w - low_52w) if high_52w > low_52w else 0.5
        gaps = sum(1 for i in range(-20, 0) if abs(closes[i]/closes[i-1]-1)>0.05)
        if pct < 0.2 and gaps == 0:
            return False, f'僵尸股(52周低位{pct*100:.0f}% 无跳空)'

    # 6. 换手率异常(按市值分段: 超过该市值段P99 = 高危)
    mcap_yi = fund.get('total_mv', 0) / 10000  # 亿
    turnover = vol * close / fund.get('total_mv', 1) if fund.get('total_mv', 0) > 0 else 0
    if mcap_yi > 1000 and turnover > 5:    return False, f'大盘换手{turnover:.0f}%>5%(P99)'
    if 300 < mcap_yi <= 1000 and turnover > 9: return False, f'中盘换手{turnover:.0f}%>9%(P99)'
    if 100 < mcap_yi <= 300 and turnover > 13: return False, f'中盘换手{turnover:.0f}%>13%(P99)'
    if 50 < mcap_yi <= 100 and turnover > 18:  return False, f'小盘换手{turnover:.0f}%>18%(P99)'
    if mcap_yi <= 50 and turnover > 15:        return False, f'小盘换手{turnover:.0f}%>15%(P99)'

    # 7. 量比极缩量 (<0.5 = 没人交易)
    vol_ratio = vol / np.mean(vols[-20:]) if len(vols) >= 20 else 1
    if vol_ratio < 0.5:
        return False, f'量比{vol_ratio:.1f}(极缩量)'

    # 8. 阴跌股: 连跌10天+无放量(慢慢失血, 不触发任何止损)
    if len(closes) >= 15:
        down_streak = 0
        for i in range(-15, -1):
            if closes[i] < closes[i-1] and -0.03 < closes[i]/closes[i-1]-1 < -0.003:
                down_streak += 1
            else:
                down_streak = 0
        if down_streak >= 10:
            return False, f'阴跌股(连跌{down_streak}天)'

    return True, 'OK'


def _get_stock_name(code: str) -> str:
    try:
        from src.tools.a_stock_api import get_stock_name_for_ticker
        return get_stock_name_for_ticker(code) or code
    except: return code

def _get_industry(code: str) -> str:
    try:
        from src.tools.a_stock_api import get_stock_info
        info = get_stock_info(code)
        return info.industry if info else ""
    except: return ""


# ══════════════════════════════════════════════════════
# 黄金坑策略扫描 (牛市专用)
# ══════════════════════════════════════════════════════

def check_csi300_ma60() -> tuple[bool, float, float]:
    """CSI300 > MA60? 返回(闸门开, CSI300, MA60)"""
    try:
        csi_p = pd.read_pickle(os.path.join(CACHE, "csi300.pkl"))
        csi_p = csi_p.sort_values('trade_date')
        csi_close = float(csi_p['close'].iloc[-1])
        csi_ma60 = float(csi_p['close'].iloc[-60:].mean())
        return (csi_close > csi_ma60, csi_close, csi_ma60)
    except:
        return (False, 0, 0)

def scan_golden_pit(prices_dict: dict, ind_map: dict, 
                     name_map: dict = None) -> list[dict]:
    """
    黄金坑策略扫描 (2026-05-08 定稿)
    
    选票: 趋势4-5/5 + 非ST
    过滤: 跌速>0.5%/d + 量比<3x + 深度-10~-18% + 5天内触达-10%
    进场: -10%, T+1
    信号超量排序: 趋势5/5优先, 跌速降序
    """
    gate_open, csi_close, csi_ma60 = check_csi300_ma60()
    if not gate_open:
        return []  # 闸门关, 黄金坑不激活
    
    signals = []
    for code, df in prices_dict.items():
        try:
            df = df.copy().sort_values('trade_date').reset_index(drop=True)
            close = df['close'].values.astype(float)
            vol = df['vol'].values.astype(float)
            n = len(df)
            if n < 250: continue
            
            end = n - 1
            
            # 趋势得分
            ma20 = pd.Series(close).rolling(20).mean().values[end]
            ma50 = pd.Series(close).rolling(50).mean().values[end]
            ma60 = pd.Series(close).rolling(60).mean().values[end]
            ma120 = pd.Series(close).rolling(120).mean().values[end]
            ma250 = pd.Series(close).rolling(250).mean().values[end]
            
            t1 = close[end] > ma250
            t2 = pd.Series(close).rolling(50).mean().values[end] > \
                 pd.Series(close).rolling(50).mean().values[max(0, end-20)]
            t3 = ma20 > ma60
            t4 = ma50 > ma120
            t5 = ma120 > ma250
            trend = sum([t1, t2, t3, t4, t5])
            if trend < 4: continue
            
            # 找最近峰值和当前回调
            lookback = min(120, end)
            peak_idx = end - lookback + np.argmax(close[end-lookback:end+1])
            peak_price = close[peak_idx]
            current_dd = (close[end] / peak_price - 1) * 100
            
            if current_dd > -5 or current_dd < -18: continue
            
            days_from_peak = end - peak_idx
            if days_from_peak <= 0: continue
            daily_speed = abs(current_dd) / days_from_peak
            if daily_speed < 0.5: continue
            
            # 量比
            dv = np.mean(vol[max(peak_idx, end-20):end+1])
            pv = np.mean(vol[max(0, peak_idx-20):peak_idx+1])
            vol_ratio = dv / pv if pv > 0 else 1
            if vol_ratio >= 3.0: continue
            
            # -10%触达检查
            target = peak_price * 0.90
            cross_idx = None
            for j in range(peak_idx+1, end+1):
                if close[j] <= target:
                    cross_idx = j; break
            if cross_idx is None: continue
            days_since = end - cross_idx
            if days_since > 5: continue
            
            # ST / 科创板过滤
            name = name_map.get(code, '') if name_map else ''
            if 'ST' in name or '*ST' in name: continue
            if code.startswith('688'): continue  # 科创板
            
            # 120日收益
            ret120 = (close[end] / close[max(0, end-120)] - 1) * 100
            
            # 排序分
            rank_score = trend * 100 + daily_speed * 10
            
            signals.append({
                'code': code, 'name': name,
                'industry': ind_map.get(code, ''),
                'trend': trend, 'dd': current_dd,
                'speed': daily_speed, 'vol_ratio': vol_ratio,
                'ret120': ret120, 'price': close[end],
                'peak': peak_price, 'rank': rank_score,
            })
        except:
            continue
    
    # 排序: 趋势5优先, 跌速降序
    signals.sort(key=lambda x: (-x['trend'], -x['speed']))
    return signals


# ══════════════════════════════════════════════════════
# Layer 1+2: 大师 + 共识 (v2 信号簇版)
# ══════════════════════════════════════════════════════

def analyze_one_stock(code: str, df: pd.DataFrame, fundamentals: dict, 
                      sector_ret60: float = 0.0) -> dict:
    """v5.2: 返回评级 + 暴跌/缩量/独跌指标"""
    from src.masters import run_all_masters, get_consensus, get_rating

    if 'vol' not in df.columns and 'volume' in df.columns:
        df = df.copy(); df['vol'] = df['volume']

    fund = fundamentals.get(code)
    results = run_all_masters(df, fund)
    consensus = get_consensus(results, min_clusters=3, require_contrarian=True)
    rating = get_rating(consensus)

    name = _get_stock_name(code)
    industry = _get_industry(code)
    closes = df['close'].values
    close = float(closes[-1])

    # v5.2: 暴跌/缩量/独跌指标
    ret60d = close / closes[-60] - 1 if len(closes) >= 60 else 0
    vols = df['vol'].values if 'vol' in df.columns else np.ones(len(closes))
    vol_ratio = float(vols[-1] / np.mean(vols[-60:])) if len(vols) >= 60 else 1
    excess_crash = abs(ret60d) - abs(sector_ret60) if sector_ret60 != 0 else 0

    # v5.2: 入选检查 (仅★★★★ + 暴跌 + 缩量 + 不独跌)
    is_crash = ret60d < -0.20
    is_lowvol = vol_ratio < 0.8
    is_sector_crash = excess_crash < 0.15  # 不独跌
    score = (0.8 - vol_ratio) * 2 + abs(ret60d) * 1.5 if (is_crash and is_lowvol) else 0
    qualified = (rating == '★★★★' and is_crash and is_lowvol and is_sector_crash)

    return {
        'ts_code': code, 'name': name, 'industry': industry, 'close': close,
        'votes': consensus['total_votes'],
        'rating': rating,
        'has_risk': consensus.get('has_risk', False),
        'consensus': consensus['consensus'],
        'high_confidence': consensus['high_confidence'],
        # v5.2 新增
        'ret60d': ret60d,
        'vol_ratio': vol_ratio,
        'excess_crash': excess_crash,
        'is_crash': is_crash,
        'is_lowvol': is_lowvol,
        'is_sector_crash': is_sector_crash,
        'score': score,
        'qualified': qualified,
        'voters': ', '.join(f"{n}({g})" for n, g, s in consensus['voters']),
    }


# ══════════════════════════════════════════════════════
# Layer 2.5: 情绪验证 (可选)
# ══════════════════════════════════════════════════════

def fuse_emotion(picks: list[dict], top_n: int = 20) -> list[dict]:
    """对高确信候选做LLM情绪验证"""
    try:
        from src.emotion.fusion import analyze_emotion
    except ImportError:
        print("[情绪] 模块不可用,跳过")
        return picks
    
    print(f"[情绪] 验证Top {min(top_n, len(picks))} 候选...")
    for p in picks[:top_n]:
        try:
            code = p['ts_code']
            fusion = analyze_emotion(code)
            p['emotion_score'] = fusion.get('fusion_score', 50)
            p['emotion_label'] = fusion.get('label', 'neutral')
            p['emotion_confidence'] = fusion.get('confidence', 0)
        except Exception:
            p['emotion_score'] = 50; p['emotion_label'] = 'neutral'; p['emotion_confidence'] = 0
    
    return picks


# ══════════════════════════════════════════════════════
# 股票池 + 价格
# ══════════════════════════════════════════════════════

def get_stock_pool() -> list[str]:
    """股票池 = CSI全市场 + 16板块补漏"""
    all_codes = set()
    # CSI 300+500+1000
    path = os.path.join(CACHE, "csi500_stocks.pkl")
    if os.path.exists(path):
        s = pd.read_pickle(path)
        if isinstance(s, pd.Series): s = s.tolist()
        all_codes.update(s)
        print(f"  CSI全市场: {len(s)} 只")
    # 16板块(抓小票)
    try:
        from src.tools.a_stock_api import get_16_sector_stocks
        sd = get_16_sector_stocks()
        for _, stocks in sd.items():
            all_codes.update(stocks[:30])
        print(f"  +16板块补漏: {len(all_codes)} 只(合并)")
    except:
        pass
    return sorted(all_codes)


def _fetch_one(code: str, start_str: str, end_str: str) -> tuple[str, Optional[pd.DataFrame]]:
    try:
        from src.tools.data_fetcher import get_prices
        import time as _time
        _time.sleep(0.05)  # 限速: 2并发×0.05s≈40次/秒, 安全
        prices = get_prices(code, start_str, end_str)
        if not prices or len(prices) < MIN_TRADING_DAYS: return code, None
        rows = [{"date": p.date, "open": p.open, "high": p.high,
                 "low": p.low, "close": p.close, "vol": p.volume}
                for p in prices if hasattr(p, 'date') and p.date]
        df = pd.DataFrame(rows)
        if df.empty or len(df) < MIN_TRADING_DAYS: return code, None
        return code, df.sort_values("date").reset_index(drop=True)
    except: return code, None


def fetch_all_prices(pool: list[str], days: int = FETCH_DAYS) -> dict[str, pd.DataFrame]:
    """v5.2: 优先从缓存加载, 只补拉增量。缓存主力: prices_full.pkl"""
    t0 = time.time()
    result = {}
    
    # Step 1: 从 prices_full.pkl 加载主力缓存
    cache_path = os.path.join(CACHE, "prices_full.pkl")
    cache_loaded = 0
    latest_cache_date = None
    if os.path.exists(cache_path):
        try:
            cache_data = pd.read_pickle(cache_path)
            for code, df in cache_data.items():
                if code not in pool: continue
                df = df.sort_values('trade_date').reset_index(drop=True)
                if 'vol' not in df.columns and 'volume' in df.columns:
                    df['vol'] = df['volume']
                result[code] = df
                cache_loaded += 1
                # 跟踪最新日期
                if len(df) > 0:
                    last_date = pd.to_datetime(df['trade_date'].iloc[-1])
                    if latest_cache_date is None or last_date > latest_cache_date:
                        latest_cache_date = last_date
            print(f"  缓存加载: {cache_loaded}/{len(pool)}只, 最新日期: {latest_cache_date.strftime('%Y-%m-%d') if latest_cache_date else 'N/A'}")
        except Exception as e:
            print(f"  缓存加载失败: {e}, 降级到全量拉取")
    
    # Step 2: 补拉增量 (从最新缓存日期到今日)
    today_str = datetime.now().strftime("%Y-%m-%d")
    need_delta = latest_cache_date is not None and latest_cache_date.strftime("%Y-%m-%d") < today_str
    
    if need_delta:
        # 只拉缓存里有的股票(新股票不管)
        delta_start = (latest_cache_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  补拉增量: {delta_start} → {today_str} ({len(result)}只)")
        
        # 只为已有缓存的股票补拉(新股票下次全量跑会进缓存)
        updated = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {}
            for code in list(result.keys()):
                futs[ex.submit(_fetch_one, code, delta_start, today_str)] = code
            for fut in as_completed(futs):
                code = futs[fut]
                try:
                    _, delta_df = fut.result()
                    if delta_df is not None and len(delta_df) > 0:
                        # 合并到已有数据
                        existing = result[code]
                        delta_df = delta_df.sort_values('date').reset_index(drop=True)
                        if 'vol' not in delta_df.columns and 'volume' in delta_df.columns:
                            delta_df['vol'] = delta_df['volume']
                        # 去重合并
                        combined = pd.concat([existing, delta_df]).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
                        result[code] = combined
                        updated += 1
                except Exception as e:
                    if updated < 3:
                        print(f"    ⚠ {code} 增量失败: {e}")
        print(f"  增量更新: {updated}只")
    
    # Step 3: 如果缓存完全不可用, 降级到全量拉取
    if cache_loaded == 0:
        print(f"  降级: 全量Tushare拉取...")
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(_fetch_one, c, start, end): c for c in pool}
            for fut in as_completed(futs):
                code, df = fut.result()
                if df is not None: result[code] = df
    
    print(f"  最终可用: {len(result)}/{len(pool)} ({time.time()-t0:.0f}s)")
    return result


# ══════════════════════════════════════════════════════
# Layer 3: 报告
# ══════════════════════════════════════════════════════

def write_report(picks: list[dict], gate: dict, stock_count: int,
                 elapsed: float, has_fundamentals: bool,
                 stat_pass: int = 0, stat_total: int = 0, stat_qual: int = 0,
                 ai_startups: list = None,
                 gp_signals: list = None, csi300_ma60: tuple = None) -> str:
    now = datetime.now()
    month_dir = os.path.join(ARCHIVE, now.strftime("%Y-%m"))
    os.makedirs(month_dir, exist_ok=True)
    path = os.path.join(month_dir, f"daily_picks_{now.strftime('%Y%m%d_%H%M')}.md")

    # v5.2: no rating groups, no position sizing — unified TOP N
    qualified = [p for p in picks if p.get('qualified')]

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 每日选股 v5.2 (数据驱动极简版)\n\n")
        f.write(f"生成: {now.strftime('%Y-%m-%d %H:%M')}\n\n")

        # Gate + Regime
        f.write(f"## 北向闸门\n\n- 状态: **{gate['level']}**\n- 仓位: {gate.get('position',0)*100:.0f}%\n\n")
        
        # v5.2: MA60环境
        try:
            csi300 = pd.read_pickle(os.path.join(CACHE, "csi300.pkl"))
            csi300 = csi300.sort_values('trade_date')
            csi300_close = float(csi300['close'].iloc[-1])
            csi300_ma60 = float(csi300['close'].iloc[-60:].mean())
            below_ma60 = csi300_close < csi300_ma60
            f.write(f"## 市场环境\n\n- CSI300: {csi300_close:.0f} | MA60: {csi300_ma60:.0f}\n")
            f.write(f"- 趋势: {'⚠️ MA60以下 → 仓位减半(≤2只)' if below_ma60 else '✓ MA60以上 → 正常仓位(≤4只)'}\n\n")
        except:
            f.write(f"## 市场环境\n\n- 无CSI300数据\n\n")

        # Position sizing
        max_pos = 2 if below_ma60 else 4
        f.write(f"## 仓位建议\n\n- 最多 **{max_pos}** 只, 每只 **¥25,000**\n")
        f.write(f"- 同行业 ≤ 2只\n")
        if not gate['open']:
            f.write("\n> ⚠️ 北向闸门关闭，不开新仓。\n")
        f.write(f"\n- 基本面: {'✓' if has_fundamentals else '✗'}\n\n")

        # Exit rules
        f.write(f"## 出场规则 (数据驱动)\n\n")
        f.write(f"| 条件 | 动作 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| RSI>70 + 放量>1.5x | 全出 ← 市场说反弹结束 |\n")
        f.write(f"| 60天到期 | 全出 |\n")
        f.write(f"| 最低持有20天 | 20天内不平仓 |\n")
        f.write(f"| CSI300<MA60 | 仓位减到2只 |\n")
        f.write(f"| 同行业>2只 | 排名低的踢掉 |\n\n")

        # Qualified picks (v5.2: all ★★★★ with crash+lowvol+sector)
        top = picks[:max_pos]
        f.write(f"## 精选TOP{max_pos} (★★★☆+暴跌+缩量+不独跌)\n\n")
        if top:
            f.write("| # | 代码 | 名称 | 行业 | 价格 | 票 | 60日跌 | 量比 | 独跌 | 得分 |\n")
            f.write("|---|------|------|------|------|----|--------|------|------|------|\n")
            for i, p in enumerate(top, 1):
                ret_str = f"{p.get('ret60d',0)*100:+.0f}%"
                vol_str = f"{p.get('vol_ratio',1):.2f}"
                exc_str = f"{p.get('excess_crash',0)*100:+.0f}%"
                f.write(f"| {i} | {p['ts_code']} | {p['name']} | {p.get('industry','?')} | {p['close']:.2f} | {p['votes']} | {ret_str} | {vol_str} | {exc_str} | {p.get('score',0):.1f} |\n")
        else:
            f.write("无合格信号\n")

        # All qualified
        remaining = picks[max_pos:]
        if remaining:
            f.write(f"\n## 备选 ({len(remaining)}只)\n\n")
            f.write("| # | 代码 | 名称 | 行业 | 价格 | 票 | 60日跌 | 量比 | 得分 |\n")
            f.write("|---|------|------|------|------|----|--------|------|------|\n")
            for i, p in enumerate(remaining[:30], 1):
                ret_str = f"{p.get('ret60d',0)*100:+.0f}%"
                vol_str = f"{p.get('vol_ratio',1):.2f}"
                f.write(f"| {i} | {p['ts_code']} | {p['name']} | {p.get('industry','?')} | {p['close']:.2f} | {p['votes']} | {ret_str} | {vol_str} | {p.get('score',0):.1f} |\n")

        # Stats
        f.write(f"\n## 统计\n\n"
                f"- 股票池: {stock_count}\n"
                f"- 通过质量: {stat_pass} | 大师分析: {stat_total} | ★★★★: {sum(1 for p in picks if p.get('rating')=='★★★★')}\n"
                f"- 合格(暴跌+缩量+不独跌): {stat_qual}\n"
                f"- AI启动信号: {len(ai_startups) if ai_startups else 0}只\n"
                f"- 耗时: {elapsed:.0f}s\n")
        
        # AI startup signals
        if ai_startups:
            f.write(f"\n## 🔥 AI启动信号 (量>1.5x + 涨>4% + 收在高位)\n\n")
            f.write("| # | 代码 | 名称 | 行业 | 价格 | 涨幅 | 量比 | 共振 |\n")
            f.write("|---|------|------|------|------|------|------|------|\n")
            for i, s in enumerate(ai_startups[:20], 1):
                res_tag = '✅' if s.get('resonance')=='共振' else '⚠️'
                f.write(f"| {i} | {s['ts_code']} | {s['name']} | {s['industry']} | {s['close']:.2f} | {s['ret_today']:+.1%} | {s['vol_ratio']:.1f}x | {res_tag}{s.get('resonance','?')} |\n")
        
        # ── 黄金坑策略 ──
        if gp_signals:
            gp_open, gp_csi, gp_ma60 = csi300_ma60 if csi300_ma60 else (True, 0, 0)
            gp_top = gp_signals[:4]  # TOP4
            
            f.write(f"\n## ⛏️ 黄金坑策略 (牛市趋势回调)\n\n")
            f.write(f"- CSI300: {gp_csi:.0f} | MA60: {gp_ma60:.0f} | {'✅ 牛市模式' if gp_open else '❌ 闸门关闭'}\n")
            f.write(f"- 策略: 趋势4-5/5 + 跌速>0.5%/d + 深度-10~-18% + 5天内触达\n")
            f.write(f"- 信号: {len(gp_signals)}只 | 精选TOP4 | 持有50-60天到期\n\n")
            
            if gp_top:
                f.write("| # | 代码 | 名称 | 行业 | 趋势 | 回调 | 跌速 | 量比 | 120d | 现价 |\n")
                f.write("|---|------|------|------|------|------|------|------|------|------|\n")
                for i, s in enumerate(gp_top, 1):
                    f.write(f"| {i} | {s['code']} | {s['name']} | {s['industry']} | "
                           f"{s['trend']}/5 | {s['dd']:+.1f}% | {s['speed']:.2f}/d | "
                           f"{s['vol_ratio']:.1f}x | {s['ret120']:+.0f}% | {s['price']:.2f} |\n")
            
            remaining = gp_signals[4:]
            if remaining:
                f.write(f"\n**备选 ({len(remaining)}只):** ")
                f.write(", ".join(f"{s['name']}({s['trend']}/5,{s['dd']:+.0f}%)" for s in remaining[:15]))
                f.write("\n")

    print(f"\\n[REPORT] {path}")
    return path


# ══════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════

def daily_generate(max_stocks: int = 2000, min_price: float = 3.0,
                   max_price: float = 200.0, use_emotion: bool = True) -> list[dict]:
    t_start = time.time()
    
    # ── Data ──
    print("=== 加载数据 ===")
    fundamentals = load_fundamentals()
    has_fund = len(fundamentals) > 0

    # ── 行业数据 + 板块上下文 ──
    sw_path = os.path.join(CACHE, "sw_industry.pkl")
    ind_map = {}
    if os.path.exists(sw_path):
        sw = pd.read_pickle(sw_path)
        ind_map = dict(zip(sw['ts_code'], sw['industry']))
        # 注入行业到 fundamentals
        for code, ind in ind_map.items():
            if code in fundamentals:
                fundamentals[code]['industry'] = ind
        print(f"[行业] {len(ind_map)}只映射, {sw['industry'].nunique()}个行业")
    else:
        print("[行业] 无 sw_industry.pkl, 行业大师降级")
    
    # ── Layer 0 ──
    print("\\n=== Layer 0: 北向闸门 ===")
    gate = check_north_gate()
    print(f"  状态: {gate['level']} | 仓位: {gate.get('position',0)*100:.0f}%")
    
    # ── Stock pool ──
    print("\\n=== 股票池 ===")
    pool = get_stock_pool()
    if max_stocks and len(pool) > max_stocks:
        pool = pool[:max_stocks]
    print(f"  {len(pool)} 只")
    
    # ── Prices ──
    print("\\n=== 拉取价格 ===")
    prices_dict = fetch_all_prices(pool)
    if len(prices_dict) < 10:
        print("价格拉取失败")
        return []
    
    # ── 行业板块上下文 (在跑大师前计算一次) ──
    sector_ctx = {}
    sector_ret60_map = {}  # v5.2: 每板块60日收益(用于独跌判断)
    if ind_map:
        from collections import defaultdict
        ind_returns = defaultdict(list)
        ind_ret60 = defaultdict(list)
        for code, df in prices_dict.items():
            ind = ind_map.get(code, '')
            if ind and len(df) >= 60:
                c = float(df['close'].iloc[-1])
                ret_20d = c / float(df['close'].iloc[-20]) - 1
                ret_60d = c / float(df['close'].iloc[-60]) - 1
                ret_5d = c / float(df['close'].iloc[-5]) - 1 if len(df) >= 5 else 0
                vols = df['vol'].values if 'vol' in df.columns else df.get('volume', pd.Series([0])).values
                vol_ratio = vols[-1] / np.mean(vols[-20:]) if len(vols) >= 20 else 1
                ind_returns[ind].append({'ret_20d': ret_20d, 'ret_5d': ret_5d, 'vol_ratio': vol_ratio})
                ind_ret60[ind].append(ret_60d)
        
        for ind in ind_ret60:
            if len(ind_ret60[ind]) >= 3:
                sector_ret60_map[ind] = float(np.median(ind_ret60[ind]))
        
        all_r20 = []
        for ind, rets in ind_returns.items():
            if len(rets) < 3: continue
            r20 = [r['ret_20d'] for r in rets]; r5 = [r['ret_5d'] for r in rets]
            all_r20.extend(r20)
            
            # 暴涨统计: 20日>30%的股票数 + 平均量比
            surge = [r for r in rets if r['ret_20d'] > 0.3]
            surge_count = len(surge)
            avg_surge_vol = float(np.mean([s['vol_ratio'] for s in surge])) if surge else 0
            
            sector_ctx[ind] = {
                'ret_20d_median': float(np.median(r20)),
                'ret_5d_median': float(np.median(r5)),
                'n_stocks': len(rets),
                'surge_count': surge_count,
                'avg_surge_vol': avg_surge_vol,
                'is_emerging': (surge_count >= 3 and avg_surge_vol >= 1.5),
            }
        
        market_ret_20d = float(np.median(all_r20)) if all_r20 else 0
        # 市场环境分类 (数据驱动: 板块扩散只在极端市有效)
        if market_ret_20d > 0.05: regime = 'strong_bull'
        elif market_ret_20d > 0: regime = 'mild_bull'
        elif market_ret_20d > -0.02: regime = 'mild_bear'
        else: regime = 'strong_bear'
        sector_ctx['__market__'] = {
            'ret_20d_median': market_ret_20d,
            'regime': regime,
            'is_extreme': abs(market_ret_20d) > 0.02,  # 极端市(板块扩散有效)
        }
        
        from src.masters import set_sector_context
        set_sector_context(sector_ctx)
        emerging = sum(1 for v in sector_ctx.values() if isinstance(v, dict) and v.get('is_emerging'))
        print(f"[板块] {len(sector_ctx)-1}个行业, {emerging}个Emerging Theme(≥3只暴涨+放量)")
    else:
        print("[板块] 无行业数据, 行业大师降级")

    # ── Layer 1+2: 策略选择 ──
    gp_open, gp_csi, gp_ma60 = check_csi300_ma60()
    print(f"\n=== 市场: {'牛市(黄金坑)' if gp_open else '熊市(v5.2)'} CSI300={gp_csi:.0f} MA60={gp_ma60:.0f} ===\n")
    
    picks = []
    gp_signals = []
    stat_total = stat_pass = stat_qual = 0
    
    if gp_open:
        # ═══ 牛市: 黄金坑策略 ═══
        print(f"=== ⛏️ 黄金坑策略 ===")
        gp_name_map = {}
        try:
            gp_batch = pd.read_pickle(os.path.join(CACHE, "golden_pit_batch_results.pkl"))
            gp_name_map = {s['code']: s['name'] for s in gp_batch}
        except: pass
        gp_signals = scan_golden_pit(prices_dict, ind_map, gp_name_map)
        stat_total = stat_pass = stat_qual = len(gp_signals)
        print(f"  信号: {len(gp_signals)}只 (TOP4: {', '.join(s['name'] for s in gp_signals[:4]) if len(gp_signals)>=4 else 'N/A'})")
        picks = [{'ts_code': s['code'], 'name': s['name'], 'industry': s['industry'],
                  'close': s['price'], 'votes': s['trend'], 'qualified': True} for s in gp_signals[:4]]
    else:
        # ═══ 熊市: v5.2大师共识 ═══
        print(f"=== Layer 1+2: 大师分析 ({len(prices_dict)}只) ===")
        for code, df in prices_dict.items():
            try:
                c = float(df['close'].iloc[-1])
                if c < min_price or c > max_price: continue
                passed, _ = check_quality(code, df, fundamentals)
                if not passed: continue
                stat_pass += 1
                ind = ind_map.get(code, '')
                sec_ret = sector_ret60_map.get(ind, 0)
                result = analyze_one_stock(code, df, fundamentals, sec_ret)
                stat_total += 1
                if result['qualified']:
                    if has_fund and code in fundamentals:
                        result['pe'] = fundamentals[code].get('pe_ttm', 999)
                    picks.append(result)
                    stat_qual += 1
            except: continue
        picks.sort(key=lambda x: x.get('score', 0), reverse=True)
        print(f"  大师分析完成: {stat_total}只, ★★★★: {stat_qual}")

    # ── AI启动扫描 (牛熊都跑) ──
    print(f"\n=== AI启动扫描 ===")
    ai_startups = []
    ai_sectors = {'半导体','通信设备','IT设备','软件服务','互联网','元器件','电气设备','专用机械'}
    for code, df in prices_dict.items():
        ind = ind_map.get(code, '')
        if ind not in ai_sectors: continue
        if len(df) < 20: continue
        closes = df['close'].values.astype(float)
        vols = df['vol'].values if 'vol' in df.columns else df.get('volume', pd.Series([0])).values.astype(float)
        if len(vols) < 20: continue
        ret_today = closes[-1]/closes[-2]-1
        vol_ratio = vols[-1]/np.mean(vols[-20:])
        high_confirm = closes[-1] > float(df['high'].values[-1])*0.85
        if vol_ratio > 1.5 and ret_today > 0.04 and high_confirm:
            # 大盘共振检测
            csi_ret = 0.0
            try:
                csi_p = pd.read_pickle(os.path.join(CACHE, "csi300.pkl"))
                csi_p = csi_p.sort_values('trade_date')
                csi_ret = float(csi_p['pct_chg'].iloc[-1]) / 100
            except: pass
            resonance = '共振' if csi_ret > 0.005 else '弱共振'
            name = _get_stock_name(code)
            ai_startups.append({
                'ts_code': code, 'name': name, 'industry': ind,
                'close': float(closes[-1]), 'ret_today': ret_today,
                'vol_ratio': vol_ratio, 'resonance': resonance,
            })
    ai_startups.sort(key=lambda x: x['vol_ratio'], reverse=True)
    print(f"  启动信号: {len(ai_startups)}只")
    
    # ── 黄金坑扫描 ──
    print(f"\n=== ⛏️ 黄金坑策略扫描 ===")
    gp_open, gp_csi, gp_ma60 = check_csi300_ma60()
    gp_signals = []
    if gp_open:
        # Load name map
        gp_name_map = {}
        try:
            gp_batch = pd.read_pickle(os.path.join(CACHE, "golden_pit_batch_results.pkl"))
            gp_name_map = {s['code']: s['name'] for s in gp_batch}
        except:
            pass
        gp_signals = scan_golden_pit(prices_dict, ind_map, gp_name_map)
        print(f"  信号: {len(gp_signals)}只 (TOP4: {', '.join(s['name'] for s in gp_signals[:4]) if len(gp_signals)>=4 else '不足'})")
    else:
        print(f"  CSI300({gp_csi:.0f}) < MA60({gp_ma60:.0f}) — 闸门关闭, 改用v5.2")
    
    # ── Layer 2.6: 情绪(可选) ──
    if use_emotion:
        print(f"\n=== Layer 2.6: 情绪验证 ===")
        picks = fuse_emotion(picks)
    
    # ── Layer 3 ──
    print(f"\n=== Layer 3: 报告 ===")
    elapsed = time.time() - t_start
    report_path = write_report(picks, gate, len(pool), elapsed, has_fund, 
                                stat_pass, stat_total, stat_qual, ai_startups,
                                gp_signals, (gp_open, gp_csi, gp_ma60))

    # ── Layer 4: 持仓检查 ──
    print(f"\\n=== Layer 4: 持仓出场检查 ===")
    holdings_path = os.path.join(ROOT, "data", "holdings.json")
    if os.path.exists(holdings_path):
        import json
        from src.signals.position_manager import ExitEngine
        with open(holdings_path, encoding='utf-8') as fh:
            holdings_data = json.load(fh)
        engine = ExitEngine()
        exit_signals = []
        for h in holdings_data.get('holdings', []):
            code = h['ts_code']
            if code in prices_dict:
                df = prices_dict[code]
                if 'vol' not in df.columns and 'volume' in df.columns:
                    df = df.copy(); df['vol'] = df['volume']
                signals = engine.check(h, df, gate['open'])
                if signals:
                    c = float(df['close'].iloc[-1])
                    ret = (c / h['entry_price'] - 1) * 100
                    for sig in signals:
                        exit_signals.append(f"  {code} {h['name']}: {sig['reason']} (现价¥{c:.2f} 浮{ret:+.1f}%)")
                        try:
                            print(exit_signals[-1])
                        except UnicodeEncodeError:
                            print(f"  {code} {h['name']}: {sig['reason']}", flush=True)
        if exit_signals:
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(f"\n## ⚠️ 出场信号\n\n")
                for s in exit_signals:
                    f.write(f"- {s}\n")

        # ── 板块联动检查 ──
        sector_warnings = []
        for h in holdings_data.get('holdings', []):
            code = h['ts_code']
            prefix = code[:3]  # 代码前缀代理板块
            # 找同板块股票
            peers = [c for c in prices_dict if c[:3] == prefix and c != code]
            broken = 0
            for p_code in peers[:10]:  # 只查前10只
                p_df = prices_dict[p_code]
                p_closes = p_df['close'].values if 'close' in p_df.columns else []
                if len(p_closes) >= 22:
                    p_ma20 = np.mean(p_closes[-20:])
                    p_ma20_prev = np.mean(p_closes[-21:-1])
                    if p_closes[-1] < p_ma20 and p_closes[-2] < p_ma20_prev:
                        broken += 1
            if broken >= 2:
                c = float(prices_dict[code]['close'].iloc[-1])
                sector_warnings.append(f"  {code} {h['name']}: 同板块{broken}只破MA20 (现价¥{c:.2f})")
                try:
                    print(sector_warnings[-1])
                except UnicodeEncodeError:
                    pass
        if sector_warnings:
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(f"\n## ⚠️ 板块联动预警\n\n")
                for s in sector_warnings:
                    f.write(f"- {s}\n")
        elif holdings_data.get('holdings'):
            try:
                print("  无出场信号，继续持有")
            except UnicodeEncodeError:
                pass
    else:
        print("  无持仓文件，跳过")

    # v5.2 summary
    qualified_count = sum(1 for p in picks if p.get('qualified'))
    print(f"\n{'='*50}")
    print(f"完成! {elapsed:.0f}s | 闸门:{gate['level']} | 合格(暴跌+缩量+不独跌):{qualified_count}")
    if picks:
        top_info = []
        for p in picks[:5]:
            ret_pct = p.get('ret60d', 0) * 100
            score = p.get('score', 0)
            top_info.append((p['ts_code'], p['name'], f'跌{ret_pct:.0f}%', f'分{score:.1f}'))
        print(f"TOP5: {top_info}")
    
    return picks


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="每日主升浪信号 v3.0")
    p.add_argument("--max-stocks", type=int, default=2000)
    p.add_argument("--min-price", type=float, default=3.0)
    p.add_argument("--max-price", type=float, default=200.0)
    p.add_argument("--no-emotion", action="store_true", help="跳过LLM情绪验证")
    args = p.parse_args()
    
    picks = daily_generate(
        max_stocks=args.max_stocks, min_price=args.min_price,
        max_price=args.max_price, use_emotion=not args.no_emotion,
    )
    print(f"\\n共 {len(picks)} 个候选")
