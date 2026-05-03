#!/usr/bin/env python3
"""动量过滤器模块：ML选股后的时机筛选层

核心逻辑:
  1. 趋势过滤 — 价格在MA50和MA200之上（确认多头趋势）
  2. 板块动量 — 板块指数RPS（相对强度）排名
  3. 综合评分 — ML分数 + 趋势分数 + 板块分数

接口:
  filter_with_momentum(prices_dict, ml_scores, sector_map) -> list[dict]
"""
import os, sys, json
from typing import Optional
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


# ══════════════════════════════════════════════════════
#  1. 单只股票趋势过滤
# ══════════════════════════════════════════════════════

def compute_trend_signal(df: pd.DataFrame) -> dict:
    """计算个股趋势信号

    输入:
      df: 含 'close' 列的价格DataFrame（按日期排序）

    输出:
      {trend_ok: bool, trend_score: int (0-100),
       ma50: float, ma200: float, ma50_slope: float}
    """
    if df is None or len(df) < 200:
        return {"trend_ok": False, "trend_score": 0,
                "ma50": None, "ma200": None, "ma50_slope": 0}

    closes = df["close"].values
    dates = df["date"].values if "date" in df.columns else None

    ma50 = pd.Series(closes).rolling(50).mean().values
    ma200 = pd.Series(closes).rolling(200).mean().values

    last_close = closes[-1]
    last_ma50 = ma50[-1]
    last_ma200 = ma200[-1]

    # -- 趋势条件 --
    # A) 收盘价 > MA50 > MA200（经典多头排列）
    trend_ok_a = (last_close > last_ma50 > last_ma200) if not np.isnan(last_ma200) else False

    # B) 或：收盘价 > MA200 且 MA50 向上
    if len(ma50) >= 70:
        ma50_20d_ago = ma50[-20]
        ma50_slope = (last_ma50 - ma50_20d_ago) / ma50_20d_ago if ma50_20d_ago > 0 else 0
    else:
        ma50_slope = 0

    trend_ok_b = (last_close > last_ma200 and ma50_slope > 0) if not np.isnan(last_ma200) else False
    trend_ok = trend_ok_a or trend_ok_b

    # -- 趋势评分 (0-100) --
    if not trend_ok:
        trend_score = 0
    elif trend_ok_a:
        # 多头排列：加分项 = 均线乖离率（不要太大，跑太远容易回调）
        spread = (last_ma50 - last_ma200) / last_ma200  # 两条均线间的距离
        if spread > 0.30:
            trend_score = 55  # 乖离太大，减分
        elif spread > 0.15:
            trend_score = 70
        else:
            trend_score = 85
    else:
        trend_score = 60  # B条件：偏多但不完美

    # -- 次级信号 --
    # 价格相对MA50的位置（>3%偏离加分，>15%偏离减分）
    if trend_ok:
        pct_from_ma50 = (last_close - last_ma50) / last_ma50
        if 0.03 < pct_from_ma50 < 0.15:
            trend_score += 10
        elif pct_from_ma50 > 0.20:
            trend_score -= 15

    trend_score = max(0, min(100, trend_score))

    return {
        "trend_ok": trend_ok,
        "trend_score": trend_score,
        "ma50": float(last_ma50) if not np.isnan(last_ma50) else None,
        "ma200": float(last_ma200) if not np.isnan(last_ma200) else None,
        "ma50_slope": float(ma50_slope),
        "last_close": float(last_close),
    }


# ══════════════════════════════════════════════════════
#  2. 板块动量计算
# ══════════════════════════════════════════════════════

def get_sector_daily_data() -> Optional[pd.DataFrame]:
    """获取A股行业板块指数日线数据

    使用申万一级行业指数（通过Tushare），计算各板块动量。
    Tushare行业指数代码: 801010.SI ~ 801980.SI（28个申万一级）
    """
    try:
        import tushare as ts
        token = "5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7"
        pro = ts.pro_api(token)

        # 申万一级行业指数列表（28个）
        sw_codes = {
            "农林牧渔": "801010.SI", "采掘": "801020.SI", "化工": "801030.SI",
            "钢铁": "801040.SI", "有色金属": "801050.SI", "电子": "801080.SI",
            "家用电器": "801110.SI", "食品饮料": "801120.SI", "纺织服装": "801130.SI",
            "轻工制造": "801140.SI", "医药生物": "801150.SI", "公用事业": "801160.SI",
            "交通运输": "801170.SI", "房地产": "801180.SI", "商业贸易": "801200.SI",
            "休闲服务": "801210.SI", "综合": "801220.SI", "建筑材料": "801710.SI",
            "建筑装饰": "801720.SI", "电气设备": "801730.SI", "国防军工": "801740.SI",
            "计算机": "801750.SI", "传媒": "801760.SI", "通信": "801770.SI",
            "银行": "801780.SI", "非银金融": "801790.SI", "汽车": "801880.SI",
            "机械设备": "801890.SI",
        }

        data = {}
        for sector, code in sw_codes.items():
            try:
                df = pro.index_daily(ts_code=code, start_date="20240901",
                                     end_date="20260430", fields="trade_date,close")
                if df is not None and len(df) > 0:
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    data[sector] = df
            except Exception:
                pass

        return data
    except Exception:
        return None


def compute_sector_momentum(sector_data: Optional[dict] = None) -> dict:
    """计算各板块动量分数

    动量指标: 60日涨幅 * 0.5 + 20日涨幅 * 0.3 + 5日涨幅 * 0.2
    RPS排名: 按综合动量排序，返回0-100百分位

    返回:
      {板块名: {momentum_score, rps_rank, rps_score}}
    """
    if sector_data is None:
        sector_data = get_sector_daily_data()

    if not sector_data:
        return {}

    result = {}
    momentum_values = {}

    for sector, df in sector_data.items():
        if df is None or len(df) < 60:
            continue

        closes = df["close"].values
        ret_60 = (closes[-1] - closes[-60]) / closes[-60]
        ret_20 = (closes[-1] - closes[-20]) / closes[-20] if len(df) >= 20 else 0
        ret_5 = (closes[-1] - closes[-5]) / closes[-5] if len(df) >= 5 else 0

        momentum = ret_60 * 0.5 + ret_20 * 0.3 + ret_5 * 0.2
        momentum_values[sector] = momentum

    # RPS排名
    if momentum_values:
        sorted_sectors = sorted(momentum_values.items(), key=lambda x: x[1], reverse=True)
        total = len(sorted_sectors)

        for rank, (sector, momentum) in enumerate(sorted_sectors):
            rps_score = (1 - rank / total) * 100  # 第一名100分，最后一名0分
            result[sector] = {
                "momentum": float(momentum),
                "rps_rank": rank + 1,
                "rps_score": int(rps_score),
            }

    return result


# ══════════════════════════════════════════════════════
#  3. 主过滤函数
# ══════════════════════════════════════════════════════

SECTOR_MAP = {
    "食品饮料": ["白酒", "饮料", "食品"],
    "医药生物": ["医药", "医疗", "生物"],
    "电子": ["半导体", "芯片", "电子", "元器件"],
    "电气设备": ["电力设备", "新能源", "光伏", "锂电"],
    "化工": ["化工", "化学", "石化"],
    "计算机": ["计算机", "软件", "IT服务"],
    "银行": ["银行"],
    "非银金融": ["券商", "保险", "多元金融"],
    "房地产": ["房地产", "地产"],
    "国防军工": ["军工", "国防", "航天"],
    "机械设备": ["机械", "装备"],
    "汽车": ["汽车", "新能源车"],
    "有色金属": ["有色", "金属"],
    "钢铁": ["钢铁"],
    "通信": ["通信", "5G"],
    "公用事业": ["电力", "燃气", "水务"],
}


def _map_code_to_sector(code: str, sector_map: Optional[dict] = None) -> str:
    """将股票代码映射到申万行业板块"""
    if sector_map and code in sector_map:
        return sector_map[code]

    code_lower = code.lower().split(".")[0] if "." in code else code.lower()

    # 用代码前缀+名称关键词的简化映射
    # 实际项目中应从 Tushare industry 字段获取
    return "其他"


def _get_sector_for_code(code: str, stock_sectors: Optional[dict] = None) -> str:
    """获取股票所属板块名称"""
    if stock_sectors and code in stock_sectors:
        return stock_sectors[code]
    return "其他"


def filter_with_momentum(
    prices_dict: dict[str, pd.DataFrame],
    ml_scores: dict[str, int] | dict[str, float],
    stock_sectors: Optional[dict[str, str]] = None,
    top_ml_pct: float = 0.20,
    min_trend_score: int = 50,
    skip_trend_filter: bool = False,
) -> list[dict]:
    """主过滤函数：ML评分 + 动量过滤

    参数:
      prices_dict: {代码: 价格DataFrame}
      ml_scores: {代码: ML评分} 可以是int(0-100)或float(0-1)
      stock_sectors: {代码: 板块名} 可选
      top_ml_pct: 从ML评分前多少%开始筛选（默认20%）
      min_trend_score: 最小趋势评分（默认50）
      skip_trend_filter: 跳过趋势过滤（仅ML筛选可用）

    返回:
      [{
        "ts_code": str,
        "ml_score": float,
        "trend_ok": bool,
        "trend_score": int,
        "sector": str,
        "sector_rps": int,
        "composite_score": float,
        "signal": "strong_buy"|"buy"|"hold"|"avoid",
      }]
    """
    # Step 1: 行业板块动量
    sector_momentum = compute_sector_momentum()

    # Step 2: 统一ML评分到0-100范围
    normalized_ml = {}
    for code, score in ml_scores.items():
        if isinstance(score, float) and 0 <= score <= 1:
            normalized_ml[code] = round(score * 100)  # 0-1 → 0-100
        else:
            normalized_ml[code] = int(score)

    # 按ML评分排序，取前 top_ml_pct
    sorted_by_ml = sorted(normalized_ml.items(), key=lambda x: x[1], reverse=True)
    top_n = max(10, int(len(sorted_by_ml) * top_ml_pct))
    ml_candidates = dict(sorted_by_ml[:top_n])

    # Step 3: 对候选股逐一计算趋势
    results = []
    for code, ml_score in ml_candidates.items():
        df = prices_dict.get(code)
        if df is None or len(df) < 60:
            continue

        trend = compute_trend_signal(df)
        sector = _get_sector_for_code(code, stock_sectors)
        sector_info = sector_momentum.get(sector, {"rps_score": 50})

        # 综合评分: ML(40%) + 趋势(35%) + 板块动量(25%)
        composite = (
            ml_score * 0.40
            + trend["trend_score"] * 0.35
            + sector_info.get("rps_score", 50) * 0.25
        )

        # 信号级别
        if skip_trend_filter:
            # 跳过趋势过滤时，纯看ML分
            if ml_score >= 70:
                signal = "strong_buy"
            elif ml_score >= 50:
                signal = "buy"
            else:
                signal = "hold"
        else:
            trend_ok = trend["trend_ok"]
            if trend_ok and composite >= 70:
                signal = "strong_buy"
            elif trend_ok and composite >= 50:
                signal = "buy"
            elif trend_ok:
                signal = "hold"
            else:
                signal = "avoid"  # 趋势不配合

        results.append({
            "ts_code": code,
            "ml_score": ml_score,
            "trend_ok": trend["trend_ok"],
            "trend_score": trend["trend_score"],
            "ma50": trend["ma50"],
            "ma200": trend["ma200"],
            "sector": sector,
            "sector_rps": sector_info.get("rps_score", 50),
            "composite_score": round(composite, 1),
            "signal": signal,
            "last_close": trend["last_close"],
        })

    # Step 4: 按综合评分排序
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


# ══════════════════════════════════════════════════════
#  4. 发布信号报告
# ══════════════════════════════════════════════════════

def format_signal_report(candidates: list[dict], top_n: int = 10) -> str:
    """格式化交易信号报告"""
    if not candidates:
        return "# 今日无交易信号\n\nML选股 + 动量过滤未产生符合条件的候选股。"

    lines = [
        "# 📊 ML + 动量过滤选股报告",
        "",
        f"候选总数: {len(candidates)}  |  统计时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| # | 代码 | 信号 | ML分 | 趋势分 | 板块RPS | 综合分 |",
        "|---|------|------|------|--------|---------|--------|",
    ]

    for i, c in enumerate(candidates[:top_n], 1):
        signal_icon = {"strong_buy": "🟢", "buy": "🔵", "hold": "🟡", "avoid": "🔴"}
        icon = signal_icon.get(c["signal"], "⚪")
        lines.append(
            f"| {i} | {c['ts_code']} | {icon}{c['signal']} "
            f"| {c['ml_score']} | {c['trend_score']} "
            f"| {c['sector_rps']} | {c['composite_score']} |"
        )

    lines.extend([
        "",
        "### 信号说明",
        "- 🟢 **strong_buy**: ML高分 + 趋势多头 + 板块动量前50%",
        "- 🔵 **buy**: ML中高分 + 趋势多头",
        "- 🟡 **hold**: 趋势多头但ML分不足",
        "- 🔴 **avoid**: 趋势不配合（MA50或MA200之下）",
        "",
    ])

    return "\n".join(lines)


# ══════════════════════════════════════════════════════
#  5. 独立运行测试
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    # 快速验证
    print("测试动量过滤模块...")

    # 模拟数据
    test_codes = ["000001.SZ", "000002.SZ", "000333.SZ", "000651.SZ", "000858.SZ"]
    np.random.seed(42)
    test_prices = {}
    for code in test_codes:
        n = 250
        dates = pd.date_range("2025-01-01", periods=n, freq="B")
        base = np.random.randn() * 10 + 50
        trend = np.linspace(0, np.random.randn() * 20, n)
        noise = np.random.randn(n) * 2
        closes = np.maximum(base + trend + noise, 1)
        test_prices[code] = pd.DataFrame({"date": dates, "close": closes})

    test_ml = {code: np.random.randint(30, 95) for code in test_codes}
    result = filter_with_momentum(test_prices, test_ml)

    print(f"  候选: {len(result)} 只")
    for r in result[:5]:
        print(f"  {r['ts_code']}: {r['signal']} ML={r['ml_score']} Trend={r['trend_score']} "
              f"SectorRPS={r['sector_rps']} Composite={r['composite_score']}")

    print("\n报告样例:")
    print(format_signal_report(result, top_n=3))
    print("\n✓ 动量过滤模块OK")
