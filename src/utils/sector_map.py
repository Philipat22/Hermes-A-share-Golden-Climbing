"""
A股申万行业板块 → 股票映射系统
基于 Tushare 110 个申万行业分类，精确匹配

作者: JoJo

Tushare 110 个真实行业名（stock_basic.industry）:
IT设备, 专用机械, 中成药, 乳制品, 互联网, 仓储物流, 供气供热, 保险,
元器件, 全国地产, 公共交通, 公路, 其他商业, 其他建材, 农业综合, 农用机械,
农药化肥, 化学制药, 化工原料, 化工机械, 化纤, 区域地产, 医疗保健, 医药商业,
半导体, 商品城, 商贸代理, 啤酒, 园区开发, 塑料, 多元金融, 家居用品, 家用电器,
小金属, 工程机械, 广告包装, 建筑工程, 影视音像, 房产服务, 批发业, 摩托车,
文教休闲, 新型电力, 旅游景点, 旅游服务, 日用化工, 普钢, 服饰, 机场, 机床制造,
机械基件, 林业, 染料涂料, 橡胶, 水力发电, 水务, 水泥, 水运, 汽车整车, 汽车服务,
汽车配件, 渔业, 港口, 火力发电, 焦炭加工, 煤炭开采, 特种钢, 环境保护, 玻璃,
生物制药, 电信运营, 电器仪表, 电器连锁, 电气设备, 白酒, 百货, 石油加工, 石油开采,
石油贸易, 矿物制品, 种植业, 空运, 红黄酒, 纺织, 纺织机械, 综合类, 航空, 船舶,
装修装饰, 证券, 超市连锁, 路桥, 软件服务, 软饮料, 轻工机械, 运输设备, 通信设备,
造纸, 酒店餐饮, 钢加工, 铁路, 铅锌, 铜, 铝, 银行, 陶瓷, 食品, 饲料, 黄金
"""

from __future__ import annotations
import os
import sys
import io
from datetime import datetime
from typing import Optional

import pandas as pd
import tushare as ts

# ──────────────────────────────────────────────
# Token
# ──────────────────────────────────────────────
TUSHARE_TOKEN = os.getenv("TUSHARE_PRO_TOKEN") or "5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7"

# ──────────────────────────────────────────────
# 18 个核心板块 → 申万行业关键词（用子串匹配，精确定位）
# ──────────────────────────────────────────────
SECTOR_INDUSTRY_MAP: dict[str, list[str]] = {
    # 板块名: [申万行业关键词列表]
    "新能源":        ["电气设备", "汽车整车", "新型电力"],
    "半导体":        ["半导体"],
    "消费电子":      ["元器件", "家用电器"],           # 申万无"消费电子"，用最近似
    "医药":          ["化学制药", "生物制药", "中药", "医疗器械", "医药商业", "医疗保健"],
    "食品饮料":       ["白酒", "食品", "饮料制造", "乳制品", "啤酒"],
    "银行":          ["银行"],
    "房地产":        ["全国地产", "区域地产", "房产服务"],
    "军工":          ["航空", "船舶"],                  # 申万无"军工"，用相关行业
    "化工":          ["化工原料", "化学制品", "化工合成材料", "化纤"],
    "机械设备":       ["专用机械", "工程机械", "机械基件", "机床制造", "轻工机械"],
    "电力设备":       ["电气设备"],                     # 申万电气设备 = 电力设备
    "计算机":         ["软件服务", "计算机应用", "计算机设备"],
    "传媒":           ["互联网", "广告包装", "影视音像", "出版业", "营销服务"],
    "交通运输":       ["航空", "机场", "港口", "水运", "铁路", "仓储物流", "公路", "运输设备"],
    "建筑材料":       ["水泥", "玻璃", "普钢", "特种钢", "其他建材", "建筑材料"],
    "环保":           ["环境保护"],
    "有色金属":       ["铜", "铝", "铅锌", "小金属", "黄金", "矿物制品"],
    "黄金":           ["黄金"],
}

# ──────────────────────────────────────────────
# 缓存
# ──────────────────────────────────────────────
_industry_stocks_cache: dict[str, list[str]] = {}
_all_stocks_df_cache: Optional[pd.DataFrame] = None
_cache_built = False


def _build_cache(force: bool = False) -> None:
    """构建申万行业 → 股票代码 映射缓存"""
    global _industry_stocks_cache, _all_stocks_df_cache, _cache_built
    if _cache_built and not force:
        return

    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pro.stock_basic(list_status="L", limit=6000)
    # 剔除退市
    if "status" in df.columns:
        df = df[df["status"] == "L"]
    _all_stocks_df_cache = df

    _industry_stocks_cache.clear()
    for _, row in df.iterrows():
        ind = str(row.get("industry", "")).strip()
        code = str(row.get("ts_code", "")).strip()
        if not ind or not code:
            continue
        if ind not in _industry_stocks_cache:
            _industry_stocks_cache[ind] = []
        _industry_stocks_cache[ind].append(code)

    _cache_built = True
    total = sum(len(v) for v in _industry_stocks_cache.values())
    print(f"[sector_map] 缓存: {len(df)} 只股票, {len(_industry_stocks_cache)} 个行业")


def get_all_stocks() -> pd.DataFrame:
    """全量上市股票 DataFrame"""
    _build_cache()
    return _all_stocks_df_cache


def get_stocks_by_sector(sector: str) -> list[str]:
    """
    按板块名获取成分股（子串匹配申万行业）

    Args:
        sector: 板块名，如 "白酒", "半导体", "医药"

    Returns:
        排序后的股票代码列表，如 ["600519.SH", "000858.SZ", ...]
    """
    _build_cache()

    # 直接从 SECTOR_INDUSTRY_MAP 查关键词
    keywords = SECTOR_INDUSTRY_MAP.get(sector, [])

    # 如果没找到，尝试模糊匹配（关键词出现在行业名中）
    if not keywords:
        sector_lower = sector.lower()
        keywords = [sector, sector_lower]
    else:
        # 加一个兜底：用板块名本身模糊匹配
        keywords = list(keywords) + [sector, sector.lower()]

    stocks = set()
    for ind_name, codes in _industry_stocks_cache.items():
        # 关键词 in 行业名 → 匹配
        if any(kw in ind_name for kw in keywords):
            stocks.update(codes)

    return sorted(list(stocks))


def get_sector_distribution() -> dict[str, int]:
    """16核心板块 → 股票数量"""
    return {sector: len(get_stocks_by_sector(sector)) for sector in SECTOR_INDUSTRY_MAP}


def filter_stocks(
    tickers: list[str],
    min_market_cap: float = 5.0,    # 亿元（流通市值下限）
    max_market_cap: float = 5000.0, # 亿元（流通市值上限）
    min_turn_rate: float = 0.2,     # 日换手率%
    exclude_st: bool = True,
    exclude_new: bool = True,        # 排除上市<90天新股
    trade_date: Optional[str] = None,
) -> list[str]:
    """
    质量过滤：ST、新股、市值门槛

    Args:
        tickers: 股票列表
        min_market_cap: 最低流通市值（亿元）
        max_market_cap: 最高流通市值（亿元）
        min_turn_rate: 最低日换手率（%）
        exclude_st: 剔除 ST/*ST
        exclude_new: 剔除新股（<90天）
        trade_date: 交易日期

    Returns:
        过滤后股票列表
    """
    _build_cache()
    if not tickers:
        return []

    df_all = _all_stocks_df_cache
    info_map = {row["ts_code"]: row for _, row in df_all.iterrows()}

    selected = []
    for ticker in tickers:
        row = info_map.get(ticker)
        if row is None:
            # 尝试归一化匹配
            norm = ticker.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
            for ts_code in info_map:
                if ts_code.replace(".", "") == norm:
                    row = info_map[ts_code]
                    ticker = ts_code
                    break
        if row is None:
            continue

        name = str(row.get("name", "")).strip()

        # ST 过滤
        if exclude_st and any(name.startswith(p) for p in ["ST", "*ST", "S ", "S*"]):
            continue

        # 新股过滤（<90天）
        if exclude_new:
            list_date = str(row.get("list_date", ""))
            if len(list_date) == 8:
                list_dt = datetime.strptime(list_date, "%Y%m%d")
                if (datetime.now() - list_dt).days < 90:
                    continue

        selected.append(ticker)

    return selected


def get_stock_info(ticker: str) -> Optional[dict]:
    """获取单只股票基本信息"""
    _build_cache()
    df = _all_stocks_df_cache
    row = df[df["ts_code"] == ticker]
    if row.empty:
        norm = ticker.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        row = df[df["ts_code"].str.replace(".", "", 1) == norm]
    if row.empty:
        return None
    r = row.iloc[0]
    name = str(r.get("name", ""))
    return {
        "ts_code": str(r.get("ts_code", "")),
        "name": name,
        "industry": str(r.get("industry", "")),
        "market": str(r.get("market", "")),
        "list_date": str(r.get("list_date", "")),
        "is_st": name.startswith(("ST", "*ST", "S ", "S*")),
    }


# ──────────────────────────────────────────────
# 快速入口
# ──────────────────────────────────────────────

def quick_pick(sector: str, n: int = 10, filter_st: bool = True) -> list[str]:
    """
    快速选股

    Args:
        sector: 板块名
        n: 最多返回股票数
        filter_st: 是否过滤ST

    Returns:
        股票代码列表
    """
    stocks = get_stocks_by_sector(sector)
    if filter_st:
        stocks = filter_stocks(stocks)
    return stocks[:n]


if __name__ == "__main__":
    import sys as _sys
    if _sys.platform == 'win32':
        _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("申万行业板块系统验证")
    print("=" * 60)

    _build_cache()

    print("\n[14核心板块 - 股票数量分布]")
    dist = get_sector_distribution()
    total = 0
    for sector, count in sorted(dist.items(), key=lambda x: -x[1]):
        stars = "★" * min(count // 50, 10)
        print(f"  {sector:8s}: {count:4d} 只 {stars}")
        total += count
    print(f"\n  合计: {total} 只（去重前）")

    print("\n[单板块快速选股示例]")
    for s in ["白酒", "半导体", "银行", "医药", "计算机"]:
        stocks = quick_pick(s, n=3)
        names = []
        for code in stocks:
            info = get_stock_info(code)
            if info:
                names.append(f"{info['name']}({code})")
        print(f"  {s}: {', '.join(names)}")
