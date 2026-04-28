"""
API Wrapper for A-Share (A股) Market Data
==========================================
本文件替换原 Financial Datasets API，改用 Tushare Pro + AKShare 获取A股数据。
保留原接口函数签名，agents无需任何修改。

作者: JoJo (华尔街金融大咖)
"""

# ============================================================
# 重新导出 A股数据接口（与原 api.py 接口完全兼容）
# ============================================================

from src.tools.a_stock_api import (
    # 核心数据
    get_prices,
    get_price_data,
    prices_to_df,
    PriceBar,
    # 财务数据
    get_financial_metrics,
    FinancialMetrics,
    # 新闻
    get_company_news,
    NewsItem,
    # 融资融券（替代insider trades）
    get_insider_trades,
    MarginTrade,
    # 财务科目搜索
    search_line_items,
    # 市值
    get_market_cap,
    get_float_market_cap,
    # 涨跌停
    get_limit_list,
    LimitUpData,
    # 北向资金
    get_north_money,
    # 股票信息
    get_stock_info,
    AStockInfo,
    # 过滤
    filter_st_stocks,
    normalize_ts_code,
)

# ============================================================
# 兼容性别名（原接口类型名称）
# ============================================================
Price = PriceBar  # 别名兼容

# ============================================================
# 环境变量说明
# ============================================================
"""
设置以下环境变量（已在 .env 中配置）:

TUSHARE_PRO_TOKEN=你的Tushare Pro Token
DEEPSEEK_API_KEY=你的DeepSeek API Key

注意：不再需要 FINANCIAL_DATASETS_API_KEY（已替换为A股数据源）
"""
