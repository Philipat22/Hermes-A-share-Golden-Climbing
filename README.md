# A股智能投研系统

基于 DeepSeek + LangGraph + 18位投资大师 Agent 的 A股分析系统。

## 项目来源

基于 [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)，已针对 A股市场深度改造。

## 核心功能

- **数据源**：Tushare Pro + AKShare（支持 K线、财务数据、北向资金、舆情等）
- **板块选股**：申万行业分类系统（16大板块 × 110个行业）
- **多 Agent 并行分析**：基本面 + 技术面 + 舆情 + 风控
- **投资大师**：Warren Buffett、Charlie Munger、Ben Graham、Michael Burry、Cathie Wood 等18位风格
- **风控引擎**：波动率 × 相关性 → 动态仓位上限

## 快速开始

```bash
# 1. 安装依赖
poetry install

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key 和 Tushare Token

# 3. 运行分析
poetry run python src/main_astock.py --sector 白酒

# 指定大师
poetry run python src/main_astock.py --sector 半导体 --analysts warren_buffett charlie_munger
```

## 目录结构

```
src/
  tools/
    a_stock_api.py      ← A股数据接口（Tushare + AKShare）
  utils/
    sector_map.py       ← 申万行业板块映射
  agents/
    fundamentals.py      ← 基本面分析 Agent
    technicals.py        ← 技术面分析 Agent
    sentiment.py         ← 舆情分析 Agent
    risk_manager.py      ← 风控 Agent
    portfolio_manager.py ← 组合管理 Agent
  graph/
    state.py             ← LangGraph 状态管理
  main_astock.py         ← A股主入口
```

## 数据源说明

| 数据类型 | 来源 | 接口 |
|----------|------|------|
| A股日线 | Tushare Pro | `pro.daily()` |
| 财务指标 | Tushare Pro | `pro.fina_indicator()` |
| 北向资金 | Tushare Pro | `pro.moneyflow_hsgt()` |
| 舆情新闻 | Tushare Pro | `pro.news()` |
| 美股/期货 | AKShare | `ak.index_us_stock_sina()` |
