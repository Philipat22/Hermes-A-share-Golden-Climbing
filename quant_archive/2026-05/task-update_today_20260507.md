# 数据更新脚本构建任务

## 目标
根据Hermes的spec，构建 `update_today.py` 完成每日数据增量更新，补全缓存到当天。

## 交付物
- **核心脚本**: `D:\AIHedgeFund\ai-hedge-fund-main\src\signals\update_today.py`
- **v5.2验证报告**: `D:\AIHedgeFund\ai-hedge-fund-main\quant_archive\2026-05\daily_picks_20260507_2139.md`

## 完成状态

| 缓存文件 | 最新日期 | 数据量 |
|---------|---------|-------|
| prices_full.pkl | 2026-05-07 | 5516只股票, 301,503条 |
| macro_north_flow.pkl | 2026-05-07 | 1,957行 |
| fundamentals_daily.pkl | 2026-05-07 | 605,493行 |

v5.2全流程耗时: 9秒（之前151秒，无新API调用，纯缓存命中）

## 方法论备注（给Hermes）
1. **按天批量拉取代按股拉取**：`pro.daily(trade_date=dt)` 单次返回全市场~5000只，比按1908只逐个拉取高效10倍以上
2. **首次构建65次API调用**（~67s），日后增量仅1-3次API调用（<3s）
3. **GBK编码限制**：PowerShell下中文字符/emoji导致print崩溃，需纯ASCII输出
4. **pandas类型处理**：datetime64→str转换需严格统一，避免`isin()`/`sort_values()`的类型混搭崩溃
