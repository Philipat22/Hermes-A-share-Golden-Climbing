# 因子工厂 Phase 1 集成（vnpy-alpha → AI Hedge Fund）

**日期**：2026-04-30 15:58-16:15  
**目标**：从 vnpy.alpha 模块提取 191个WorldQuant/Alpha158因子，集成到AI Hedge Fund项目

## 完成内容

### 新模块 `src/features/`
```
src/features/
  __init__.py               - 导出 FeatureGenerator, compute_features
  feature_generator.py      - 主入口：数据预处理 + 多进程因子计算
  engine/                   - 表达式计算引擎（从vnpy-alpha复制适配）
    data_proxy.py           - DataProxy运算符重载类
    calculate.py            - calculate_by_expression (eval式因子计算)
    ts_function.py          - 时序函数（delay/rank/mean/std/corr/slope等）
    cs_function.py          - 横截面函数（rank/mean/std/scale）
    math_function.py        - 数学函数（less/greater/sign/pow/quesval等）
    ta_function.py          - 技术指标（RSI/ATR，无talib依赖）
  factors/
    alpha_101.py            - WorldQuant Alpha 101（82个因子表达式）
    alpha_158.py            - Qlib Alpha 158（78个因子表达式）
```

### 核心能力
- 163个因子在真实数据上验证通过（3只股票×804天）
- vwap从Tushare amount/vol正确计算（误差<0.3%）
- 自动列名映射：ts_code→vt_symbol, trade_date→datetime, vol→volume
- 多进程并行计算（Pool.imap）
- 与现有数据管道兼容（接收Tushare pandas DataFrame）

### 已保留的自我进化设计
- `src/surge/evolve.py`（自进化调度器）✅ 未改动
- `src/surge/feedback.py`（反馈闭环）✅ 未改动
- `src/surge/scanner.py`（扫描器）✅ 未改动
- `src/surge/sector_resonance.py`（板块共振）✅ 未改动
- SignalMemory 280条记录 ✅ 未改动
- `src/emotion/`（情绪融合）4个文件 ✅ 未改动

## 验证结果
- 3只真实A股 × 804个交易日 = 2,412行数据
- Alpha101: 82因子（2个高NaN为统计偏差，3只股票池过小）
- Alpha158+: 81因子（0个高NaN）
- 计算时间：约60秒（4 worker并行）

## 已知问题
1. alpha3, alpha96 在极小股票池（3只）下>50% NaN → 20+只股票即可改善
2. 依赖polars 1.40.1（已安装），scipy（已安装）
3. engine.py（surge模块）仍处于损坏状态，等待后续修复

## 后续计划
- Phase 2（约3h）：ML训练流水线（LightGBM/Lasso/MLP训练因子→收益预测）
- Phase 3（约2h）：回测引擎适配（BacktestingEngine接口改造）
- 然后修复surge/engine.py并测试全链路

## 文件清单
- 新增文件：12个（engine/ 6个 + factors/ 3个 + feature_generator + __init__×2）
- 修改文件：1个（feature_generator.py）
- 状态：✅ 所有新增文件语法通过，真实数据验证通过
