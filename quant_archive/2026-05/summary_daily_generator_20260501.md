# 每日信号生成系统完成

## 目标
搭建主升浪策略的每日自动出票流水线：批量拉价格 → 因子计算 → ML预测 → 形态扫描 → 情绪融合 → 排名报告。

## 变更文件

### 新增
- **`src/signals/daily_generator.py`** (13KB) — 全自动流水线主入口
  - `daily_generate()` — 一键执行所有步骤
  - `fetch_all_prices()` — 并行拉取300只股票价格 (~15s)
  - `_compute_ml_scores_cached()` — 163因子+LightGBM预测，带缓存 (~14min首次)
  - `_fuse_emotion()` — 情绪融合增强Top候选
  - `_write_picks_report()` — 生成Markdown选股报告

### 修改
- **`src/surge/scanner.py`** — 新增 `ml_scores` + `prices_dict` 参数
  - 支持传入预计算ML评分（避免逐只股票算因子）
  - 支持传入预拉取价格数据（避免重复网络请求）

## 流水线流程
```
stock_pool(300只) → fetch_all_prices(并行, 15s)
                  → build_dataset(163因子, ~14min)
                  → LightGBM predict(1s)
                  → scan_market(形态识别, 2min)
                  → emotion fusion(30s)
                  → daily_picks_YYYYMMDD.md
                  Total: ~17min
```

## 定时任务
- 已设置 cron: **交易日15:10** 自动运行 (`1d113310`)
- 输出到 `quant_archive/YYYY-MM/daily_picks_*.md`
- cron完成后在会话中总结Top-3选股结果

## E2E验证结果
- 2只测试股票成功跑通全流程
- 茅台/五粮液均检测到W底形态（score 26-29）
- 情绪融合接口调通（analyze_emotion）
- ML模型加载正常（160 features, AUC 0.6519）
- 报告写入UTF-8正确

## 明天的首次运行
- 首次会做全量因子计算 (~14min)，结果缓存供当天复用
- 第300只全量跑需等因子计算完成
- 可配置 `--max-stocks 100` 加速首次测试
