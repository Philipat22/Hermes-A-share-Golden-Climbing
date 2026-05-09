# AI Hedge Fund — A股量化投研系统

三条腿全景选股：北向择时 + v5.2大师共识 + 黄金坑趋势回调

## 策略概览

| 策略 | 市场状态 | 逻辑 | 状态 |
|------|---------|------|------|
| v5.2 大师共识 | CSI300 < MA60 | 暴跌抄底,均值回归 | ✅ 成熟 |
| 北向择时 | 全周期 | 宏观仓位调节 | ✅ 成熟 |
| 黄金坑 | CSI300 > MA60 | 趋势回调,牛回头上车 | ✅ 定稿 |

## 黄金坑策略 (2026-05-08 定稿)

```
选票: 趋势得分4-5/5 + C1(60日>P60) + C4(涨幅>跌幅×2) + 非ST
闸门: CSI300 > MA60
过滤: 跌速>0.5%/天 + 量比<3x + 深度≤-18% + 5天内触达-10%
进场: 跌到-10%, T+1买入
出场: 持有50-60天到期平仓
```

- 策略定稿: `quant_archive/2026-05/golden_pit_final_spec.md`
- 可复现脚本: `scripts/golden_pit_verify.py`

## 数据

数据缓存文件较大，使用 Git LFS 管理：
- `data/cache/prices_full.pkl` (~590MB) — 全量日线
- `data/cache/golden_pit_batch_results.pkl` (~13MB) — 批次回测

或通过 `scripts/rebuild_data.py` 从 tushare 重建。

## 安装

```bash
pip install pandas numpy tushare
```

## 验证

```bash
python scripts/golden_pit_verify.py
```
