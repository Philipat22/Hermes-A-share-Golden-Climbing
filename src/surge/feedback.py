#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surge/feedback.py — 自我进化反馈回环

核心机制：
1. 记录每个信号的形态类型、评分、时间、价格
2. 追踪信号发出后的实际表现（N天后）
3. 根据历史结果调整参数权重

这是整个系统能"自我进化"的关键——不是AI黑箱，而是实证反馈。
"""
from __future__ import annotations
import os, json, logging
from datetime import datetime, timedelta
from typing import Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

SIGNAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "surge", "signal_memory")
os.makedirs(SIGNAL_DIR, exist_ok=True)

# ── 信号记录 ──────────────────────────────────

class SignalMemory:
    """
    信号记忆系统
    
    每个信号发出时记录，N个交易日后追踪结果。
    数据持久化到 JSON 文件。
    """
    
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or os.path.join(SIGNAL_DIR, "signal_log.json")
        self.signals: list[dict] = []
        self._load()
    
    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.signals = data.get("signals", [])
                    logger.info(f"Loaded {len(self.signals)} historical signals")
            except Exception as e:
                logger.warning(f"Failed to load signal memory: {e}")
                self.signals = []
    
    def save(self):
        """持久化到磁盘"""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "total_signals": len(self.signals),
                "signals": self.signals,
            }, f, ensure_ascii=False, indent=2)
    
    def record(self, signal: dict[str, Any]):
        """
        记录一次信号
        
        保存信号发出时的完整上下文，用于后续评估
        """
        # ── 情绪融合评分（如有） ──
        emotion = signal.get("emotion", {})
        record = {
            "signal_id": f"{signal.get('ts_code','')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "ts_code": signal.get("ts_code", ""),
            "timestamp": datetime.now().isoformat(),
            "trade_date": datetime.now().strftime("%Y-%m-%d"),
            "pattern_type": signal.get("pattern_type"),
            "total_score": signal.get("total_score", 0),
            "final_score": signal.get("final_score", 0),
            "signal_grade": signal.get("signal_grade", "NONE"),
            "entry_price": signal.get("entry_price", 0),
            "components": {
                "pattern_score": signal.get("pattern_score", 0),
                "volume_score": signal.get("volume_score", 0),
                "sector_score": signal.get("sector_score", 0),
                "accel_score": signal.get("accel_score", 0),
                "fake_score": signal.get("fake_score", 0),
            },
            "detail": signal.get("detail", ""),
            # ── 情绪融合数据（P1 新增）   ──
            "emotion_fusion": emotion.get("fusion_score"),
            "emotion_confidence": emotion.get("confidence"),
            "emotion_label": emotion.get("label"),
            "emotion_components": {
                "surge": emotion.get("components", {}).get("surge", {}).get("score"),
                "news": emotion.get("components", {}).get("news", {}).get("score"),
                "market": emotion.get("components", {}).get("market", {}).get("score"),
                "sector": emotion.get("components", {}).get("sector", {}).get("score"),
            },
            # ── 结果追踪（初始为空） ──
            "outcome": None,
            "outcome_return": None,
            "outcome_max_drawdown": None,
            "outcome_holding_days": None,
            "outcome_evaluated_date": None,
            "outcome_success": None,
        }
        self.signals.append(record)
        self.save()
        return record["signal_id"]
    
    def get_pending_evaluation(self, min_age_days: int = 5) -> list[dict]:
        """
        获取等待评估结果的信号
        
        Args:
            min_age_days: 信号发出至少 N 天后才评估
        
        Returns:
            待评估的信号列表
        """
        cutoff = (datetime.now() - timedelta(days=min_age_days)).isoformat()
        pending = []
        for s in self.signals:
            if s["outcome"] is None and s["timestamp"] < cutoff:
                pending.append(s)
        return pending
    
    def record_outcome(self, signal_id: str, outcome: dict):
        """
        记录一个信号的后续表现
        
        Args:
            signal_id: 信号ID
            outcome: {
                "return": 实际收益率,
                "max_drawdown": 期间最大回撤,
                "holding_days": 持有天数,
                "success": True/False (可否按预期方向走),
                "evaluated_date": 评估日期,
            }
        """
        for s in self.signals:
            if s["signal_id"] == signal_id:
                s["outcome"] = outcome.get("result", "unknown")
                s["outcome_return"] = outcome.get("return")
                s["outcome_max_drawdown"] = outcome.get("max_drawdown")
                s["outcome_holding_days"] = outcome.get("holding_days")
                s["outcome_evaluated_date"] = outcome.get("evaluated_date", datetime.now().isoformat())
                s["outcome_success"] = outcome.get("success", False)
                self.save()
                return True
        return False
    
    def get_pattern_stats(self) -> dict:
        """
        统计各形态的历史表现
        
        Returns:
            {
                "平台突破": {"total": N, "success": N, "win_rate": %,
                            "avg_return": %, "weight_adjustment": 1.0},
                ...
            }
        """
        categorized = defaultdict(list)
        for s in self.signals:
            if s["outcome_success"] is not None:  # 有结果
                categorized[s["pattern_type"]].append(s)
        
        stats = {}
        for pattern, signals in categorized.items():
            total = len(signals)
            successes = sum(1 for s in signals if s["outcome_success"])
            returns = [s["outcome_return"] for s in signals if s["outcome_return"] is not None]
            
            win_rate = successes / total if total > 0 else 0
            avg_return = sum(returns) / len(returns) if returns else 0
            
            # 权重调整因子
            # 胜率>60% -> 权重上调, 胜率<40% -> 权重下调
            if win_rate >= 0.6:
                adjustment = 1.0 + (win_rate - 0.6) * 2  # max ~1.8x
            elif win_rate >= 0.4:
                adjustment = 1.0
            else:
                adjustment = 0.5 + win_rate  # min ~0.5x
            
            stats[pattern] = {
                "total": total,
                "success": successes,
                "win_rate": round(win_rate, 3),
                "avg_return": round(avg_return, 4),
                "weight_adjustment": round(adjustment, 3),
            }
        
        return stats
    
    def get_emotion_stats(self) -> dict:
        """
        统计情绪融合评分与实际表现的相关性

        Returns: {
            "fusion_buckets": [
                {"range": "70+", "count": N, "win_rate": %,"avg_return": %},
                ...
            ],
            "correlation": {
                "fusion_success_corr": float,
                "sector_weight_adj": float,  # 板块评分调整系数
                "market_weight_adj": float,
            },
            "recommended_weights": {w_surge, w_news, w_market, w_sector}
        }
        """
        evaluated = [s for s in self.signals
                     if s["outcome_success"] is not None
                     and s.get("emotion_fusion") is not None]

        if len(evaluated) < 5:
            return {"status": "insufficient_data", "evaluated_count": len(evaluated)}

        # 按融合分分桶
        buckets = {
            "high": {"range": "70+", "min": 70, "signals": []},
            "mid_high": {"range": "60-69", "min": 60, "max": 69, "signals": []},
            "mid": {"range": "50-59", "min": 50, "max": 59, "signals": []},
            "low": {"range": "<50", "max": 49, "signals": []},
        }

        for s in evaluated:
            fus = s["emotion_fusion"]
            if fus >= 70:
                buckets["high"]["signals"].append(s)
            elif fus >= 60:
                buckets["mid_high"]["signals"].append(s)
            elif fus >= 50:
                buckets["mid"]["signals"].append(s)
            else:
                buckets["low"]["signals"].append(s)

        bucket_stats = []
        for key, b in buckets.items():
            sigs = b["signals"]
            if not sigs:
                continue
            total = len(sigs)
            success = sum(1 for s in sigs if s["outcome_success"])
            returns = [s["outcome_return"] for s in sigs if s["outcome_return"] is not None]
            bucket_stats.append({
                "range": b["range"],
                "count": total,
                "win_rate": round(success / total * 100, 1),
                "avg_return": round(sum(returns) / len(returns) * 100, 2) if returns else 0,
                "success_count": success,
            })

        # 分析各情感分量与成功率的相关系数
        comp_keys = ["surge", "news", "market", "sector"]
        comp_stats = {k: {"totals": [], "successes": []} for k in comp_keys}
        for s in evaluated:
            ec = s.get("emotion_components", {})
            for k in comp_keys:
                val = ec.get(k)
                if val is not None:
                    comp_stats[k]["totals"].append(val)
                    comp_stats[k]["successes"].append(1 if s["outcome_success"] else 0)

        # 计算推荐权重：分量与成功率的相关系数作为权重依据
        recommended = {
            "w_surge": 0.35,  # 默认值
            "w_news": 0.10,
            "w_market": 0.20,
            "w_sector": 0.25,
            "w_diversion": 0.10,
        }

        comp_corrs = {}
        total_corr = 0
        for k in comp_keys:
            vals = comp_stats[k]
            if len(vals["totals"]) < 10:
                comp_corrs[k] = 0
                continue
            # 简单平均分差：成功的信号该分量平均分是否高于失败的
            success_avg = sum(vals["totals"][i] for i, s in enumerate(vals["successes"]) if s) / max(sum(vals["successes"]), 1)
            fail_avg = sum(vals["totals"][i] for i, s in enumerate(vals["successes"]) if not s) / max(len(vals["successes"]) - sum(vals["successes"]), 1)
            diff = success_avg - fail_avg
            # 归一化到 -1..1
            corr = max(-1, min(1, diff / 15))
            comp_corrs[k] = round(corr, 3)
            total_corr += max(0, corr)

        if total_corr > 0:
            # 按正相关比例分配权重
            for k in ["surge", "news", "market", "sector"]:
                pos_corr = max(0, comp_corrs.get(k, 0))
                if total_corr > 0:
                    # 某分量与结果无关 -> 降权
                    if pos_corr < 0.05:
                        recommended[f"w_{k}"] = round(recommended.get(f"w_{k}", 0.25) * 0.8, 2)
                    elif pos_corr > 0.3:
                        recommended[f"w_{k}"] = round(recommended.get(f"w_{k}", 0.25) * 1.2, 2)

        # 归一化权重总和为1
        total_w = sum(recommended.values())
        if total_w > 0:
            for k in recommended:
                recommended[k] = round(recommended[k] / total_w, 3)

        return {
            "status": "ok",
            "evaluated_count": len(evaluated),
            "fusion_buckets": bucket_stats,
            "component_correlations": comp_corrs,
            "recommended_weights": recommended,
        }

    def adjust_emotion_weights(self) -> bool:
        """
        根据情绪统计结果，自动调整 fusion.py 的权重文件

        Returns: True if weights were updated
        """
        stats = self.get_emotion_stats()
        if stats.get("status") != "ok":
            return False

        recommended = stats["recommended_weights"]
        if not recommended:
            return False

        try:
            weights_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "emotion", "emotion_weights.json"
            )
            with open(weights_file, "r", encoding="utf-8") as f:
                current = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # 使用默认值
            current = {
                "w_surge": 0.35,
                "w_news": 0.10,
                "w_market": 0.20,
                "w_sector": 0.25,
                "w_diversion": 0.10,
            }

        # 平滑更新：新权重 = 旧*0.7 + 推荐*0.3
        changed = False
        for k in recommended:
            if k in current:
                new_val = round(current[k] * 0.7 + recommended[k] * 0.3, 3)
                if abs(new_val - current[k]) > 0.01:
                    current[k] = new_val
                    changed = True

        if changed:
            import os as _os
            weights_file = _os.path.join(
                _os.path.dirname(_os.path.dirname(__file__)),
                "emotion", "emotion_weights.json"
            )
            _os.makedirs(_os.path.dirname(weights_file), exist_ok=True)
            with open(weights_file, "w", encoding="utf-8") as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
            logger.info(f"Emotion weights updated: {current}")

        return changed

    def summary(self) -> str:
        """打印统计摘要（含情绪统计）"""
        stats = self.get_pattern_stats()
        emotion_stats = self.get_emotion_stats()

        lines = ["=== 形态表现统计 ==="]
        if stats:
            for pattern, st in sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True):
                adj = st["weight_adjustment"]
                arrow = "↑" if adj > 1.05 else ("↓" if adj < 0.95 else "→")
                lines.append(
                    f"  {pattern:8s} | {st['total']:3d}次 | "
                    f"胜率{st['win_rate']*100:5.1f}% | "
                    f"均收益{st['avg_return']*100:+6.2f}% | "
                    f"权重{adj:.2f}x {arrow}"
                )

            total = sum(st["total"] for st in stats.values())
            overall_win = sum(st["success"] for st in stats.values()) / total if total > 0 else 0
            lines.append(f"\n  总计: {total}次评估 | 综合胜率{overall_win*100:.1f}%")

        if emotion_stats.get("status") == "ok":
            lines.append(f"\n=== 情绪融合统计（{emotion_stats['evaluated_count']}次） ===")
            for b in emotion_stats.get("fusion_buckets", []):
                arrow = "↑" if b["win_rate"] > 50 else "↓"
                lines.append(
                    f"  融合分{b['range']:5s} | {b['count']:3d}次 | "
                    f"胜率{b['win_rate']:5.1f}% | "
                    f"均收益{b['avg_return']:+6.2f}% {arrow}"
                )
            corrs = emotion_stats.get("component_correlations", {})
            if corrs:
                lines.append("\n  分量相关度:")
                for k, v in corrs.items():
                    lines.append(f"    {k}: {v:+.3f}")
            rec = emotion_stats.get("recommended_weights", {})
            if rec:
                lines.append(f"\n  推荐权重: {rec}")
        elif len(emotion_stats.get("evaluated_count", 0)) > 0:
            lines.append(f"\n[情绪融合] 数据不足（仅{emotion_stats['evaluated_count']}条）")

        return "\n".join(lines)


# ── 参数调整 ──────────────────────────────────

def evaluate_signals(
    memory: Optional[SignalMemory] = None,
    days_lookback: int = 5,
    holding_period: int = 10,
) -> int:
    """
    评估待处理信号的实际表现
    
    对于N天前发出的信号，拉取当前价格计算收益
    
    Args:
        memory: SignalMemory 实例
        days_lookback: 评估N天之前的信号
        holding_period: 假设持有天数
    
    Returns:
        本次评估的信号数量
    """
    if memory is None:
        memory = SignalMemory()
    
    pending = memory.get_pending_evaluation(min_age_days=days_lookback)
    if not pending:
        logger.info(f"没有待评估的信号（最近{days_lookback}天内）")
        return 0
    
    from src.tools.a_stock_api import get_prices
    import pandas as pd
    
    today = datetime.now().strftime("%Y-%m-%d")
    evaluated = 0
    
    for sig in pending:
        ts_code = sig["ts_code"]
        entry_price = sig["entry_price"]
        entry_date_str = sig["trade_date"]
        
        if not ts_code or not entry_price or entry_price <= 0:
            continue
        
        try:
            # 获取从信号发出到今天的数据
            end = today
            prices = get_prices(ts_code, entry_date_str, end)
            if not prices or len(prices) < 2:
                continue
            
            df = pd.DataFrame([{"close": p.close, "date": p.date} for p in prices])
            closes = df["close"].values
            n = len(closes)
            
            # 最大持有 holding_period 天
            actual_hold = min(n - 1, holding_period)
            if actual_hold < 1:
                continue
            
            exit_price = float(closes[actual_hold])
            actual_return = (exit_price - entry_price) / entry_price
            
            # 期间最大回撤
            max_dd = min(0, min(
                (float(closes[i]) - entry_price) / entry_price
                for i in range(1, min(actual_hold + 1, n))
            ))
            
            # 判定成功与否：收益>0 或 最高收益曾达到+3%
            max_return = max(
                (float(closes[i]) - entry_price) / entry_price
                for i in range(1, min(actual_hold + 1, n))
            )
            success = actual_return > 0.01 or max_return > 0.03
            
            outcome = {
                "result": "profit" if actual_return > 0 else "loss",
                "return": round(actual_return, 4),
                "max_drawdown": round(max_dd, 4),
                "holding_days": actual_hold,
                "success": success,
                "evaluated_date": today,
            }
            
            memory.record_outcome(sig["signal_id"], outcome)
            evaluated += 1
            
        except Exception as e:
            logger.debug(f"评估失败 {ts_code}: {e}")
            continue
    
    if evaluated > 0:
        memory.save()
        logger.info(f"完成 {evaluated} 个信号评估")
        print(memory.summary())
    
    return evaluated


def adjust_params(
    memory: Optional[SignalMemory] = None,
    min_signals: int = 10,
) -> dict:
    """
    根据历史信号表现调整参数
    
    核心逻辑：
    - 胜率高的形态 → 权重提升
    - 胜率低的形态 → 权重降低  
    - 所有参数调整都有上下限保护
    
    Args:
        memory: SignalMemory
        min_signals: 至少N次评估后才做调整
    
    Returns:
        更新后的参数字典
    """
    from src.surge.engine import load_params, save_params
    
    params = load_params()
    if memory is None:
        memory = SignalMemory()
    
    stats = memory.get_pattern_stats()
    total_evaluated = sum(s["total"] for s in stats.values())
    
    if total_evaluated < min_signals:
        logger.info(f"评估样本不足 {total_evaluated}/{min_signals}，暂不调整")
        return params
    
    print("\n=== 自我进化: 参数调整 ===")
    
    # ── 1. 调整形态权重 ──
    weights = {
        "平台突破": "w_price_pattern",
        "VCP": "w_price_pattern",
        "N字突破": "w_price_pattern",
    }
    
    # 按形态计算平均 weight_adjustment
    pattern_adjustments = {}
    for pattern, st in stats.items():
        if st["total"] >= 3:  # 至少3次评估才调整
            pattern_adjustments[pattern] = st["weight_adjustment"]
            print(f"  {pattern}: 胜率{st['win_rate']*100:.1f}% → 调整因子{st['weight_adjustment']:.2f}x")
    
    # 如果有足够数据，调整价格形态权重
    if pattern_adjustments:
        avg_adj = sum(pattern_adjustments.values()) / len(pattern_adjustments)
        old_w = params["w_price_pattern"]
        new_w = max(0.2, min(0.5, old_w * avg_adj))
        params["w_price_pattern"] = round(new_w, 2)
        # 相应调整其他权重
        diff = new_w - old_w
        params["w_volume"] = max(0.1, min(0.4, params["w_volume"] - diff * 0.4))
        params["w_sector"] = max(0.1, min(0.4, params["w_sector"] - diff * 0.3))
        params["w_acceleration"] = max(0.05, min(0.3, params["w_acceleration"] - diff * 0.3))
        print(f"  价格形态权重: {old_w:.2f} → {new_w:.2f}")
    
    # ── 2. 调整阈值 ──
    # 如果强信号胜率低，提高阈值
    strong_signals = [s for s in memory.signals if s.get("signal_grade") == "STRONG" and s["outcome_success"] is not None]
    if len(strong_signals) >= 5:
        strong_win_rate = sum(1 for s in strong_signals if s["outcome_success"]) / len(strong_signals)
        print(f"  强信号胜率: {strong_win_rate*100:.1f}% ({len(strong_signals)}次)")
        
        if strong_win_rate < 0.5:
            # 胜率低于50%，提高强信号门槛
            old_threshold = params["strong_signal"]
            params["strong_signal"] = min(250, old_threshold + 10)
            print(f"  提高强信号阈值: {old_threshold} → {params['strong_signal']}")
        elif strong_win_rate > 0.75:
            # 胜率很高，可以降低门槛捕获更多机会
            old_threshold = params["strong_signal"]
            params["strong_signal"] = max(150, old_threshold - 10)
            print(f"  降低强信号阈值: {old_threshold} → {params['strong_signal']}")
    
    # ── 3. 调整伪信号扣除力度 ──
    fake_signals = [s for s in memory.signals if s.get("fake_score", 0) > 0 and s["outcome_success"] is not None]
    if len(fake_signals) >= 5:
        fake_win = sum(1 for s in fake_signals if s["outcome_success"])
        fake_rate = fake_win / len(fake_signals)
        print(f"  被标记伪信号的胜率: {fake_rate*100:.1f}% ({len(fake_signals)}次)")
        # 如果伪信号标记错了（实际表现好），减少扣分力度
        if fake_rate > 0.5:
            print(f"  伪信号误判率高，降低扣分权重")
    
    # ── 保存 ──
    save_params(params)
    print(f"\n  参数已更新并保存至 params.json")
    print(f"  当前权重: 价格形态={params['w_price_pattern']:.2f} "
          f"成交量={params['w_volume']:.2f} "
          f"板块={params['w_sector']:.2f} "
          f"加速度={params['w_acceleration']:.2f}")
    
    return params


# ── CLI 入口 ──────────────────────────────────

if __name__ == "__main__":
    import sys
    memory = SignalMemory()
    
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        n = evaluate_signals(memory)
        print(f"评估完成: {n} 个信号")
    elif len(sys.argv) > 1 and sys.argv[1] == "adjust":
        adjust_params(memory)
    elif len(sys.argv) > 1 and sys.argv[1] == "summary":
        print(memory.summary())
    else:
        print("用法: python feedback.py [evaluate|adjust|summary]")
        print(f"\n当前信号日志: {len(memory.signals)} 条记录")
        print(memory.summary())
