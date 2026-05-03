#!/usr/bin/env python3
"""每日主升浪信号生成器。

全流水线：获取股票池 → 批量价格 → 因子计算 → ML预测 → 
        形态扫描 → 情绪融合 → 排名报告

执行: python src/signals/daily_generator.py
"""
import os, sys, json, pickle, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

# ── 项目根路径 ──
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

ARCHIVE = os.path.join(ROOT, "quant_archive")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
MODEL_PATH = os.path.join(ROOT, "data", "models", "surge_lgbm.pkl")
os.makedirs(ARCHIVE, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── 配置 ──
MAX_WORKERS = 8          # 并行拉取线程数
MIN_TRADING_DAYS = 60    # 最低交易天数
FETCH_DAYS = 180         # 拉取多少天历史数据


# ══════════════════════════════════════════════════════
#  1. 股票池
# ══════════════════════════════════════════════════════

def get_stock_pool(max_per_sector: int = 30) -> list[str]:
    """从16板块获取股票池"""
    from src.tools.a_stock_api import get_16_sector_stocks
    sd = get_16_sector_stocks()
    pool = []
    for sector, stocks in sd.items():
        pool.extend(stocks[:max_per_sector])
    # 去重
    seen = set()
    pool_dedup = [s for s in pool if not (s in seen or seen.add(s))]
    print(f"股票池: {len(pool_dedup)} 只 (取自 {len(sd)} 个板块)")
    return pool_dedup


# ══════════════════════════════════════════════════════
#  2. 并行拉取价格
# ══════════════════════════════════════════════════════

def _fetch_one(code: str, start_str: str, end_str: str) -> tuple[str, Optional[pd.DataFrame]]:
    """拉取单只股票价格"""
    try:
        from src.tools.data_fetcher import get_prices
        prices = get_prices(code, start_str, end_str)
        if not prices or len(prices) < MIN_TRADING_DAYS:
            return code, None
        rows = [{"date": p.date, "open": p.open, "high": p.high,
                 "low": p.low, "close": p.close, "vol": p.volume}
                for p in prices if hasattr(p, 'date') and p.date]
        df = pd.DataFrame(rows)
        if df.empty or len(df) < MIN_TRADING_DAYS:
            return code, None
        df = df.sort_values("date").reset_index(drop=True)
        return code, df
    except Exception as e:
        return code, None


def fetch_all_prices(
    stock_pool: list[str],
    days: int = FETCH_DAYS,
) -> dict[str, pd.DataFrame]:
    """并行拉取所有股票价格"""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    t0 = time.time()
    result: dict[str, pd.DataFrame] = {}
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_one, code, start, end): code for code in stock_pool}
        for fut in as_completed(futs):
            code, df = fut.result()
            if df is not None:
                result[code] = df
            done += 1
            if done % 100 == 0:
                print(f"  价格: {done}/{len(stock_pool)} ({len(result)} OK)")

    elapsed = time.time() - t0
    print(f"价格拉取: {len(result)}/{len(stock_pool)} 完成, 耗时 {elapsed:.0f}s")
    return result


# ══════════════════════════════════════════════════════
#  3. 批量 ML 评分
# ══════════════════════════════════════════════════════

def _compute_ml_scores_cached(
    prices_dict: dict[str, pd.DataFrame],
) -> dict[str, int]:
    """批量计算 ML 评分，带缓存

    缓存策略: 每天第一次计算需要 ~14min, 结果缓存到 data/cache/ 供当天复用
    """
    today = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"ml_scores_{today}.pkl"
    cache_path = os.path.join(CACHE_DIR, cache_key)

    # 尝试加载当日缓存
    if os.path.exists(cache_path):
        print(f"[ML] 加载缓存: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 实计算
    print("[ML] 无缓存，开始因子计算 (~14min) ...")
    t0 = time.time()

    from src.ml.pipeline import SurgeMLPipeline
    codes = list(prices_dict.keys())

    # 构建数据集 (163 因子计算)
    pln = SurgeMLPipeline()
    X, y, meta = pln.build_dataset(prices_dict, codes)

    if len(X) == 0:
        print("[ML] 因子计算失败，返回空")
        return {}

    factor_cols = getattr(pln, '_factor_names', None)
    if factor_cols is None:
        factor_cols = [c for c in meta.columns if c not in ('vt_symbol', 'datetime', 'label', 'forward_ret')]

    print(f"[ML] 因子完成: {X.shape[0]} 行, {len(factor_cols)} 因子, {time.time()-t0:.0f}s")

    # Load model & predict
    import lightgbm as lgb
    if not os.path.exists(MODEL_PATH):
        print(f"[ML] 模型文件不存在: {MODEL_PATH}")
        return {}

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Fill NaN and predict
    X_df = pd.DataFrame(X, columns=factor_cols).fillna(0).astype(np.float32)
    probs = model.predict(X_df.values)

    # Map predictions to stocks: take latest date's prediction per stock
    meta_df = pd.DataFrame(meta)
    meta_df["score"] = (probs * 100).astype(int)
    meta_df["score"] = meta_df["score"].clip(0, 100)

    # Keep last row per stock (most recent date)
    meta_df = meta_df.sort_values(["vt_symbol", "datetime"])
    latest = meta_df.groupby("vt_symbol").last().reset_index()
    result: dict[str, int] = dict(zip(latest["vt_symbol"], latest["score"]))

    # Cache
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[ML] 评分完成: {len(result)} 只, 缓存至 {cache_path}")

    return result


# ══════════════════════════════════════════════════════
#  4. 情绪融合
# ══════════════════════════════════════════════════════

def _fuse_emotion(signals: list[dict], top_n: int = 20) -> list[dict]:
    """对 Top 候选股做情绪融合"""
    from src.emotion.fusion import analyze_emotion

    for s in signals:
        try:
            code = s.get("ts_code", "")
            fusion = analyze_emotion(code)
            s["emotion_score"] = fusion.get("fusion_score", 50)
            s["emotion_confidence"] = fusion.get("confidence", 0)
            s["emotion_label"] = fusion.get("label", "neutral")
        except Exception:
            s["emotion_score"] = 50
            s["emotion_confidence"] = 0
            s["emotion_label"] = "neutral"

    return signals[:top_n]


# ══════════════════════════════════════════════════════
#  5. 生成报告
# ══════════════════════════════════════════════════════

def _write_picks_report(
    signals: list[dict],
    ml_info: dict,
) -> str:
    """生成选股报告 Markdown"""
    now = datetime.now()
    month_dir = os.path.join(ARCHIVE, now.strftime("%Y-%m"))
    os.makedirs(month_dir, exist_ok=True)

    path = os.path.join(month_dir, f"daily_picks_{now.strftime('%Y%m%d_%H%M')}.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 每日主升浪选股报告\n\n")
        f.write(f"生成时间: {now.strftime('%Y-%m-%d %H:%M')}\n\n")

        # ── Top Picks 推荐 ──
        f.write("## Top Picks\n\n")
        f.write("| # | 代码 | 名称 | 行业 | 价格 | 综合分 | 形态 | 形态分 | ML分 | 情绪 | 板块分 | 理由 |\n")
        f.write("|---|------|------|------|------|--------|------|--------|------|------|--------|------|\n")

        for i, s in enumerate(signals, 1):
            # 简述理由
            reason_parts = []
            pattern = s.get("pattern_type", "")
            if pattern and pattern != "none":
                reason_parts.append(f"{pattern}")
            ml = s.get("ml_score", 0)
            if ml >= 70:
                reason_parts.append("ML高置信")
            emo = s.get("emotion_label", "")
            if emo in ("bullish", "积极"):
                reason_parts.append("情绪偏多")
            detail = s.get("detail", "")[:50]
            if detail:
                reason_parts.append(detail)
            reason = " / ".join(reason_parts[:3]) or "—"

            # 融合评分 = surge_final (已含ML) + emotion_bonus
            final = s.get("final_score", 0)
            emo_score = s.get("emotion_score", 0)
            if emo_score >= 60:
                final = min(100, final + 5)
            elif emo_score >= 50:
                final = min(100, final + 2)

            f.write(
                f"| {i} | {s.get('ts_code','')} "
                f"| {s.get('name','')} "
                f"| {s.get('industry','')} "
                f"| {s.get('close',0)} "
                f"| {final} "
                f"| {pattern or '-'} "
                f"| {s.get('pattern_score',0)} "
                f"| {ml} "
                f"| {emo_score} "
                f"| {s.get('sector_score',0)} "
                f"| {reason} |\n"
            )

        # ── 详细信号列表 ──
        f.write("\n## 全部信号\n\n")
        f.write("| # | 代码 | 评分 | 形态 | 等级 | 详情 |\n")
        f.write("|---|------|------|------|------|------|\n")
        for i, s in enumerate(signals, 1):
            f.write(
                f"| {i} | {s.get('ts_code','')} "
                f"| {s.get('final_score',0)} "
                f"| {s.get('pattern_type','')} "
                f"| {s.get('signal_grade','')} "
                f"| {str(s.get('detail',''))[:80]} |\n"
            )

        # ── 统计 ──
        grades = {}
        for s in signals:
            g = s.get("signal_grade", "NONE")
            grades[g] = grades.get(g, 0) + 1
        f.write(f"\n## 统计\n\n")
        f.write(f"- 股票池: {ml_info.get('pool_size', 0)} 只\n")
        f.write(f"- 信号总数: {len(signals)}\n")
        f.write(f"- 等级分布: STRONG={grades.get('STRONG',0)} WEAK={grades.get('WEAK',0)} FAKE={grades.get('FAKE',0)} NONE={grades.get('NONE',0)}\n")
        f.write(f"- ML评分可用: {ml_info.get('ml_available', 0)} 只\n")
        f.write(f"- 生成耗时: {ml_info.get('elapsed', 0):.0f} 秒\n")

    print(f"\n[REPORT] 选股报告: {path}")
    return path


# ══════════════════════════════════════════════════════
#  6. 主入口
# ══════════════════════════════════════════════════════

def daily_generate(
    max_stocks: int = 300,
    min_price: float = 3.0,
    max_price: float = 200.0,
    fetch_prices: bool = True,
    save_report: bool = True,
    force_ml: bool = False,
) -> list[dict]:
    """
    每日主升浪信号生成（全自动流水线）

    Args:
        max_stocks: 扫描股票数上限 (0=全部)
        min_price:  最低股价
        max_price:  最高股价
        fetch_prices: 是否重新拉取价格（否则使用缓存）
        save_report:  是否保存报告
        force_ml:     是否强制重算 ML 评分

    Returns:
        sorted picks list
    """
    t_start = time.time()

    # ── 1. 股票池 ──
    pool = get_stock_pool(max_per_sector=9999)
    if max_stocks and len(pool) > max_stocks:
        pool = pool[:max_stocks]
    print(f"\n股票池: {len(pool)} 只")

    # ── 2. 价格 ──
    print("\n=== 拉取价格 ===")
    prices_dict = fetch_all_prices(pool)
    if len(prices_dict) < 10:
        print("价格拉取失败，退出")
        return []

    # ── 3. ML 评分 ──
    print("\n=== ML 评分 ===")
    ml_scores = _compute_ml_scores_cached(prices_dict)

    # ── 4. 形态扫描 ──
    print("\n=== 形态扫描 ===")
    from src.surge.scanner import scan_market
    signals = scan_market(
        stock_pool=pool,
        min_price=min_price,
        max_price=max_price,
        save_report=False,
        ml_scores=ml_scores,
        prices_dict=prices_dict,
        record_signals=True,
    )
    print(f"形态扫描完成: {len(signals)} 个信号")

    # ── 5. 情绪融合 (Top 30) ──
    print("\n=== 情绪融合 ===")
    top = _fuse_emotion(signals[:30], top_n=20)
    print(f"Top {len(top)} 候选情绪融合完成")

    # ── 6. 报告 ──
    if save_report:
        ml_info = {
            "pool_size": len(pool),
            "ml_available": len(ml_scores),
            "elapsed": time.time() - t_start,
        }
        _write_picks_report(top, ml_info)

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"每日信号生成完成! 耗时 {elapsed:.0f}s")
    print(f"Top-3: {[s.get('ts_code','') for s in top[:3]]}")

    return top


# ══════════════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="每日主升浪信号生成")
    parser.add_argument("--max-stocks", type=int, default=300, help="扫描股票数上限")
    parser.add_argument("--min-price", type=float, default=3.0, help="最低股价")
    parser.add_argument("--max-price", type=float, default=200.0, help="最高股价")
    parser.add_argument("--force-ml", action="store_true", help="强制重算 ML 评分")
    parser.add_argument("--no-fetch", action="store_true", help="不重新拉取价格（使用缓存）")
    parser.add_argument("--no-report", action="store_true", help="不保存报告")
    args = parser.parse_args()

    picks = daily_generate(
        max_stocks=args.max_stocks,
        min_price=args.min_price,
        max_price=args.max_price,
        fetch_prices=not args.no_fetch,
        save_report=not args.no_report,
        force_ml=args.force_ml,
    )
    print(f"\n共 {len(picks)} 个候选")
