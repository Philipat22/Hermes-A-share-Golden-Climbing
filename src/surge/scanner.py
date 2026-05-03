#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surge/scanner.py — 全市场形态扫描器

对指定股票池中的每只股票运行引擎分析，输出评分排序后的候选池
"""
from __future__ import annotations
import os, json, logging
from datetime import datetime
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# 存档路径
ARCHIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "quant_archive"
)

# 输出列名
SCORE_COLS = [
    "ts_code", "name", "industry", "close", "total_score",
    "final_score", "pattern_type", "pattern_score",
    "volume_score", "sector_score", "accel_score", "fake_score",
    "signal_grade", "detail",
]


def _get_stock_name(ts_code: str) -> str:
    try:
        from src.tools.a_stock_api import get_stock_name_for_ticker
        return get_stock_name_for_ticker(ts_code) or ts_code
    except Exception:
        return ts_code


def _get_industry(ts_code: str) -> str:
    try:
        from src.tools.a_stock_api import get_stock_info
        info = get_stock_info(ts_code)
        return info.industry if info else ""
    except Exception:
        return ""


def scan_market(
    stock_pool: Optional[list[str] | dict] = None,
    days: int = 120,
    min_price: float = 3.0,
    max_price: float = 200.0,
    save_report: bool = True,
    params: Optional[dict] = None,
    max_per_sector: int = 30,
    record_signals: bool = True,
    ml_scores: Optional[dict[str, int]] = None,
    prices_dict: Optional[dict[str, pd.DataFrame]] = None,
) -> list[dict]:
    """
    全市场形态扫描

    Args:
        stock_pool: 股票代码列表。None 时使用 16 板块池
        days: 获取多少天的数据（用于计算）
        min_price: 最低股价过滤
        max_price: 最高股价过滤
        save_report: 是否保存报告到 quant_archive
        params: 参数覆盖
        record_signals: 是否自动记录信号到 feedback SignalMemory

    Returns:
        sorted signals list (按 final_score 降序)
    """
    from src.surge.engine import analyze_stock, load_params, classify_signal
    from src.tools.a_stock_api import get_prices, get_16_sector_stocks, get_stock_info

    params = params or load_params()

    # ── 获取股票池（可能是 dict 或 list） ──
    if stock_pool is None:
        _sector_dict = get_16_sector_stocks()
        # 从每个板块取前 max_per_sector 只
        stock_pool = []
        for sector, stocks in _sector_dict.items():
            stock_pool.extend(stocks[:max_per_sector])
        print(f"股票池: {len(stock_pool)} 只 (取自 {len(_sector_dict)} 个板块，每板块最多{max_per_sector}只)")
    elif isinstance(stock_pool, dict):
        # 如果是 dict，flatten
        _flattened = []
        for _, stocks in stock_pool.items():
            _flattened.extend(stocks[:max_per_sector])
        stock_pool = _flattened
        print(f"股票池: {len(stock_pool)} 只 (dict flatten)")
    else:
        print(f"股票池: {len(stock_pool)} 只 (自定义)")

    # ── 拉取数据并分析 ──
    end_date = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - __import__("datetime").timedelta(days=days)).strftime("%Y-%m-%d")

    all_signals: dict[str, dict] = {}
    total = len(stock_pool)

    for i, code in enumerate(stock_pool):
        try:
            if prices_dict and code in prices_dict:
                df = prices_dict[code]
                # Ensure required columns
                for c in ['close','high','low','open','vol','date']:
                    if c not in df.columns and c == 'vol':
                        df['vol'] = df.get('volume', 0)
                    elif c not in df.columns:
                        df[c] = 0
            else:
                prices = get_prices(code, start, end_date)
                if not prices or len(prices) < 40:
                    continue
                df = pd.DataFrame([{
                    "close": p.close, "high": p.high, "low": p.low,
                    "open": p.open, "vol": p.volume,
                    "date": p.date,
                } for p in prices])

            # 股价过滤
            latest_close = float(df["close"].iloc[-1])
            if latest_close < min_price or latest_close > max_price:
                continue

            # 分析
            signal = analyze_stock(code, df, all_signals, params, ml_scores=ml_scores)
            if signal["detected"]:
                signal["name"] = _get_stock_name(code)
                signal["industry"] = _get_industry(code)
                signal["close"] = round(latest_close, 2)
                signal["signal_grade"] = classify_signal(signal, params)
                all_signals[code] = signal

        except Exception as e:
            logger.debug(f"[{i+1}/{total}] {code}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  扫描进度: {i+1}/{total} ({len(all_signals)} 个信号)")

    # ── 按最终评分排序 ──
    sorted_signals = sorted(all_signals.values(), key=lambda x: x.get("final_score", 0), reverse=True)

    # ── 打印摘要 ──
    print(f"\n扫描完成: {len(sorted_signals)} 个形态信号")
    grades = {}
    for s in sorted_signals:
        g = s.get("signal_grade", "NONE")
        grades[g] = grades.get(g, 0) + 1
    print(f"  等级分布: {grades}")

    strong = [s for s in sorted_signals if s["signal_grade"] == "STRONG"]
    weak = [s for s in sorted_signals if s["signal_grade"] == "WEAK"]

    if strong:
        print(f"\n* 强信号 ({len(strong)}):")
        for s in strong[:10]:
            print(f"  {s['ts_code']:12s} {s.get('name',''):8s} "
                  f"评分{s['final_score']:3d} {s.get('pattern_type',''):6s} "
                  f"{s.get('detail','')[:60]}")

    if weak:
        print(f"\n弱信号 ({len(weak)}):")
        for s in weak[:5]:
            print(f"  {s['ts_code']:12s} {s.get('name',''):8s} "
                  f"评分{s['final_score']:3d} {s.get('pattern_type',''):6s}")

    # ── 保存报告 ──
    if save_report:
        report_path = _save_report(sorted_signals, params)
    else:
        report_path = None

    # ── 自动记录到反馈系统（自我进化） ──
    if record_signals and sorted_signals:
        try:
            from src.surge.feedback import SignalMemory
            memory = SignalMemory()
            recorded = 0
            emotion_count = 0
            for s in sorted_signals:
                if s.get("signal_grade") in ("STRONG", "WEAK"):
                    # 对强信号+Top弱信号做情绪融合补全
                    if s.get("signal_grade") == "STRONG" or emotion_count < 15:
                        try:
                            from src.emotion import analyze_emotion
                            er = analyze_emotion(s["ts_code"])
                            s["emotion"] = er
                        except Exception:
                            s["emotion"] = {"fusion_score": s.get("final_score", 50),
                                             "confidence": 50, "label": "neutral",
                                             "components": {}}
                        emotion_count += 1
                    memory.record(s)
                    recorded += 1
            print(f"\n[feedback] recorded {recorded} signals to SignalMemory "
                  f"(emotion: {emotion_count})")
            print(f"[feedback] total in memory: {len(memory.signals)}")
        except Exception as ex:
            logger.debug(f"feedback record failed: {ex}")

    return sorted_signals


def _save_report(signals: list[dict], params: dict) -> str:
    """保存扫描报告到 quant_archive"""
    import json
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    now = datetime.now()
    month_dir = os.path.join(ARCHIVE_DIR, now.strftime("%Y-%m"))
    os.makedirs(month_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(month_dir, f"surge_scan_{now.strftime('%Y%m%d_%H%M')}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": now.isoformat(),
            "total_signals": len(signals),
            "params": {k: v for k, v in params.items() if not k.startswith("_")},
            "signals": signals,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[JSON] 报告已保存: {json_path}")

    # Markdown summary
    md_path = json_path.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Surge 形态扫描报告\n\n")
        f.write(f"生成时间: {now.strftime('%Y-%m-%d %H:%M')}\n\n")

        grades = {}
        for s in signals:
            g = s.get("signal_grade", "NONE")
            grades[g] = grades.get(g, 0) + 1
        f.write(f"**信号总数: {len(signals)}**\n")
        f.write(f"STRONG={grades.get('STRONG',0)} WEAK={grades.get('WEAK',0)} FAKE={grades.get('FAKE',0)}\n\n")

        if signals:
            f.write("| # | 代码 | 名称 | 行业 | 价格 | 评分 | 形态 | 形态分 | 量价分 | 板块分 | 加速分 | 伪信号 | 等级 |\n")
            f.write("|---|------|------|------|------|------|------|--------|--------|--------|--------|--------|------|\n")
            for i, s in enumerate(signals[:50], 1):
                f.write(
                    f"| {i} | {s.get('ts_code','')} "
                    f"| {s.get('name','')} "
                    f"| {s.get('industry','')} "
                    f"| {s.get('close',0)} "
                    f"| {s.get('final_score',0)} "
                    f"| {s.get('pattern_type','')} "
                    f"| {s.get('pattern_score',0)} "
                    f"| {s.get('volume_score',0)} "
                    f"| {s.get('sector_score',0)} "
                    f"| {s.get('accel_score',0)} "
                    f"| {s.get('fake_score',0)} "
                    f"| {s.get('signal_grade','')} |\n"
                )

    print(f"[MD] 报告已保存: {md_path}")
    return json_path


if __name__ == "__main__":
    print("=== Surge 全市场形态扫描 ===")
    signals = scan_market()
    print(f"\n共发现 {len(signals)} 个信号")
