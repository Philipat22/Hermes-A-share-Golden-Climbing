"""
Surge Engine Walk-Forward Backtester (cached data)
"""
import sys, os, time, json, pickle
from datetime import datetime, timedelta
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.tools.data_fetcher import get_16_sector_stocks, get_pro_api
from src.surge.engine import analyze_stock, load_params, _build_sector_cache, classify_signal
from src.ml.predictor import batch_predict_scores

FORWARD_DAYS = [5, 10, 20, 60]
BENCHMARK = "000300.SH"
TOP_N = 5
CACHE_PATH = "data/cache/backtest_prices.pkl"
MARKET_MA_WINDOW = 50


def _format_date(d: str) -> str:
    d = d.replace("-", "")
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _slice_df(df, end_date: str, window: int = 60):
    if df is None or df.empty or "date" not in df.columns:
        return None
    ed = _format_date(end_date)
    sliced = df[df["date"] <= ed].tail(window).copy()
    return sliced if len(sliced) >= 15 else None


def _forward_rets(prices: dict, code: str, eval_date: str, days: list[int]) -> dict:
    df = prices.get(code)
    if df is None or "date" not in df.columns:
        return {d: None for d in days}
    df = df.sort_values("date")
    ed = _format_date(eval_date)

    ref_row = df[df["date"] == ed]
    if ref_row.empty:
        before = df[df["date"] < ed]
        if before.empty:
            return {d: None for d in days}
        ref = before.iloc[-1]["close"]
    else:
        ref = ref_row.iloc[0]["close"]

    after = df[df["date"] > ed].sort_values("date")
    result = {}
    for nd in days:
        if len(after) >= nd:
            result[nd] = round((after.iloc[nd-1]["close"] - ref) / ref * 100, 2)
        elif len(after) > 0:
            result[nd] = round((after.iloc[-1]["close"] - ref) / ref * 100, 2)
        else:
            result[nd] = None
    return result


def run(start: str = "2025-01-01", end: str = "2026-03-31",
        preloaded: Optional[dict] = None) -> dict:
    t0 = time.time()

    # Stock pool
    sd = get_16_sector_stocks()
    pool = []
    for _, stocks in sd.items():
        pool.extend(stocks[:30])
    pool = sorted(set(pool))
    print(f"Pool: {len(pool)} stocks")

    # Eval dates
    pro = get_pro_api()
    cal = pro.trade_cal(start_date=start.replace("-",""), end_date=end.replace("-",""))
    td = cal[cal["is_open"] == 1]["cal_date"].astype(str).tolist()
    if not td:
        return {}
    # Every ~20 trading days
    evals = [td[i] for i in range(19, len(td), 20)]
    print(f"Eval dates: {len(evals)}")

    # Load cached prices
    if preloaded:
        prices = preloaded
    else:
        with open(CACHE_PATH, "rb") as f:
            prices = pickle.load(f)
        print(f"Loaded {len(prices)} stocks from cache")

    params = load_params()

    # ── ML scores: pre-compute once for all eval dates ──
    ml_scores_by_date: dict[str, dict[str, int]] = {}
    use_ml = True
    try:
        ml_scores_by_date = batch_predict_scores(prices, evals, pool)
        if ml_scores_by_date:
            print(f"ML scores: {len(ml_scores_by_date)} dates, "
                  f"~{sum(len(v) for v in ml_scores_by_date.values()) // max(1,len(ml_scores_by_date))} stocks/date")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ML scores unavailable: {e}")
        use_ml = False

    # Add benchmark to prices if missing (use index_daily for indices)
    if BENCHMARK not in prices:
        from src.tools.data_fetcher import get_index_data
        prices[BENCHMARK] = get_index_data(BENCHMARK, "2024-10-01", "2026-04-30")

    # ── Market regime filter ──
    bm_df = prices.get(BENCHMARK)
    def _market_ok(eval_date: str) -> bool:
        """Check if CSI300 > 50-day MA at eval_date"""
        if bm_df is None or "date" not in bm_df.columns:
            return True
        sliced = bm_df[bm_df["date"] <= _format_date(eval_date)]
        if len(sliced) < MARKET_MA_WINDOW:
            return True
        ma50 = sliced.tail(MARKET_MA_WINDOW)["close"].mean()
        cur = sliced.iloc[-1]["close"]
        return cur > ma50

    def _stock_trend_ok(df, min_len: int = 200) -> bool:
        """Stock must be in uptrend: close > MA50 > MA200."""
        if df is None or len(df) < min_len:
            return True
        closes = df["close"].values
        cur = closes[-1]
        ma50 = np.mean(closes[-50:])
        ma200 = np.mean(closes[-200:])
        return cur > ma50 > ma200

    # Walk forward
    periods = []
    skipped = 0
    for pi, ed in enumerate(evals):
        print(f"\n  [{pi+1}/{len(evals)}] {ed[:4]}-{ed[4:6]}-{ed[6:]}...", end=" ")
        sys.stdout.flush()
        pt0 = time.time()

        # Check market regime
        if not _market_ok(ed):
            print(f"SKIP (market below 50MA)")
            skipped += 1
            continue

        # Phase 1: single-threaded (CPU-bound, threading adds overhead)
        p1 = {}
        for code in pool:
            full = prices.get(code)
            if full is None:
                continue
            if not _stock_trend_ok(full):
                continue
            sdf = _slice_df(full, ed)
            if sdf is None:
                continue
            try:
                ml_scores = ml_scores_by_date.get(ed, {}) if use_ml else None
                sig = analyze_stock(code, sdf, {}, params, ml_scores=ml_scores)
                if classify_signal(sig, params) in ("STRONG", "WEAK"):
                    p1[code] = sig
            except Exception:
                pass

        # Phase 2: with pre-built sector_cache
        sec_cache = _build_sector_cache(p1) if p1 else {}
        p2 = {}
        for code in pool:
            full = prices.get(code)
            if full is None:
                continue
            if not _stock_trend_ok(full):
                continue
            sdf = _slice_df(full, ed)
            if sdf is None:
                continue
            try:
                sig = analyze_stock(code, sdf, p1, params, sec_cache, ml_scores)
                if classify_signal(sig, params) in ("STRONG", "WEAK"):
                    p2[code] = sig
            except Exception:
                pass

        # Rank
        ranked = sorted(p2.values(), key=lambda x: x.get("total_score", 0), reverse=True)
        top = ranked[:TOP_N]

        prd = {"date": ed, "n_signals_p1": len(p1), "n_signals_p2": len(p2), "picks": []}
        for pick in top:
            code = pick["ts_code"]
            rets = _forward_rets(prices, code, ed, FORWARD_DAYS)
            pick["forward_returns"] = rets
            prd["picks"].append(pick)
        prd["benchmark"] = _forward_rets(prices, BENCHMARK, ed, FORWARD_DAYS)

        # Log one-liner
        if top:
            scores = [p.get("total_score", 0) for p in top]
            pats = [p.get("pattern_type", "?") for p in top[:3]]
            print(f"{len(p2)} sigs, top={top[0]['ts_code']}({top[0]['total_score']},{top[0].get('pattern_type','?')}) "
                  f"{time.time()-pt0:.0f}s")
        else:
            print(f"{len(p2)} sigs, no picks {time.time()-pt0:.0f}s")

        periods.append(prd)

    return _aggregate(periods, time.time() - t0, len(pool), skipped)


def _aggregate(periods: list, tot_time: float, pool_size: int, skipped: int = 0) -> dict:
    report = {
        "config": {"pool_size": pool_size, "n_evals": len(periods), "skipped": skipped,
                    "forward_days": FORWARD_DAYS, "benchmark": "CSI300", "top_n": TOP_N},
        "runtime_s": round(tot_time, 1),
        "metrics": {},
        "periods": periods,
    }

    total_picks = sum(len(p.get("picks", [])) for p in periods)
    report["total_picks"] = total_picks
    report["picks_per_date"] = round(total_picks / max(1, len(periods)), 1)

    for nd in FORWARD_DAYS:
        pr = []
        for prd in periods:
            for p in prd.get("picks", []):
                r = p.get("forward_returns", {}).get(nd)
                if r is not None:
                    pr.append(r)

        br = [prd.get("benchmark", {}).get(nd) for prd in periods
              if prd.get("benchmark", {}).get(nd) is not None]

        m = {"n": len(pr)}
        if pr:
            a = np.array(pr)
            m.update(win_rate=round(float(np.mean(a > 0)), 4),
                     avg_return=round(float(np.mean(a)), 2),
                     median_return=round(float(np.median(a)), 2),
                     max_return=round(float(np.max(a)), 2),
                     min_return=round(float(np.min(a)), 2),
                     std_return=round(float(np.std(a)), 2))
        if br:
            m["benchmark_avg"] = round(float(np.mean(br)), 2)
            m["excess"] = round(m.get("avg_return", 0) - m["benchmark_avg"], 2)
        else:
            m["benchmark_avg"], m["excess"] = 0, 0
        report["metrics"][f"{nd}d"] = m

    return report


def pp(report: dict):
    print("\n" + "=" * 72)
    print("           SURGE ENGINE WALK-FORWARD BACKTEST")
    print("=" * 72)
    c = report.get("config", {})
    print(f"\n  Config:")
    print(f"    Pool:    {c.get('pool_size')} stocks, {c.get('n_evals')} eval dates")
    print(f"    Horizon: {', '.join(f'{d}d' for d in c.get('forward_days', FORWARD_DAYS))}")
    print(f"    Top N:   {c.get('top_n')}")
    print(f"    Runtime: {report.get('runtime_s')}s")
    print(f"    Total picks: {report.get('total_picks')} ({report.get('picks_per_date')}/date)")

    print(f"\n  {'─'*72}")
    h = f"  {'Horizon':>8} | {'WinRate':>8} | {'AvgRet':>8} | {'Median':>8} | {'Max':>8} | {'Min':>8} | {'Excess':>8} | {'N':>5}"
    print(h)
    print(f"  {'─'*72}")
    for nd in FORWARD_DAYS:
        m = report.get("metrics", {}).get(f"{nd}d", {})
        if m.get("n", 0) == 0:
            print(f"  {nd:>4}d   | {'N/A':>8}")
            continue
        wr = m.get("win_rate", 0)
        print(f"  {nd:>4}d   | {wr*100:>6.1f}%  | {m['avg_return']:>+6.2f}% | "
              f"{m['median_return']:>+6.2f}% | {m['max_return']:>+6.2f}% | "
              f"{m['min_return']:>+6.2f}% | {m['excess']:>+6.2f}% | {m['n']:>4}")
    print(f"  {'─'*72}")

    # Best/worst periods
    if report.get("periods"):
        p20 = []
        for p in report["periods"]:
            frs = [k.get("forward_returns", {}).get(20, 0) or 0 for k in p.get("picks", [])]
            p20.append((p["date"], np.mean(frs) if frs else 0))
        p20.sort(key=lambda x: x[1])
        print(f"\n  Best period:  {p20[-1][0][:4]}-{p20[-1][0][4:6]}-{p20[-1][0][6:]}  avg20d={p20[-1][1]:+.2f}%")
        print(f"  Worst period: {p20[0][0][:4]}-{p20[0][0][4:6]}-{p20[0][0][6:]}  avg20d={p20[0][1]:+.2f}%")


def save(report: dict):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "quant_archive",
        datetime.now().strftime("%Y-%m"),
        f"backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {path}")
    return path


if __name__ == "__main__":
    r = run()
    save(r)
    pp(r)
