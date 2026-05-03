#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Surge 选股引擎 — A股突破形态识别与多因子评分

核心功能:
- 平台突破识别 (platform breakout)
- N型反转识别 (N-pattern reversal)
- VCP 波动收缩形态 (Volatility Contraction Pattern)
- 多因子加权评分 (价格形态/量能/板块共振/加速度)
- LightGBM 辅助信号

配置: params.json (权重、阈值、LGBM参数)
"""
from __future__ import annotations

import math, json, os, logging

from datetime import datetime, timedelta

from typing import Any, Optional

import pandas as pd

import numpy as np



logger = logging.getLogger(__name__)

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "params.json")

DEFAULT_PARAMS = {
    # Platform breakout
    "platform_min_days": 15,
    "platform_max_amplitude": 0.15,
    "breakout_volume_ratio": 1.8,
    # N-shape
    "n_first_leg_min_return": 0.10,
    "n_pullback_max": 0.05,
    # Volume
    "volume_ma_window": 20,
    "high_vol_ratio": 1.8,
    "low_vol_ratio": 0.6,
    # VCP
    "vcp_long_window": 60,
    "vcp_short_window": 20,
    "vcp_volatility_ratio": 0.6,
    # Acceleration
    "accel_window": 10,
    "accel_threshold": 0.05,
    # Weights
    "w_price_pattern": 0.35,
    "w_volume": 0.20,
    "w_sector": 0.20,
    "w_acceleration": 0.25,
    # Classification
    "weak_signal": 55,
    "strong_signal": 80,
    "sector_min_peers": 3,
}


def load_params() -> dict:

    """Load parameters from file, fall back to defaults."""
    try:
        with open(PARAMS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict) and len(loaded) > 5:
            return loaded
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return dict(DEFAULT_PARAMS)





#  ?



def detect_platform_breakout(

    df: pd.DataFrame,

    params: Optional[dict] = None,

) -> dict[str, Any]:

    """Detect platform/consolidation breakout pattern."""
    if params is None:

        params = load_params()

    min_days = params["platform_min_days"]

    max_amp = params["platform_max_amplitude"]

    vol_ratio = params["breakout_volume_ratio"]



    result = {"detected": False, "score": 0, "detail": "", 

              "consol_low": None, "consol_high": None,

              "breakout_price": None, "days": 0, "amplitude": 0}



    if df is None or len(df) < max(min_days + 10, 60):

        return result



    closes = df["close"].values

    highs = df["high"].values

    lows = df["low"].values

    volumes = df["vol"].values if "vol" in df.columns else df.get("volume", df.get("amount", pd.Series([0]*len(df)))).values



    # ?min_days ?

    recent = df.tail(min_days)

    consol_high = float(recent["high"].max())

    consol_low = float(recent["low"].min())

    amplitude = (consol_high - consol_low) / consol_low if consol_low > 0 else 1



    result["consol_low"] = consol_low

    result["consol_high"] = consol_high

    result["amplitude"] = round(amplitude, 4)

    result["days"] = min_days



    # 1?

    if amplitude > max_amp:

        result["detail"] = f"{amplitude*100:.1f}% > {max_amp*100:.0f}%"

        return result



    # 2?

    ma20 = df["close"].rolling(20).mean()

    latest_close = float(closes[-1])

    ma20_val = float(ma20.iloc[-1])

    if ma20_val > 0 and abs(latest_close - ma20_val) / ma20_val > 0.08:

        result["detail"] += f" MA20({abs(latest_close-ma20_val)/ma20_val*100:.1f}%)"

        return result



    # 3?

    if latest_close <= consol_high:

        result["detail"] += ""

        return result



    result["breakout_price"] = latest_close



    # 4

    ma_vol = float(np.mean(volumes[-params["volume_ma_window"]:])) if len(volumes) >= params["volume_ma_window"] else float(np.mean(volumes))

    latest_vol = float(volumes[-1])

    vol_confirmed = latest_vol > ma_vol * vol_ratio



    # 

    score = 0

    # ?

    amp_score = max(0, min(100, (1 - amplitude / max_amp) * 100))

    # MA20

    ma_score = max(0, min(100, 100 - abs(latest_close - ma20_val) / ma20_val * 500))

    # ?

    vol_score = 80 if vol_confirmed else 30

    # 

    days_score = min(100, min_days * 4)



    score = int(amp_score * 0.3 + ma_score * 0.25 + vol_score * 0.25 + days_score * 0.2)



    result["detected"] = True

    result["score"] = score

    result["detail"] = (

        f" {amplitude*100:.1f}% {min_days}d "

        f"MA20{abs(latest_close-ma20_val)/ma20_val*100:.1f}% "

        f"{'' if vol_confirmed else ''}"

    )

    return result





def detect_n_shape(

    df: pd.DataFrame,

    params: Optional[dict] = None,

) -> dict[str, Any]:

    """Detect N-shape (cup and handle) pattern."""
    if params is None:

        params = load_params()

    min_return = params["n_first_leg_min_return"]

    max_pullback = params["n_pullback_max"]



    result = {"detected": False, "score": 0, "detail": "",

              "first_leg_return": 0, "pullback_depth": 0,

              "second_leg_volume_ratio": 0}



    if df is None or len(df) < 60:

        return result



    closes = df["close"].values

    volumes = df["vol"].values if "vol" in df.columns else df.get("volume", df.get("amount", pd.Series([0]*len(df)))).values

    highs = df["high"].values

    lows = df["low"].values



    n = len(closes)

    # ?0?020-40

    recent20 = df.tail(20)

    prev20 = df.iloc[-40:-20] if len(df) > 40 else df.head(20)



    # 40-20

    first_low = float(prev20["low"].min())

    first_high = float(prev20["high"].max())

    first_return = (first_high - first_low) / first_low if first_low > 0 else 0



    if first_return < min_return:

        result["detail"] = f"{first_return*100:.1f}% < {min_return*100:.0f}%"

        return result



    # ?0

    second_low = float(recent20["low"].min())

    pullback_depth = (second_low - first_high) / first_high if first_high > 0 else 0

    # pullback_depth ?

    pullback_pct = abs(pullback_depth)



    if pullback_pct > max_pullback:

        result["detail"] = f"{pullback_pct*100:.1f}% > {max_pullback*100:.1f}%"

        return result



    # ?

    latest_close = float(closes[-1])

    second_return = (latest_close - second_low) / second_low if second_low > 0 else 0



    if second_return < 0.05:  # 3%

        result["detail"] = ""

        return result



    # 

    vol_20d_avg = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))

    pullback_vol_avg = float(np.mean(volumes[-10:]))

    vol_shrink = pullback_vol_avg < vol_20d_avg * params["low_vol_ratio"]



    # 

    first_leg_score = min(100, first_return / min_return * 50)

    # 

    if pullback_pct <= 0.382:

        pullback_score = 100

    elif pullback_pct <= 0.5:

        pullback_score = 70

    else:

        pullback_score = 40

    second_leg_vol_score = 80 if vol_shrink else 30



    score = int(first_leg_score * 0.3 + pullback_score * 0.4 + second_leg_vol_score * 0.3)



    result["detected"] = True

    result["score"] = score

    result["first_leg_return"] = round(first_return, 4)

    result["pullback_depth"] = round(pullback_depth, 4)

    result["second_leg_volume_ratio"] = round(pullback_vol_avg / vol_20d_avg, 4) if vol_20d_avg > 0 else 0

    result["detail"] = (

        f"N {first_return*100:.1f}% "

        f"{pullback_pct*100:.1f}% "

        f"{'' if vol_shrink else ''} "

        f"{second_return*100:.1f}%"

    )

    return result





def detect_vcp(

    df: pd.DataFrame,

    params: Optional[dict] = None,

) -> dict[str, Any]:

    """Detect Volatility Contraction Pattern (VCP)."""
    if params is None:

        params = load_params()



    result = {"detected": False, "score": 0, "detail": "",

              "short_volatility": 0, "long_volatility": 0, "vol_ratio": 0,

              "price_drop_max": 0}



    if df is None or len(df) < params["vcp_long_window"] + 10:

        return result



    # 

    daily_range = (df["high"] - df["low"]) / df["close"].shift(1)

    daily_range = daily_range.dropna()



    if len(daily_range) < params["vcp_long_window"]:

        return result



    short_window = params["vcp_short_window"]

    long_window = params["vcp_long_window"]

    vol_ratio_threshold = params["vcp_volatility_ratio"]



    short_vol = float(daily_range.tail(short_window).mean())

    long_vol = float(daily_range.tail(long_window).mean())



    vol_ratio = short_vol / long_vol if long_vol > 0 else 99

    result["short_volatility"] = round(short_vol, 4)

    result["long_volatility"] = round(long_vol, 4)

    result["vol_ratio"] = round(vol_ratio, 4)



    if vol_ratio > vol_ratio_threshold:

        result["detail"] = f"vol ratio: {short_vol*100:.1f}% / {long_vol*100:.1f}% = {vol_ratio:.2f} > {vol_ratio_threshold:.2f}"

        return result



    # VCP

    recent = df.tail(short_window)

    max_close = float(recent["close"].max())

    min_close = float(recent["close"].min())

    price_drop = (min_close - max_close) / max_close if max_close > 0 else 0

    result["price_drop_max"] = round(price_drop, 4)



    if price_drop < -0.05:  # ?%

        result["detail"] += f" {price_drop*100:.1f}%"

        return result



    # 

    # 

    vol_score = max(0, min(100, (1 - vol_ratio) * 200))

    # 

    price_score = 100 if price_drop >= -0.02 else (80 if price_drop >= -0.035 else 50)

    # ?

    stability_score = max(0, min(100, 100 - short_vol * 5000))



    score = int(vol_score * 0.4 + price_score * 0.3 + stability_score * 0.3)



    result["detected"] = True

    result["score"] = score

    result["detail"] = (

        f"VCP {short_vol*100:.2f}%/{long_vol*100:.2f}% "

        f"= {vol_ratio:.2f} {-price_drop*100:.1f}%"

    )

    return result





def detect_v_reversal(

    df: pd.DataFrame,

    params: Optional[dict] = None,

) -> dict[str, Any]:

    """Detect V-shaped reversal pattern."""
    if params is None:

        params = load_params()



    result = {"detected": False, "score": 0, "detail": "",

              "drop_depth": 0, "recovery_strength": 0, "reverse_days": 0,

              "drop_volume_surge": False, "recovery_volume_surge": False}



    if df is None or len(df) < 60:

        return result



    closes = df["close"].values

    volumes = df["vol"].values if "vol" in df.columns else df.get("volume", df.get("amount", pd.Series([0]*len(df)))).values



    n = len(closes)



    # ?0

    lookback = min(30, n - 1)

    recent = df.tail(lookback)

    min_idx_rel = int(recent["close"].values.argmin())

    min_idx = n - lookback + min_idx_rel



    if min_idx < 5 or min_idx >= n - 3:

        result["detail"] = ""

        return result



    # ?

    left_high = float(max(closes[max(0, min_idx - 20):min_idx + 1]))

    min_price = float(closes[min_idx])

    drop_depth = (min_price - left_high) / left_high if left_high > 0 else 0



    if drop_depth > -0.08:  # 8%

        result["detail"] = f" {drop_depth*100:.1f}%"

        return result



    # ?

    max_after = float(max(closes[min_idx:]))

    recovery = (max_after - min_price) / min_price if min_price > 0 else 0



    if recovery < 0.05:  # 5%

        result["detail"] = f" {recovery*100:.1f}%"

        return result



    # 

    reverse_days = n - min_idx

    result["reverse_days"] = reverse_days

    result["drop_depth"] = round(drop_depth, 4)

    result["recovery_strength"] = round(recovery, 4)



    # ?+ 

    vol_20d_avg = float(np.mean(volumes[-min(20, n):]))

    

    # 5

    drop_vol_avg = float(np.mean(volumes[max(0, min_idx - 5):min_idx + 1])) if min_idx >= 5 else 0

    drop_vol_surge = drop_vol_avg > vol_20d_avg * 1.3 if vol_20d_avg > 0 else False

    result["drop_volume_surge"] = drop_vol_surge



    # 

    recovery_vol_avg = float(np.mean(volumes[min_idx:])) if n > min_idx else 0

    recovery_vol_surge = recovery_vol_avg > vol_20d_avg * 1.2 if vol_20d_avg > 0 else False

    result["recovery_volume_surge"] = recovery_vol_surge



    # 

    drop_score = min(100, abs(drop_depth) / 0.15 * 100)  # ?5%?00?

    recovery_score = min(100, recovery / 0.1 * 100)  # 10%?00?

    vol_score = 80 if recovery_vol_surge else (50 if drop_vol_surge else 30)

    speed_score = max(30, 100 - reverse_days * 3)  # ?



    score = int(drop_score * 0.3 + recovery_score * 0.3 + vol_score * 0.25 + speed_score * 0.15)



    result["detected"] = True

    result["score"] = score

    result["detail"] = (

        f"V?{abs(drop_depth)*100:.1f}% "

        f"{recovery*100:.1f}% "

        f"{reverse_days}d "

        f"{'' if recovery_vol_surge else ''}"

    )

    return result





def detect_w_bottom(

    df: pd.DataFrame,

    params: Optional[dict] = None,

) -> dict[str, Any]:

    """Detect W-bottom (double bottom) pattern."""
    if params is None:

        params = load_params()



    result = {"detected": False, "score": 0, "detail": "",

              "left_bottom": 0, "right_bottom": 0, "neckline": 0,

              "break_confirmed": False, "bottom_spacing": 0}



    if df is None or len(df) < 60:

        return result



    closes = df["close"].values

    highs = df["high"].values

    lows = df["low"].values

    volumes = df["vol"].values if "vol" in df.columns else df.get("volume", df.get("amount", pd.Series([0]*len(df)))).values



    n = len(closes)

    if n < 45:

        return result



    # ?5?

    # ??5-252?5-5

    seg1 = df.iloc[-45:-20] if n >= 45 else df.iloc[:max(1, n-20)]

    seg2 = df.iloc[-20:-3] if n >= 20 else df.iloc[:max(1, n-3)]



    if len(seg1) < 5 or len(seg2) < 5:

        return result



    left_low = float(seg1["low"].min())

    left_low_idx = int(seg1["low"].idxmin()) if hasattr(seg1["low"], "idxmin") else int(seg1["low"].values.argmin())

    right_low = float(seg2["low"].min())



    # 

    between_segs = df.iloc[-40:-5] if n >= 40 else df

    neckline = float(between_segs["high"].max())



    result["left_bottom"] = round(left_low, 2)

    result["right_bottom"] = round(right_low, 2)

    result["neckline"] = round(neckline, 2)



    # 1?%?

    bottom_diff = (right_low - left_low) / left_low if left_low > 0 else 0

    if bottom_diff < -0.01:  # 1%

        result["detail"] = f"({right_low:.2f}) < ({left_low:.2f})"

        return result



    # 2?%?

    left_peak = float(seg1["high"].max())

    rebound = (left_peak - left_low) / left_low if left_low > 0 else 0

    if rebound < 0.03:

        result["detail"] = f"{rebound*100:.1f}%"

        return result



    result["bottom_spacing"] = round(rebound, 4)



    # 3?

    latest_close = float(closes[-1])

    break_confirmed = latest_close > neckline * 1.01  # 1%

    result["break_confirmed"] = break_confirmed



    # 4?

    ma_vol = float(np.mean(volumes[-20:])) if n >= 20 else float(np.mean(volumes))

    latest_vol = float(volumes[-1])

    vol_break = latest_vol > ma_vol * 1.3 if ma_vol > 0 else False



    # 5?

    right_vol_avg = float(np.mean([volumes[i] for i in range(-20, -3)])) if n >= 20 else 0

    left_vol_avg = float(np.mean([volumes[i] for i in range(-45, -20)])) if n >= 45 else 0

    vol_shrink_right = right_vol_avg < left_vol_avg * 0.8 if left_vol_avg > 0 else False



    # 

    # 

    symmetry = max(0, 100 - abs(bottom_diff) * 500)

    # 

    break_score = 100 if break_confirmed else (40 if latest_close > neckline else 0)

    # ?

    vol_score = 100 if vol_break and vol_shrink_right else (60 if vol_break else (40 if vol_shrink_right else 20))

    # 

    neck_score = min(100, neckline / left_low * 50) if left_low > 0 else 50



    score = int(symmetry * 0.25 + break_score * 0.35 + vol_score * 0.25 + neck_score * 0.15)



    result["detected"] = True

    result["score"] = score

    result["detail"] = (

        f"W?{left_low:.2f}{right_low:.2f} "

        f"{neckline:.2f} "

        f"{'' if vol_break else ''}"

    )

    return result





#  ?



def measure_acceleration(

    df: pd.DataFrame,

    params: Optional[dict] = None,

) -> dict[str, Any]:

    """Measure price acceleration."""
    if params is None:

        params = load_params()

    window = params["accel_window"]

    threshold = params["accel_threshold"]



    result = {"acceleration": 0, "trend_status": "unknown",

              "recent_return": 0, "prior_return": 0,

              "is_accelerating": False, "score": 0, "detail": ""}



    if df is None or len(df) < window * 3:

        return result



    closes = df["close"].values

    n = len(closes)



    # 

    recent_return = (closes[-1] - closes[-1 - window]) / closes[-1 - window] if closes[-1 - window] > 0 else 0

    # 

    prior_return = (closes[-1 - window] - closes[-1 - window * 2]) / closes[-1 - window * 2] if closes[-1 - window * 2] > 0 else 0



    acceleration = recent_return - prior_return



    result["recent_return"] = round(recent_return, 4)

    result["prior_return"] = round(prior_return, 4)

    result["acceleration"] = round(acceleration, 4)



    # 

    # 405%1585

    accel_mag = abs(acceleration)

    prop = min(85, max(10, int(40 + (accel_mag / 0.05) * 15)))



    # ?

    if recent_return > 0 and acceleration > threshold:

        result["trend_status"] = ""

        result["is_accelerating"] = True

        result["score"] = prop

        result["detail"] = f"{prior_return*100:.1f}%->{recent_return*100:.1f}% accel {acceleration*100:.1f}%"

    elif recent_return > 0 and acceleration >= -threshold:

        result["trend_status"] = ""

        result["score"] = max(40, prop - 10)

        result["detail"] = f" {prior_return*100:.1f}%->{recent_return*100:.1f}%"

    elif recent_return > 0 and acceleration < -threshold:

        result["trend_status"] = ""

        result["score"] = max(15, 40 - int(accel_mag / 0.05) * 10)

        result["detail"] = f" {prior_return*100:.1f}%->{recent_return*100:.1f}% {acceleration*100:.1f}%"

    elif recent_return < 0 and acceleration > threshold:

        result["trend_status"] = ""

        result["score"] = min(75, prop - 5)

        result["is_accelerating"] = True

        result["detail"] = f" {prior_return*100:.1f}%->{recent_return*100:.1f}% accel {acceleration*100:.1f}%"

    elif recent_return < -0.05:

        result["trend_status"] = ""

        result["score"] = 10

        result["detail"] = f" {recent_return*100:.1f}%"

    else:

        result["trend_status"] = ""

        result["score"] = 40

        result["detail"] = f" {recent_return*100:.1f}%"



    return result





#  ?



def score_volume_structure(

    df: pd.DataFrame,

    params: Optional[dict] = None,

) -> dict[str, Any]:

    """Score volume structure for signal quality."""
    if params is None:

        params = load_params()

    ma_window = params["volume_ma_window"]

    high_ratio = params["high_vol_ratio"]



    result = {"score": 0, "breakout_volume_confirmed": False,

              "pullback_volume_shrinking": False,

              "volume_divergence": False, "consecutive_volume": False,

              "detail": "", "recent_vol_ratio": 0}



    if df is None or len(df) < ma_window + 10:

        return result



    volumes = df["vol"].values if "vol" in df.columns else df.get("volume", df.get("amount", pd.Series([0]*len(df)))).values

    closes = df["close"].values



    ma_vol = float(np.mean(volumes[-ma_window:])) if len(volumes) >= ma_window else float(np.mean(volumes))

    latest_vol = float(volumes[-1])

    vol_ratio_ = latest_vol / ma_vol if ma_vol > 0 else 1



    result["recent_vol_ratio"] = round(vol_ratio_, 2)



    score = 50  # ?

    details = []



    # 1. ?

    if vol_ratio_ > high_ratio:

        result["breakout_volume_confirmed"] = True

        score += 15

        details.append("")



    # 2. ?0??

    if len(volumes) >= 20:

        vol_10d = volumes[-10:]

        lowest_10d = min(vol_10d)

        if lowest_10d < ma_vol * params["low_vol_ratio"]:

            result["pullback_volume_shrinking"] = True

            score += 10

            details.append("")



    # 3. ?

    if len(volumes) >= 5:

        last_3 = volumes[-3:]

        consecutive = all(last_3[i] >= volumes[-6 + i] * 1.1 for i in range(3))

        if consecutive:

            result["consecutive_volume"] = True

            score += 15

            details.append("")



    # 4. ?0>2x

    if len(volumes) >= 20:

        vol_20d = volumes[-20:-5]  # ??

        for i, v in enumerate(vol_20d):

            if v > ma_vol * 2.0:

                # ?-10?

                idx = len(volumes) - 20 + i

                if idx + 5 < len(closes) and closes[idx + 5] >= closes[idx] * 0.95:

                    score += 8

                    details.append("")

                    break



    # 5. ?

    if len(closes) >= 10 and len(volumes) >= 10:

        if closes[-1] == max(closes[-10:]):

            if vol_ratio_ < 0.8:

                result["volume_divergence"] = True

                score -= 40

                details.append("**")



    # 6. 

    if len(closes) >= 5 and len(volumes) >= 5:

        if vol_ratio_ > 1.3:

            ret_5d = (closes[-1] - closes[-5]) / closes[-5]

            if ret_5d < 0.01:

                score -= 35

                details.append("**")



    result["score"] = max(0, min(100, score))

    result["detail"] = " ".join(details) if details else ""

    return result





#  ?



_SECTOR_CACHE_LAST: tuple = (None, None)


def _build_sector_cache(all_signals: dict) -> dict:

    """Build sector signal density cache."""
    global _SECTOR_CACHE_LAST
    sig_id = id(all_signals)

    if _SECTOR_CACHE_LAST[0] is sig_id:
        return _SECTOR_CACHE_LAST[1]
    from src.tools.a_stock_api import get_stock_info

    cache: dict = {}

    for code, sig in all_signals.items():

        try:

            info = get_stock_info(code)

            if info and info.industry:

                sec = info.industry

            else:

                continue

        except Exception:

            continue

        if sec not in cache:

            cache[sec] = {"strong": 0, "weak": 0, "total": 0}

        cache[sec]["total"] += 1

        ts = sig.get("total_score", 0) if isinstance(sig, dict) else 0

        ws = load_params().get("weak_signal", 60)

        if ts >= ws:

            cache[sec]["strong"] += 1

        elif ts >= 35:

            cache[sec]["weak"] += 1

    _SECTOR_CACHE_LAST = (sig_id, cache)
    return cache





def score_sector_context(

    ts_code: str,

    all_signals: dict[str, Any],

    params: Optional[dict] = None,

    sector_cache: Optional[dict] = None,

) -> dict[str, Any]:

    """Score sector/market context."""
    if params is None:

        params = load_params()



    result = {"score": 30, "peer_signals": 0, "peers_in_sector": 0,

              "detail": "", "sector_name": "",

              "sector_index_perf": "N/A", "sector_index_code": ""}



    try:

        from src.tools.a_stock_api import get_stock_info, get_sector_index_perf

    except ImportError:

        return result



    info = get_stock_info(ts_code)

    if not info or not info.industry:

        result["detail"] = ""

        return result



    sector = info.industry

    result["sector_name"] = sector



    #  Part 1:  (0-50? 

    try:

        perf_data = get_sector_index_perf(ts_code, cache_minutes=60)

        ps = perf_data.get("perf_score", 50)

        index_score = min(50, max(0, int(ps * 50 / 85)))

        result["sector_index_perf"] = f"{perf_data.get('perf_20d', 0):.1f}%"

        result["sector_index_code"] = perf_data.get("index_code", "")

    except Exception:

        index_score = 25



    #  Part 2:  (0-50?  O(1)  

    if sector_cache is None and all_signals:

        sector_cache = _build_sector_cache(all_signals)

    sec_data = (sector_cache or {}).get(sector, {"strong": 0, "weak": 0, "total": 0})



    result["peers_in_sector"] = sec_data["total"]

    result["peer_signals"] = sec_data["strong"]



    strong_cnt = sec_data["strong"]

    weak_cnt = sec_data["weak"]



    if strong_cnt >= params.get("sector_min_peers", 3):

        peer_score = 50

        peer_detail = f"{strong_cnt} strong"

    elif strong_cnt >= 1:

        peer_score = 35

        peer_detail = f"{strong_cnt} strong"

    elif weak_cnt >= 5:

        peer_score = 25

        peer_detail = f"{weak_cnt} weak"

    else:

        peer_score = 10

        peer_detail = ""



    result["score"] = index_score + peer_score

    result["detail"] = f"{result['sector_index_perf']}({index_score})+{peer_detail}({peer_score})"



    return result





#  ?



def detect_fake_signal(

    df: pd.DataFrame,

    pattern_result: dict[str, Any],

    accel_result: dict[str, Any],

    volume_result: dict[str, Any],

    params: Optional[dict] = None,

) -> dict[str, Any]:

    """Detect potential fake signals to filter out."""
    if params is None:

        params = load_params()



    result = {"fake_score": 0, "is_likely_fake": False,

              "flags": [], "detail": ""}



    if df is None or len(df) < 30:

        return result



    closes = df["close"].values

    volumes = df["vol"].values if "vol" in df.columns else df.get("volume", df.get("amount", pd.Series([0]*len(df)))).values

    highs = df["high"].values

    lows = df["low"].values



    fake_score = 0

    flags = []



    # 1. ?

    if volume_result.get("volume_divergence"):

        fake_score += 40

        flags.append("(-40)")



    # 2. divergence in detail
    div_detail = volume_result.get("detail", "")
    if div_detail and "divergence" in div_detail:
        fake_score += 35
        flags.append("div(-35)")



    # 3. 

    if "turnover" in df.columns:

        recent_turnover = float(df["turnover"].tail(5).mean())

        if recent_turnover > 20:

            fake_score += 30

            flags.append(f"{recent_turnover:.0f}%(-30)")



    # 4. 

    if len(closes) >= 60:

        ret_60d = (closes[-1] - closes[-60]) / closes[-60] if closes[-60] > 0 else 0

        if ret_60d > 0.5:

            fake_score += 25

            flags.append(f"{ret_60d*100:.0f}%(-25)")



    # 5. ?

    if len(highs) >= 20:

        recent_range = (highs[-1] - lows[-1]) / closes[-1] if closes[-1] > 0 else 0

        avg_range = np.mean([(highs[i] - lows[i]) / closes[i] for i in range(-20, 0)]) if all(c > 0 for c in closes[-20:]) else 0

        if avg_range > 0 and recent_range > avg_range * 3:

            fake_score += 20

            flags.append("(-20)")



    # 6. neutral/downtrend = fake
    trend = accel_result.get("trend_status", "")
    if trend == "neutral" or trend == "downtrend":
        fake_score += 25
        flags.append("tr(-25)")



    # 7. ?

    if len(volumes) >= 10 and len(closes) >= 10:

        vol_increasing = all(volumes[-i] > volumes[-i-1] for i in range(1, 4))

        price_stagnant = abs(closes[-1] - closes[-4]) / closes[-4] < 0.02 if closes[-4] > 0 else False

        if vol_increasing and price_stagnant:

            fake_score += 35

            flags.append("?(-35)")



    result["fake_score"] = fake_score

    result["is_likely_fake"] = fake_score >= 60

    result["flags"] = flags

    result["detail"] = " ".join(flags) if flags else ""

    return result





#  ?



def analyze_stock(

    ts_code: str,

    df: pd.DataFrame,

    all_signals: Optional[dict[str, Any]] = None,

    params: Optional[dict] = None,

    sector_cache: Optional[dict] = None,

    ml_scores: Optional[dict[str, int]] = None,

) -> dict[str, Any]:

    """Comprehensive stock analysis."""
    params = params or load_params()

    all_signals = all_signals or {}



    result = {

        "ts_code": ts_code,

        "detected": False,

        "total_score": 0,

        "pattern_type": None,

        "pattern_score": 0,

        "volume_score": 0,

        "sector_score": 0,

        "accel_score": 0,

        "fake_score": 0,

        "ml_score": 0,

        "final_score": 0,

        "entry_price": float(df["close"].iloc[-1]) if df is not None and len(df) > 0 else 0,

        "detail": "",

        "components": {},

        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),

    }



    if df is None or len(df) < 40:

        result["detail"] = ""

        return result



    #  ?

    # 

    patterns = []

    for detector, name in [

        (detect_vcp, "VCP"),

        (detect_v_reversal, "V"),

        (detect_w_bottom, "W"),

        (detect_platform_breakout, ""),

        (detect_n_shape, "N"),

    ]:

        p = detector(df, params)

        if p["detected"]:

            patterns.append((p["score"], name, p))



    if patterns:

        best = max(patterns, key=lambda x: x[0])

        result["pattern_type"] = best[1]

        result["pattern_score"] = best[0]

        result["components"]["pattern_detail"] = best[2]

    else:

        result["pattern_score"] = 0

        result["components"]["pattern_detail"] = {"detail": ""}



    # ?

    vol = score_volume_structure(df, params)

    result["volume_score"] = vol["score"]

    result["components"]["volume_detail"] = vol



    # 

    accel = measure_acceleration(df, params)

    result["accel_score"] = accel["score"]

    result["components"]["accel_detail"] = accel



    # ?all_signals?

    sector_ctx = score_sector_context(ts_code, all_signals, params, sector_cache)

    result["sector_score"] = sector_ctx["score"]

    result["components"]["sector_detail"] = sector_ctx



    # ML model score (LightGBM on 163 Alpha factors)

    ml_val = 0

    if ml_scores and ts_code in ml_scores:

        ml_val = ml_scores.get(ts_code, 0)

    elif ml_scores is None:

        # Lazy eval for single-stock analysis: compute on the fly

        try:

            from src.ml.predictor import get_predictor

            p = get_predictor()

            if p.is_available():

                ml_val = p.score(df)

        except Exception:

            pass

    result["ml_score"] = ml_val

    result["components"]["ml_score"] = ml_val



    # ?

    fake = detect_fake_signal(df,

        result["components"].get("pattern_detail", {}),

        accel, vol, params)

    result["fake_score"] = fake["fake_score"]

    result["components"]["fake_detail"] = fake



    #   

    raw_score = (

        params["w_price_pattern"] * result["pattern_score"]

        + params["w_volume"] * result["volume_score"]

        + params["w_sector"] * result["sector_score"]

        + params["w_acceleration"] * result["accel_score"]

        + params.get("w_ml", 0) * result["ml_score"]

    )

    result["total_score"] = int(raw_score)

    result["final_score"] = max(0, int(raw_score - fake["fake_score"]))



    # detected = ?

    # signal_grade ?classify_signal ?

    result["detected"] = result["pattern_type"] is not None



    # 

    details = []

    best_pattern = result["components"].get("pattern_detail", {})

    if isinstance(best_pattern, dict):

        details.append(best_pattern.get("detail", ""))

    details.append(vol.get("detail", ""))

    details.append(accel.get("detail", ""))

    details.append(sector_ctx.get("detail", ""))

    if fake["is_likely_fake"]:

        details.append(f"** {fake.get('detail', '')}")

    result["detail"] = " | ".join(d for d in details if d)



    return result





#   



def classify_signal(signal: dict, params: Optional[dict] = None) -> str:

    """Classify signal: STRONG / WEAK / NONE / FAKE."""
    if params is None:

        params = load_params()

    if not signal.get("detected"):

        return "NONE"

    fake_th = params.get("fake_threshold", 80)
    if signal.get("fake_score", 0) >= fake_th:
        return "FAKE"

    if signal.get("final_score", 0) >= params["strong_signal"]:

        return "STRONG"

    if signal.get("final_score", 0) >= params["weak_signal"]:

        return "WEAK"

    return "NONE"





if __name__ == "__main__":

    print("Surge Engine loaded.")

    print(f"Default params: {json.dumps(load_params(), ensure_ascii=False, indent=2)}")

