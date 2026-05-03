"""
WorldQuant Alpha 101 factor expressions.
Compact dictionary format: {name: expression_string}
"""
import polars as pl

# Helper expression for returns
_R: str = "(close / ts_delay(close, 1) - 1)"

ALPHA_101: dict[str, str] = {
    # === Core signal factors (1-20) ===
    "alpha1":  f"(cs_rank(ts_argmax(pow1(quesval(0, {_R}, close, ts_std({_R}, 20)), 2.0), 5)) - 0.5)",
    "alpha2":  "(-1) * ts_corr(cs_rank(ts_delta(log(volume), 2)), cs_rank((close - open) / open), 6)",
    "alpha3":  "ts_corr(cs_rank(open), cs_rank(volume), 10) * -1",
    "alpha4":  "-1 * ts_rank(cs_rank(low), 9)",
    "alpha5":  "cs_rank((open - (ts_sum(vwap, 10) / 10))) * (-1 * abs(cs_rank((close - vwap))))",
    "alpha6":  "(-1) * ts_corr(open, volume, 10)",
    "alpha7":  "quesval2(ts_mean(volume, 20), volume, (-1 * ts_rank(abs(close - ts_delay(close, 7)), 60)) * sign(ts_delta(close, 7)), -1)",
    "alpha8":  f"-1 * cs_rank(((ts_sum(open, 5) * ts_sum({_R}, 5)) - ts_delay((ts_sum(open, 5) * ts_sum({_R}, 5)), 10)))",
    "alpha9":  "quesval(0, ts_min(ts_delta(close, 1), 5), ts_delta(close, 1), quesval(0, ts_max(ts_delta(close, 1), 5), (-1 * ts_delta(close, 1)), ts_delta(close, 1)))",
    "alpha10": "cs_rank(quesval(0, ts_min(ts_delta(close, 1), 4), ts_delta(close, 1), quesval(0, ts_max(ts_delta(close, 1), 4), (-1 * ts_delta(close, 1)), ts_delta(close, 1))))",
    "alpha11": "(cs_rank(ts_max(vwap - close, 3)) + cs_rank(ts_min(vwap - close, 3))) * cs_rank(ts_delta(volume, 3))",
    "alpha12": "sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1))",
    "alpha13": "-1 * cs_rank(ts_cov(cs_rank(close), cs_rank(volume), 5))",
    "alpha14": f"(-1 * cs_rank(({_R}) - ts_delay({_R}, 3))) * ts_corr(open, volume, 10)",
    "alpha15": "-1 * ts_sum(cs_rank(ts_corr(cs_rank(high), cs_rank(volume), 3)), 3)",
    "alpha16": "-1 * cs_rank(ts_cov(cs_rank(high), cs_rank(volume), 5))",
    "alpha17": "(-1 * cs_rank(ts_rank(close, 10))) * cs_rank(close - 2 * ts_delay(close, 1) + ts_delay(close, 2)) * cs_rank(ts_rank(volume / ts_mean(volume, 20), 5))",
    "alpha18": "-1 * cs_rank((ts_std(abs(close - open), 5) + (close - open)) + ts_corr(close, open, 10))",
    "alpha19": f"(-1 * sign(ts_delta(close, 7) + (close - ts_delay(close, 7)))) * (cs_rank(ts_sum({_R}, 250) + 1) + 1)",
    "alpha20": "(-1 * cs_rank(open - ts_delay(high, 1))) * cs_rank(open - ts_delay(close, 1)) * cs_rank(open - ts_delay(low, 1))",

    # === Conditional/Variable factors (21-40) ===
    "alpha21": "quesval2((ts_mean(close, 8) + ts_std(close, 8)), ts_mean(close, 2), -1, quesval2(ts_mean(close, 2), (ts_mean(close, 8) - ts_std(close, 8)), 1, quesval(1, (volume / ts_mean(volume, 20)), 1, -1)))",
    "alpha22": "-1 * ts_delta(ts_corr(high, volume, 5), 5) * cs_rank(ts_std(close, 20))",
    "alpha23": "quesval2(ts_mean(high, 20), high, -1 * ts_delta(high, 2), 0)",
    "alpha24": "quesval(0.05, ts_delta(ts_sum(close, 100) / 100, 100) / ts_delay(close, 100), (-1 * ts_delta(close, 3)), (-1 * (close - ts_min(close, 100))))",
    "alpha25": f"cs_rank( (-1 * {_R}) * ts_mean(volume, 20) * vwap * (high - close) )",
    "alpha26": "-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
    "alpha27": "quesval(0.5, cs_rank(ts_mean(ts_corr(cs_rank(volume), cs_rank(vwap), 6), 2)), -1, 1)",
    "alpha28": "cs_scale(ts_corr(ts_mean(volume, 20), low, 5) + (high + low) / 2 - close)",
    "alpha29": f"ts_min(ts_product(cs_rank(cs_rank(cs_scale(log(ts_sum(ts_min(cs_rank(cs_rank((-1 * cs_rank(ts_delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(ts_delay((-1 * {_R}), 6), 5)",
    "alpha30": "((cs_rank(sign(close - ts_delay(close, 1)) + sign(ts_delay(close, 1) - ts_delay(close, 2)) + sign(ts_delay(close, 2) - ts_delay(close, 3))) * -1 + 1) * ts_sum(volume, 5)) / ts_sum(volume, 20)",

    # === Complex composite factors (31-47) ===
    "alpha31": "(cs_rank(cs_rank(cs_rank(ts_decay_linear((-1) * cs_rank(cs_rank(ts_delta(close, 10))), 10)))) + cs_rank((-1) * ts_delta(close, 3))) + sign(cs_scale(ts_corr(ts_mean(volume, 20), low, 12)))",
    "alpha32": "cs_scale((ts_sum(close, 7) / 7 - close)) + (20 * cs_scale(ts_corr(vwap, ts_delay(close, 5), 230)))",
    "alpha33": "cs_rank((-1) * (open / close * -1 + 1))",
    "alpha34": f"cs_rank((cs_rank(ts_std({_R}, 2) / ts_std({_R}, 5)) * -1 + 1) + (cs_rank(ts_delta(close, 1)) * -1 + 1))",
    "alpha35": f"(ts_rank(volume, 32) * (ts_rank((close + high - low), 16) * -1 + 1)) * (ts_rank({_R}, 32) * -1 + 1)",
    "alpha36": f"((((2.21 * cs_rank(ts_corr((close - open), ts_delay(volume, 1), 15))) + (0.7 * cs_rank((open - close)))) + (0.73 * cs_rank(ts_rank(ts_delay((-1) * {_R}, 6), 5)))) + cs_rank(abs(ts_corr(vwap, ts_mean(volume, 20), 6)))) + (0.6 * cs_rank(((ts_sum(close, 200) / 200 - open) * (close - open))))",
    "alpha37": "cs_rank(ts_corr(ts_delay((open - close), 1), close, 200)) + cs_rank((open - close))",
    "alpha38": "((-1) * cs_rank(ts_rank(close, 10))) * cs_rank((close / open))",
    "alpha39": f"((-1) * cs_rank((ts_delta(close, 7) * (cs_rank(ts_decay_linear((volume / ts_mean(volume, 20)), 9)) * -1 + 1)))) * (cs_rank(ts_sum({_R}, 250)) + 1)",
    "alpha40": "((-1) * cs_rank(ts_std(high, 10))) * ts_corr(high, volume, 10)",
    "alpha41": "pow1((high * low), 0.5) - vwap",
    "alpha42": "cs_rank((vwap - close)) / cs_rank((vwap + close))",
    "alpha43": "ts_rank((volume / ts_mean(volume, 20)), 20) * ts_rank((-1) * ts_delta(close, 7), 8)",
    "alpha44": "(-1) * ts_corr(high, cs_rank(volume), 5)",
    "alpha45": "(-1) * cs_rank(ts_sum(ts_delay(close, 5), 20) / 20) * ts_corr(close, volume, 2) * cs_rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))",
    "alpha46": "quesval(0.25, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), -1, quesval(0, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1))",
    "alpha47": "((cs_rank(pow1(close, -1)) * volume / ts_mean(volume, 20)) * (high * cs_rank(high - close)) / (ts_sum(high, 5) / 5)) - cs_rank(vwap - ts_delay(vwap, 5))",

    # === Momentum/momentum factors (49-57) ===
    "alpha49": "quesval(-0.1, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1)",
    "alpha50": "(-1) * ts_max(cs_rank(ts_corr(cs_rank(volume), cs_rank(vwap), 5)), 5)",
    "alpha51": "quesval(-0.05, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1)",
    "alpha52": f"(((-1) * ts_min(low, 5)) + ts_delay(ts_min(low, 5), 5)) * cs_rank((ts_sum({_R}, 240) - ts_sum({_R}, 20)) / 220) * ts_rank(volume, 5)",
    "alpha53": "(-1) * ts_delta(((close - low) - (high - close)) / (close - low), 9)",
    "alpha54": "((-1) * ((low - close) * pow1(open, 5))) / ((low - high) * pow1(close, 5))",
    "alpha55": "(-1) * ts_corr(cs_rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), cs_rank(volume), 6)",
    "alpha57": "-1 * ((close - vwap) / ts_decay_linear(cs_rank(ts_argmax(close, 30)), 2))",

    # === Volume/volatility factors (60-66) ===
    "alpha60": "- 1 * ((2 * cs_scale(cs_rank((((close - low) - (high - close)) / (high - low)) * volume))) - cs_scale(cs_rank(ts_argmax(close, 10))))",
    "alpha61": "quesval2(cs_rank(vwap - ts_min(vwap, 16)), cs_rank(ts_corr(vwap, ts_mean(volume, 180), 18)), 1, 0)",
    "alpha62": "(cs_rank(ts_corr(vwap, ts_sum(ts_mean(volume, 20), 22), 10)) < cs_rank((cs_rank(open) + cs_rank(open)) < (cs_rank((high + low) / 2) + cs_rank(high)))) * -1",
    "alpha64": "(cs_rank(ts_corr(ts_sum(((open * 0.178404) + (low * (1 - 0.178404))), 13), ts_sum(ts_mean(volume, 120), 13), 17)) < cs_rank(ts_delta((((high + low) / 2 * 0.178404) + (vwap * (1 - 0.178404))), 4))) * -1",
    "alpha65": "(cs_rank(ts_corr(((open * 0.00817205) + (vwap * (1 - 0.00817205))), ts_sum(ts_mean(volume, 60), 9), 6)) < cs_rank(open - ts_min(open, 14))) * -1",
    "alpha66": "(cs_rank(ts_decay_linear(ts_delta(vwap, 4), 7)) + ts_rank(ts_decay_linear((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2)), 11), 7)) * -1",
    "alpha68": "(ts_rank(ts_corr(cs_rank(high), cs_rank(ts_mean(volume, 15)), 9), 14) < cs_rank(ts_delta((close * 0.518371 + low * (1 - 0.518371)), 1))) * -1",

    # === Cross-sectional/ranking factors (71-86) ===
    "alpha71": "ts_greater(ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 3), ts_rank(ts_mean(volume, 180), 12), 18), 4), 16), ts_rank(ts_decay_linear(pow1(cs_rank((low + open) - (vwap + vwap)), 2), 16), 4))",
    "alpha72": "cs_rank(ts_decay_linear(ts_corr((high + low) / 2, ts_mean(volume, 40), 9), 10)) / cs_rank(ts_decay_linear(ts_corr(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))",
    "alpha73": "ts_greater(cs_rank(ts_decay_linear(ts_delta(vwap, 5), 3)), ts_rank(ts_decay_linear((ts_delta(open * 0.147155 + low * 0.852845, 2) / (open * 0.147155 + low * 0.852845)) * -1, 3), 17)) * -1",
    "alpha74": "quesval2(cs_rank(ts_corr(close, ts_sum(ts_mean(volume, 30), 37), 15)), cs_rank(ts_corr(cs_rank(high * 0.0261661 + vwap * 0.9738339), cs_rank(volume), 11)), 1, 0) * -1",
    "alpha75": "quesval2(cs_rank(ts_corr(vwap, volume, 4)), cs_rank(ts_corr(cs_rank(low), cs_rank(ts_mean(volume, 50)), 12)), 1, 0)",
    "alpha77": "ts_less(cs_rank(ts_decay_linear((((high + low) / 2 + high) - (vwap + high)), 20)), cs_rank(ts_decay_linear(ts_corr((high + low) / 2, ts_mean(volume, 40), 3), 6)))",
    "alpha78": "pow2(cs_rank(ts_corr(ts_sum((low * 0.352233) + (vwap * (1 - 0.352233)), 20), ts_sum(ts_mean(volume, 40), 20), 7)), cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), 6)))",
    "alpha81": "quesval2(cs_rank(log(ts_product(cs_rank(pow1(cs_rank(ts_corr(vwap, ts_sum(ts_mean(volume, 10), 50), 8)), 4)), 15))), cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), 5)), 1, 0) * -1",
    "alpha83": "(cs_rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * cs_rank(cs_rank(volume))) / (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))",
    "alpha84": "pow2(ts_rank(vwap - ts_max(vwap, 15), 21), ts_delta(close, 5))",
    "alpha85": "pow2(cs_rank(ts_corr(high * 0.876703 + close * 0.123297, ts_mean(volume, 30), 10)), cs_rank(ts_corr(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))",
    "alpha86": "quesval2(ts_rank(ts_corr(close, ts_sum(ts_mean(volume, 20), 15), 6), 20), cs_rank((open + close) - (vwap + open)), 1, 0) * -1",
    "alpha88": "ts_less(cs_rank(ts_decay_linear((cs_rank(open) + cs_rank(low)) - (cs_rank(high) + cs_rank(close)), 8)), ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 21), 8), 7), 3))",

    # === Remaining factors (92-101) ===
    "alpha92": "ts_less(ts_rank(ts_decay_linear(quesval2(((high + low) / 2 + close), (low + open), 1, 0), 15), 19), ts_rank(ts_decay_linear(ts_corr(cs_rank(low), cs_rank(ts_mean(volume, 30)), 8), 7), 7))",
    "alpha94": "pow2(cs_rank(vwap - ts_min(vwap, 12)), ts_rank(ts_corr(ts_rank(vwap, 20), ts_rank(ts_mean(volume, 60), 4), 18), 3)) * -1",
    "alpha95": "quesval2(cs_rank(open - ts_min(open, 12)), ts_rank(pow1(cs_rank(ts_corr(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(volume, 40), 19), 13)), 5), 12), 1, 0)",
    "alpha96": "ts_greater(ts_rank(ts_decay_linear(ts_corr(cs_rank(vwap), cs_rank(volume), 4), 4), 8), ts_rank(ts_decay_linear(ts_argmax(ts_corr(ts_rank(close, 7), ts_rank(ts_mean(volume, 60), 4), 4), 13), 14), 13)) * -1",
    "alpha98": "cs_rank(ts_decay_linear(ts_corr(vwap, ts_sum(ts_mean(volume, 5), 26), 5), 7)) - cs_rank(ts_decay_linear(ts_rank(ts_argmin(ts_corr(cs_rank(open), cs_rank(ts_mean(volume, 15)), 21), 9), 7), 8))",
    "alpha99": "quesval2(cs_rank(ts_corr(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(volume, 60), 20), 9)), cs_rank(ts_corr(low, volume, 6)), 1, 0) * -1",
    "alpha101": "((close - open) / ((high - low) + 0.001))",
}

# Factors commented out in original (not implemented - need IndNeutralize or cap):
# alpha48, alpha56, alpha58, alpha59, alpha63, alpha67, alpha69, alpha70,
# alpha76, alpha79, alpha80, alpha82, alpha87, alpha89, alpha90, alpha91,
# alpha93, alpha97, alpha100

ALPHA_101_COUNT: int = len(ALPHA_101)
