"""Quick test of engine signal output."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import pickle
import pandas as pd

with open(os.path.join('data', 'cache', 'backtest_prices.pkl'), 'rb') as f:
    prices = pickle.load(f)

from src.surge.engine import analyze_stock, load_params, classify_signal
params = load_params()
print('Params:', params)

# Test a few stocks
test_codes = list(prices.keys())[:5]
for code in test_codes:
    df = prices[code]
    if 'date' in df.columns:
        sliced = df[df['date'] <= '2026-03-04'].copy()
        if len(sliced) > 120:
            sliced = sliced.tail(120)
        sig = analyze_stock(code, sliced, {}, params)
        cls = classify_signal(sig, params)
        print(f'{code}: detected={sig.get("detected")} score={sig.get("final_score")} '
              f'pattern={sig.get("pattern_type")} fake={sig.get("fake_score",0)} => {cls}')

# Check total signal distribution across full pool
print('\n--- Full pool scan ---')
counts = {'STRONG': 0, 'WEAK': 0, 'NONE': 0, 'FAKE': 0, 'ERROR': 0}
for code in list(prices.keys())[:100]:
    df = prices[code]
    if 'date' not in df.columns:
        continue
    try:
        sliced = df[df['date'] <= '2026-03-04'].copy()
        if len(sliced) > 120:
            sliced = sliced.tail(120)
        if len(sliced) < 60:
            counts['ERROR'] += 1
            continue
        sig = analyze_stock(code, sliced, {}, params)
        cls = classify_signal(sig, params)
        counts[cls] = counts.get(cls, 0) + 1
    except Exception as e:
        counts['ERROR'] += 1

print(f'Distribution: {counts}')
