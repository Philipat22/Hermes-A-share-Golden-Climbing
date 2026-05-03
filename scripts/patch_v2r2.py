"""Create v2r2: copy v2 with sideways threshold 0.25 -> 0.35"""
import shutil

src = r'D:\AIHedgeFund\ai-hedge-fund-main\scripts\backtest_v2_regime_risk.py'
dst = r'D:\AIHedgeFund\ai-hedge-fund-main\scripts\backtest_v2r2.py'
shutil.copy2(src, dst)

with open(dst, 'r', encoding='utf-8') as f:
    content = f.read()

# Change sideways threshold 0.25 -> 0.35
old = "'sideways':    ('5d_10%',  0.25, 5,  0.60),"
new = "'sideways':    ('5d_10%',  0.35, 5,  0.60),"
assert old in content, "Old pattern not found!"
content = content.replace(old, new)

# Change output filenames
content = content.replace('backtest_v2_', 'backtest_v2r2_')

# Also fix the comparison at the end that loads backtest_v2_dual_trades.csv
# Since this is a new run, we'll just not compare with old results
# The comparison block references backtest_v2r2_dual_trades.csv now

with open(dst, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done. Sideways threshold: 0.25 -> 0.35")
print(f"Output files: backtest_v2r2_*")
