"""Fix detect_fake_signal corruption: always-true conditions."""
import re

with open('src/surge/engine.py', 'r', encoding='utf-8') as f:
    s = f.read()

# Fix 1: Check 2 - empty string in detail (always true)
old1 = """    # 2.

    if \"\" in volume_result.get(\"detail\", \"\"):

        fake_score += 35

        flags.append(\"(-35)\")"""

new1 = """    # 2. divergence in detail
    div_detail = volume_result.get(\"detail\", \"\")
    if div_detail and \"divergence\" in div_detail:
        fake_score += 35
        flags.append(\"div(-35)\")"""

# Fix 2: Check 6 - empty trend_status (always true)
old2 = """    # 6. ?

    if accel_result.get(\"trend_status\") == \"\":

        fake_score += 25

        flags.append(\"(-25)\")"""

new2 = """    # 6. neutral/downtrend = fake
    trend = accel_result.get(\"trend_status\", \"\")
    if trend == \"neutral\" or trend == \"downtrend\":
        fake_score += 25
        flags.append(\"tr(-25)\")"""

count1 = s.count(old1)
count2 = s.count(old2)
print(f'Pattern 1 found: {count1} times')
print(f'Pattern 2 found: {count2} times')

if count1 == 1:
    s = s.replace(old1, new1)
    print('Fixed pattern 1')
if count2 == 1:
    s = s.replace(old2, new2)
    print('Fixed pattern 2')

with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
    f.write(s)

import ast
try:
    ast.parse(s)
    print('*** AST OK ***')
except SyntaxError as e:
    print(f'ERROR L{e.lineno}: {e.msg}')
