"""Analyze f-string corruption in engine.py backup."""
import re
import sys

with open('src/surge/engine.py.bak', 'rb') as f:
    raw = f.read()
if raw[:3] == b'\xef\xbb\xbf':
    raw = raw[3:]
text = raw.decode('utf-8', errors='replace')
lines = text.split('\n')

bad = 0
for i, l in enumerate(lines):
    has_fstring = False
    for j in range(len(l) - 1):
        if l[j:j+2] == 'f"':
            has_fstring = True
            break
    if has_fstring:
        opens = l.count('{')
        closes = l.count('}')
        if opens != closes:
            bad += 1
            if bad <= 8:
                ascii_part = ''.join(c for c in l if ord(c) < 128)
                print(f'L{i+1}: opens={opens} closes={closes}: {ascii_part[:120]}')

print(f'\nTotal f-string brace mismatches: {bad}')

# Also check unterminated strings
print('\n--- Unterminated strings ---')
for i, l in enumerate(lines):
    # Count double quotes
    dq = 0
    in_f = False
    for j in range(len(l)):
        if l[j:j+2] == 'f"':
            in_f = True
        if l[j] == '"':
            dq += 1
    if dq % 2 != 0:
        ascii_part = ''.join(c for c in l if ord(c) < 128)
        print(f'L{i+1}: odd quotes ({dq}): {ascii_part[:100]}')
