"""Fix specific corruption in engine.py lines 441-447."""
import re

with open('src/surge/engine.py', 'r', encoding='utf-8') as f:
    s = f.read()

fixes = [
    # Line 441: f"N??{{first_return*100:.1f}% "
    ('f"N??{{first_return*100:.1f}% "', 'f"N {first_return*100:.1f}% "'),
    # Line 445: f"{'' if vol_shrink else '?} "
    ("f\"{'' if vol_shrink else '?} \"", "f\"{'' if vol_shrink else ''} \""),
    # Line 447: f"?{second_return*100:.1f}%"
    ('f"?{second_return*100:.1f}%"', 'f"{second_return*100:.1f}%"'),
]

for old, new in fixes:
    if old in s:
        s = s.replace(old, new)
        print(f'Fixed: {old[:50]}...')
    else:
        print(f'NOT FOUND: {old[:50]}...')
        # Try to find partial match
        for pattern_part in old.split(' ')[:-1]:
            if pattern_part in s:
                idx = s.index(pattern_part)
                print(f'  Partial match at offset {idx}: ...{s[idx-10:idx+50]}...')

with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
    f.write(s)

# Verify
import ast
try:
    ast.parse(s)
    print('\n*** AST OK ***')
except SyntaxError as e:
    print(f'\nERROR: L{e.lineno}: {e.msg}')
    lines = s.split('\n')
    for j in range(max(0,e.lineno-2), min(len(lines), e.lineno+2)):
        print(f'  L{j+1}: {lines[j]}')
