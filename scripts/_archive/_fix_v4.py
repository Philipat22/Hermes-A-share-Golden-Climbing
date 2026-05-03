"""Fix line 526 f-string."""
with open('src/surge/engine.py', 'r', encoding='utf-8') as f:
    s = f.read()

old = '        result["detail"] = f": {{short_vol*100:.2f}%/{long_vol*100:.2f}% = {vol_ratio:.2f} > {vol_ratio_threshold:.2f}"'
new = '        result["detail"] = f"vol ratio: {short_vol*100:.1f}% / {long_vol*100:.1f}% = {vol_ratio:.2f} > {vol_ratio_threshold:.2f}"'

if old in s:
    s = s.replace(old, new)
    print('Fixed line 526')
else:
    print('Pattern not found exactly')
    # Search for partial
    for keyword in ['short_vol*100:.2f}%/{long_vol']:
        if keyword in s:
            idx = s.index(keyword)
            print(f'Found approx at offset {idx}')
            print(f'Context: ...{s[max(0,idx-30):idx+80]}...')

with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
    f.write(s)

import ast
try:
    ast.parse(s)
    print('\n*** AST OK ***')
except SyntaxError as e:
    print(f'\nERROR: L{e.lineno}: {e.msg}')
    lines = s.split('\n')
    for j in range(max(0,e.lineno-2), min(len(lines), e.lineno+2)):
        print(f'  L{j+1}: {lines[j]}')
