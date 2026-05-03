"""Fix engine.py line 942: unterminated f-string else clause."""
with open('src/surge/engine.py', 'r') as f:
    s = f.read()

# The exact corrupted text
old = "f\"{'' if vol_break else '?}\""
new = "f\"{'' if vol_break else ''}\""

if old in s:
    s = s.replace(old, new)
    print('Fixed line 942')
else:
    print('Exact pattern not found, trying broader match')
    # Find it via byte search
    idx = s.find("vol_break else '?}")
    if idx >= 0:
        print(f'Found at offset {idx}')
        print(f'Before: ...{s[max(0,idx-10):idx]}...')
        print(f'Match: {s[idx:idx+30]}')
    else:
        print('Not found at all')
        # Check all vol_break occurrences
        for m in re.finditer(r"vol_break", s):
            start = m.start()
            print(f'Found vol_break at {start}: ...{s[start:start+50]}...')

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
