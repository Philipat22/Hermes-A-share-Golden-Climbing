"""Fix all remaining f-string / string corruption patterns in engine.py."""
import re

with open('src/surge/engine.py', 'r', encoding='utf-8') as f:
    s = f.read()

# Fix: f-string with unterminated else clause: '? -> ''
s = s.replace("""f"{'' if vol_confirmed else '?}""", """f"{'' if vol_confirmed else ''}""")
s = s.replace("""f"{'' if vol_confirmed else '?"}""", """f"{'' if vol_confirmed else ''}""")

# Fix: '? 3%' -> ''   (corrupted comments in strings)
s = re.sub(r"""['\"]\s*\?\s*[0-9]%['\"]""", '""', s)

# Fix: '? days' -> ''
s = re.sub(r"""['\"]\s*\?\s*days""", "''", s)

# General: any line ending = "? or += "?
s = re.sub(r'(=\s*)"\s*\?', r'\1""', s)
s = re.sub(r'(\+=\s*)"\s*\?', r'\1""', s)

# Fix: multi-line docstring left-overs: single """ on a line that should be code
lines = s.split('\n')
fixed = []
in_comment_block = False
for i, line in enumerate(lines):
    stripped = line.strip()
    # If a line is just """ and previous line also has self-contained """...
    if stripped == '"""' and i > 0:
        prev = lines[i-1].strip()
        if prev.endswith('"""') and prev != '"""':
            # This is a stray docstring marker, skip it
            print(f'L{i+1}: skipping stray """')
            continue
    fixed.append(line)

s = '\n'.join(fixed)

with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
    f.write(s)

print(f'Wrote {len(fixed)} lines')

# Verify
import ast
try:
    ast.parse(s)
    print('*** AST OK ***')
except SyntaxError as e:
    print(f'ERROR: L{e.lineno}: {e.msg}')
    lines = s.split('\n')
    for j in range(max(0,e.lineno-3), min(len(lines), e.lineno+2)):
        print(f'  L{j+1}: {lines[j]}')
