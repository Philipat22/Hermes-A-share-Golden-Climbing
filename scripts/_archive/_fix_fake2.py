"""Fix check 2 corruption with correct trailing space."""
with open('src/surge/engine.py', 'r', encoding='utf-8') as f:
    s = f.read()

# Pattern 1 with trailing space after "2."
old1 = '    # 2. \n\n    if "" in volume_result.get("detail", ""):\n\n        fake_score += 35\n\n        flags.append("(-35)")'

new1 = '    # 2. divergence in detail\n    div_detail = volume_result.get("detail", "")\n    if div_detail and "divergence" in div_detail:\n        fake_score += 35\n        flags.append("div(-35)")'

count = s.count(old1)
print(f'Pattern found: {count}')
if count == 1:
    s = s.replace(old1, new1)
    print('Fixed!')
else:
    # Debug: show context around it
    idx = s.find('    # 2.')
    if idx >= 0:
        print(f'Found at offset {idx}')
        print(repr(s[idx:idx+120]))

# Also check: the "fak e_score" typo in result dict
# result = {"fak e_score": 0, ...} should be "fake_score"
s = s.replace('"fak e_score"', '"fake_score"')

with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
    f.write(s)

import ast
try:
    ast.parse(s)
    print('*** AST OK ***')
except SyntaxError as e:
    print(f'ERROR L{e.lineno}: {e.msg}')
