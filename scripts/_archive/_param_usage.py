"""Analyze each param usage to determine reasonable defaults."""
import re

with open('src/surge/engine.py', 'r') as f:
    s = f.read()

# Find all params occurrences with context
for m in re.finditer(r'params\[["\'](\w+)["\']\]', s):
    key = m.group(1)
    start = max(0, m.start() - 60)
    end = min(len(s), m.end() + 40)
    context = s[start:end].replace('\n', ' ').strip()
    line_num = s[:m.start()].count('\n') + 1
    print(f'L{line_num:4d} [{key}]: {context}')
