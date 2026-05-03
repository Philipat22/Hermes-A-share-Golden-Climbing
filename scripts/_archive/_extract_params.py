"""Extract default param values from engine.py."""
import re

with open('src/surge/engine.py', 'r') as f:
    s = f.read()

# Find param assignments that look like defaults
defaults = {}
for m in re.finditer(r'params\.get\(["\'](\w+)[\"\'],\s*([^)]+)\)', s):
    key = m.group(1)
    val = m.group(2).strip()
    defaults[key] = val

for k, v in sorted(defaults.items()):
    print(f'{k}: {v}')
