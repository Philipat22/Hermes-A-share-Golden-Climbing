"""Find all param keys used in engine.py."""
import sys, os, re
with open('src/surge/engine.py', 'r') as f:
    s = f.read()

# Find all params[... key patterns
keys = set()
for m in re.finditer(r'params\[\"(\w+)\"\]', s):
    keys.add(m.group(1))

print(sorted(keys))
