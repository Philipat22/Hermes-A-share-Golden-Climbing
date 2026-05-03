"""Check exact bytes around check 2."""
with open('src/surge/engine.py', 'r') as f:
    lines = f.readlines()
for i in [1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566]:
    raw = lines[i].encode('utf-8')
    print(f'L{i+1}: hex={raw.hex()} | {repr(lines[i].rstrip())}')
