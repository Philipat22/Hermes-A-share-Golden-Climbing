"""Find and fix unterminated string patterns from corruption."""
import re

with open('src/surge/engine.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixes = 0
for i, line in enumerate(lines):
    stripped = line.rstrip('\n\r')
    
    # Line ending with = "? or += "? 
    if re.search(r'(?:=|\+=)\s*"\s*\?\s*$', stripped):
        lines[i] = re.sub(r'"\s*\?\s*$', '""', stripped) + '\n'
        fixes += 1
        print(f'Line {i+1}: ASSIGN')
    
    # String ending in " ? 
    elif re.search(r'"\s*\?$', stripped) and not stripped.strip().endswith('"""'):
        lines[i] = re.sub(r'"\s*\?$', '""', stripped) + '\n'
        fixes += 1
        print(f'Line {i+1}: INLINE')

if fixes:
    with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'\nFixed {fixes} lines total')
else:
    print('No fixes needed')
