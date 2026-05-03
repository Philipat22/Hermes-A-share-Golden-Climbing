#!/usr/bin/env python3
"""Find real start of data_fetcher.py and trim garbage header"""
with open(r'D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_fetcher.py','r',encoding='utf-8') as f:
    lines = f.readlines()

# Find line 1 (index 0) - we know from __future__ is at line 67 (index 66)
# Look for the first real code line after line 67
real_start = 66  # index of 'from __future__ import annotations' at line 67
for i in range(66, len(lines)):
    stripped = lines[i].strip()
    # skip docstrings
    if stripped.startswith('from __future__') or stripped.startswith('import ') or stripped.startswith('from '):
        real_start = i
        break
    if stripped and not stripped.startswith('#') and not stripped.startswith('\"\"\"') and not stripped.startswith("'''"):
        real_start = i
        break

print(f'Total lines: {len(lines)}')
print(f'First code line index: {real_start} (line {real_start+1})')
print(f'Lines 1-{real_start+1} are garbage (repeated headers)')
print()
# Check if lines 46-66 are garbage
print('Lines 45-70:')
for i in range(44, min(70, len(lines))):
    print(f'  {i+1:3d}: {lines[i].rstrip()[:100]}')
