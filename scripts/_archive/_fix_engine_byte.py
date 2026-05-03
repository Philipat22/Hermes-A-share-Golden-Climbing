"""Targeted byte-level fixes for engine.py corruption."""
import re, os

path = 'src/surge/engine.py.bak'
with open(path, 'rb') as f:
    raw = f.read()

if raw[:3] == b'\xef\xbb\xbf':
    raw = raw[3:]

s = raw.decode('utf-8', errors='replace')

# Fix 1-6: Missing opening braces in f-strings
fixes = [
    ('max_amp*100:.0f}%', '{max_amp*100:.0f}%'),
    ('first_return*100:.1f}%', '{first_return*100:.1f}%'),
    ('short_vol*100:.2f}%', '{short_vol*100:.2f}%'),
    ('acceleration*100:.1f}%', '{acceleration*100:.1f}%'),
    ('recent_turnover:.0f}%(-30)', '{recent_turnover:.0f}%(-30)'),
    ('60ret_60d*100:.0f}%(-25)', '{60ret_60d*100:.0f}%(-25)'),
]

count = 0
for old, new in fixes:
    if old in s:
        s = s.replace(old, new)
        count += 1

# Fix 7: Unterminated result strings ending in " ?
s = re.sub(r'result\["detail"\]\s*=\s*"\s*\?', 'result["detail"] = ""', s)
s = re.sub(r'result\["trend_status"\]\s*=\s*"\?', 'result["trend_status"] = "unknown"', s)

# Fix 8: Pattern label corruption
for old, new in [('"V?,', '"V-reversal",'), ('"W?,', '"W-bottom",'), ('"N?,', '"N-shape",')]:
    if old in s:
        s = s.replace(old, new)
        count += 1

# Fix 9: Detail dict corruption
s = re.sub(r'\{"detail": "\?\}', '{"detail": "unknown"}', s)

# Fix 10: Bare dict-like docstrings that are actually corrupted comments
# Pattern: _SECTOR_CACHE_LAST: tuple = (None, None)\n"""(id(all_signals), {sector: {strong, weak, total}})"""
s = re.sub(
    r'_SECTOR_CACHE_LAST: tuple = \(None, None\)\s*""".*?"""',
    '_SECTOR_CACHE_LAST: tuple = (None, None)\n    """Cache sector data."""',
    s,
)

print(f'Applied {count} replacements')

# Clean: strip all lines that are pure non-ASCII corruption
lines = s.split('\n')
cleaned = []
for line in lines:
    try:
        line.encode('ascii')
        cleaned.append(line)
        continue
    except UnicodeEncodeError:
        pass
    
    # Has non-ASCII content
    # If it has any ASCII Python code structure, keep it
    ascii_part = ''.join(c for c in line if ord(c) < 128 and c != '\ufffd')
    
    if not ascii_part.strip():
        continue  # Pure corruption
    
    # Keep only ASCII part
    cleaned.append(ascii_part)

result = '\n'.join(cleaned)
result = re.sub(r'\n{3,}', '\n\n', result)

# Write
outpath = 'src/surge/engine.py'
with open(outpath, 'w', encoding='utf-8') as f:
    f.write(result)

rlines = len(result.split('\n'))
print(f'Wrote {rlines} lines')

# Verify
import py_compile
try:
    py_compile.compile(outpath, doraise=True)
    print('*** SYNTAX OK ***')
except py_compile.PyCompileError as e:
    print(f'SYNTAX ERROR: {e}')
    em = re.search(r'line (\d+)', str(e))
    if em:
        ln = int(em.group(1))
        lst = result.split('\n')
        for j in range(max(0,ln-3), min(len(lst), ln+4)):
            print(f'  L{j+1}: {lst[j][:120]}')
