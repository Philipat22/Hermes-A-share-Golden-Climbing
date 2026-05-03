"""Final nuclear fix: extract only ASCII Python code, add minimal docs."""
import re, sys

with open('src/surge/engine.py.bak', 'rb') as f:
    raw = f.read()
if raw[:3] == b'\xef\xbb\xbf':
    raw = raw[3:]
s = raw.decode('utf-8', errors='replace')

# Fix known f-string brace issues
for old, new in [
    ('max_amp*100:.0f}%', '{max_amp*100:.0f}%'),
    ('first_return*100:.1f}%', '{first_return*100:.1f}%'),
    ('short_vol*100:.2f}%', '{short_vol*100:.2f}%'),
    ('acceleration*100:.1f}%', '{acceleration*100:.1f}%'),
    ('recent_turnover:.0f}%(-30)', '{recent_turnover:.0f}%(-30)'),
    ('60ret_60d*100:.0f}%(-25)', '{60ret_60d*100:.0f}%(-25)'),
]:
    s = s.replace(old, new)

# Fix unterminated strings
s = re.sub(r'result\["detail"\]\s*=\s*"\s*\?', 'result["detail"] = ""', s)
s = re.sub(r'result\["trend_status"\]\s*=\s*"\?', 'result["trend_status"] = "unknown"', s)
for old, new in [('"V?,', '"V-reversal",'), ('"W?,', '"W-bottom",'), ('"N?,', '"N-shape",')]:
    s = s.replace(old, new)

# Remove bare corrupted docstrings like: _SECTOR_CACHE_LAST\n"""(id(all_signals), {sector: {strong, weak, total}})"""
s = re.sub(r'\(all_signals\).*?\)"""', '"""', s)
s = re.sub(r'SECTOR_CACHE_LAST.*?\n', '_SECTOR_CACHE_LAST: tuple = (None, None)\n', s)

# Now extract clean Python: keep only ASCII lines, handle docstrings
lines = s.split('\n')

DOCS = {
    "detect_platform_breakout": '    """Detect platform/consolidation breakout pattern."""',
    "detect_n_shape":           '    """Detect N-shape (cup and handle) pattern."""',
    "detect_vcp":               '    """Detect Volatility Contraction Pattern (VCP)."""',
    "detect_v_reversal":        '    """Detect V-shaped reversal pattern."""',
    "detect_w_bottom":          '    """Detect W-bottom (double bottom) pattern."""',
    "measure_acceleration":     '    """Measure price acceleration."""',
    "score_volume_structure":   '    """Score volume structure for signal quality."""',
    "detect_fake_signal":       '    """Detect potential fake signals to filter out."""',
    "score_sector_context":     '    """Score sector/market context."""',
    "analyze_stock":            '    """Comprehensive stock analysis."""',
    "classify_signal":          '    """Classify signal: STRONG / WEAK / NONE / FAKE."""',
    "load_params":              '    """Load parameters from file, fall back to defaults."""',
    "save_params":              '    """Save current parameters to file."""',
    "_build_sector_cache":     '    """Build sector signal density cache."""',
}

# State machine approach: track docstring boundaries, replace content
result = []
in_docstring = False
pending_fn = ''

for i, line in enumerate(lines):
    stripped = line.strip()
    has_triple = '"""' in stripped
    
    # Detect function definition
    m = re.search(r'def (\w+)', line)
    if m:
        pending_fn = m.group(1)
    
    if has_triple:
        tq_count = stripped.count('"""')
        
        if in_docstring:
            # We're closing a docstring
            result.append('    """')  # closing
            in_docstring = False
            continue
        
        # Opening a docstring
        if tq_count >= 2:
            # Self-contained: """text""" - replace entirely
            indent = '    ' if stripped.startswith(' ') else ''
            if pending_fn in DOCS:
                result.append(DOCS[pending_fn])
            elif indent:
                result.append(indent + '""""""')
            else:
                result.append('"""Surge pattern detection engine."""')
            # Might be inline with both open and close
            # Check if the last 3 chars are """ (close)
            if stripped.endswith('"""') and tq_count == 2:
                pass  # We already replaced it
            elif stripped.endswith('"""') and tq_count > 2:
                pass
            else:
                # Has more opening - mark as in docstring
                in_docstring = True
            continue
        
        # Single triple-quote: opener
        indent = '    ' if stripped.startswith(' ') else ''
        if pending_fn in DOCS:
            result.append(DOCS[pending_fn])
        elif indent:
            result.append(indent + '""""""')
        else:
            result.append('"""Module docstring."""')
        in_docstring = True
        continue
    
    # Skip docstring content (was replaced)
    if in_docstring:
        continue
    
    # Check if ASCII
    try:
        line.encode('ascii')
        result.append(line)
        continue
    except UnicodeEncodeError:
        pass
    
    # Non-ASCII: extract ASCII parts
    ascii_part = ''.join(c for c in line if ord(c) < 128 and c != '\ufffd')
    if ascii_part.strip():
        result.append(ascii_part)
    # else pure corruption, drop

# Close any unclosed docstring
if in_docstring:
    result.append('    """')

output = '\n'.join(result)
output = re.sub(r'\n{4,}', '\n\n\n', output)

with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
    f.write(output)

rlines = output.count('\n') + 1
print(f'Wrote {rlines} lines')

# Verify
try:
    import py_compile
    py_compile.compile('src/surge/engine.py', doraise=True)
    print('*** SYNTAX OK ***')
except py_compile.PyCompileError as e:
    print(f'SYNTAX ERROR: {e}')
    em = re.search(r'line (\d+)', str(e))
    if em:
        ln = int(em.group(1))
        # Account for potential line count mismatch
        lst = output.split('\n')
        for j in range(max(0,ln-4), min(len(lst), ln+5)):
            print(f'  L{j+1}: {lst[j][:120]}')
