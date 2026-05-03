"""Minimal fix for engine.py: f-string braces + unterminated strings + clean comments."""
import re, sys

with open('src/surge/engine.py.bak', 'rb') as f:
    raw = f.read()
if raw[:3] == b'\xef\xbb\xbf':
    raw = raw[3:]

text = raw.decode('utf-8', errors='replace')
lines = text.split('\n')

# First pass: fix specific known corruption patterns
fixes = 0
for i in range(len(lines)):
    # Missing { before variable
    for old_pattern in [
        ('max_amp*100:.0f}%', '{max_amp*100:.0f}%'),
        ('first_return*100:.1f}%', '{first_return*100:.1f}%'),
        ('short_vol*100:.2f}%', '{short_vol*100:.2f}%'),
        ('acceleration*100:.1f}%', '{acceleration*100:.1f}%'),
        ('recent_turnover:.0f}%(-30)', '{recent_turnover:.0f}%(-30)'),
        ('60ret_60d*100:.0f}%(-25)', '{60ret_60d*100:.0f}%(-25)'),
    ]:
        if old_pattern[0] in lines[i]:
            lines[i] = lines[i].replace(old_pattern[0], old_pattern[1])
            fixes += 1
    
    # Unterminated string: " ? at end
    stripped = lines[i].strip()
    if 'result["detail"]' in stripped and stripped.endswith('" ?'):
        # Replace with proper empty string
        lines[i] = lines[i][:lines[i].rindex('"')] + '"'
        fixes += 1
    
    # Single-char pattern labels
    for old, new in [('"V?,', '"V-reversal",'), ('"W?,', '"W-bottom",'), ('"N?,', '"N-shape",')]:
        if old in lines[i]:
            lines[i] = lines[i].replace(old, new)
            fixes += 1

    # detail dict closing
    if '{"detail": "?}' in lines[i]:
        lines[i] = lines[i].replace('{"detail": "?}', '{"detail": "unknown"}')
        fixes += 1
    
    if 'result["trend_status"] = "?' in lines[i]:
        lines[i] = lines[i].replace('result["trend_status"] = "?', 'result["trend_status"] = "unknown"')
        fixes += 1

print(f'String fixes: {fixes}')

# Second pass: strip corruption, replace docs with English
current_fn = ''
DOCS = {
    "detect_platform_breakout": '    """Detect platform/consolidation breakout pattern."""',
    "detect_n_shape":           '    """Detect N-shape (cup and handle) pattern."""',
    "detect_vcp":               '    """Detect Volatility Contraction Pattern (VCP)."""',
    "detect_v_reversal":        '    """Detect V-shaped reversal pattern."""',
    "detect_w_bottom":          '    """Detect W-bottom (double bottom) pattern."""',
    "measure_acceleration":     '    """Measure price acceleration using recent returns."""',
    "score_volume_structure":   '    """Score volume structure for signal quality."""',
    "detect_fake_signal":       '    """Detect potential fake signals to filter out."""',
    "score_sector_context":     '    """Score sector/market context for signal amplification."""',
    "analyze_stock":            '    """Comprehensive stock analysis: detect patterns, score, classify."""',
    "classify_signal":          '    """Classify signal: STRONG / WEAK / NONE."""',
    "load_params":              '    """Load parameters from file, fall back to defaults."""',
    "save_params":              '    """Save current parameters to file."""',
}

COMMENTS = {
    "detect_platform_breakout": '# Platform breakout detection',
    "detect_n_shape": '# N-shape detection',
    "detect_vcp": '# VCP pattern',
    "detect_v_reversal": '# V-reversal',
    "detect_w_bottom": '# W-bottom',
    "measure_acceleration": '# Acceleration',
    "score_volume_structure": '# Volume scoring',
    "detect_fake_signal": '# Fake signal filter',
    "score_sector_context": '# Sector context',
    "analyze_stock": '# Analysis entry',
    "classify_signal": '# Classification',
    "load_params": '# Load params',
    "save_params": '# Save params',
}

result = []
in_docstring = False

for line in lines:
    m = re.search(r'def (\w+)', line)
    if m:
        current_fn = m.group(1)
    
    try:
        line.encode('ascii')
        is_ascii = True
    except:
        is_ascii = False
    
    if is_ascii:
        # Track docstring
        has_triple = '"""' in line
        if has_triple:
            qcount = line.count('"""')
            if qcount % 2 == 1:
                in_docstring = not in_docstring
            elif qcount >= 2:
                pass
        result.append(line)
        continue
    
    stripped = line.strip()
    
    # Triple-quote marker
    if '"""' in stripped:
        if not in_docstring:
            if current_fn in DOCS:
                result.append(DOCS[current_fn])
            else:
                result.append('    """Docstring."""')
            in_docstring = True
        else:
            result.append('    """')
            in_docstring = False
        continue
    
    # Inside docstring - skip content
    if in_docstring:
        continue
    
    # Skip pure corruption lines
    ascii_part = ''.join(c for c in line if ord(c) < 128)
    if not ascii_part.strip():
        continue
    
    # Corrupted hash comment
    if stripped.startswith('#'):
        if current_fn in COMMENTS:
            result.append('    ' + COMMENTS[current_fn])
        else:
            result.append('    # Config')
        continue
    
    # Other non-ASCII lines: keep only ASCII chars
    result.append(ascii_part.rstrip())

output = '\n'.join(result)
output = re.sub(r'\n{3,}', '\n\n', output)

with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
    f.write(output)

out_lines = output.count('\n') + 1
print(f'Wrote {out_lines} lines')

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
        lst = output.split('\n')
        for j in range(max(0,ln-3), min(len(lst), ln+4)):
            print(f'  L{j+1}: {lst[j][:120]}')
