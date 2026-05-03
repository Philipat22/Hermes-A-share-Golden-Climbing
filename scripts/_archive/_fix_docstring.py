"""Fix engine.py: proper docstring balancing + f-string brace fix."""
import re
import sys

path = 'src/surge/engine.py.bak'
with open(path, 'rb') as f:
    raw = f.read()
if raw[:3] == b'\xef\xbb\xbf':
    raw = raw[3:]
s = raw.decode('utf-8', errors='replace')

# Step 1: Fix f-string brace issues
for old, new in [
    ('max_amp*100:.0f}%', '{max_amp*100:.0f}%'),
    ('first_return*100:.1f}%', '{first_return*100:.1f}%'),
    ('short_vol*100:.2f}%', '{short_vol*100:.2f}%'),
    ('acceleration*100:.1f}%', '{acceleration*100:.1f}%'),
    ('recent_turnover:.0f}%(-30)', '{recent_turnover:.0f}%(-30)'),
    ('60ret_60d*100:.0f}%(-25)', '{60ret_60d*100:.0f}%(-25)'),
]:
    s = s.replace(old, new)

# Fix unterminated details
s = re.sub(r'result\["detail"\]\s*=\s*"\s*\?', 'result["detail"] = ""', s)
s = re.sub(r'result\["trend_status"\]\s*=\s*"\?', 'result["trend_status"] = "unknown"', s)
for old, new in [('"V?,', '"V-reversal",'), ('"W?,', '"W-bottom",'), ('"N?,', '"N-shape",')]:
    s = s.replace(old, new)

print('Step 1: f-string/string fixes applied')

# Step 2: Replace docstrings by function name (rebuild entire docstring tracking)
lines = s.split('\n')

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
    "classify_signal":          '    """Classify signal: STRONG / WEAK / NONE / FAKE."""',
    "load_params":              '    """Load parameters from file, fall back to defaults."""',
    "save_params":              '    """Save current parameters to file."""',
    "_build_sector_cache":      '    """Build sector signal density cache from all_signals."""',
}

# Build: collect all line indices that are """ markers
triple_lines = []
for i, line in enumerate(lines):
    if '"""' in line:
        triple_lines.append(i)

# Walk through, replacing docstrings
result = []
i = 0
current_fn = ''

while i < len(lines):
    line = lines[i]
    
    # Track function
    m = re.search(r'def (\w+)', line)
    if m:
        current_fn = m.group(1)
    
    # Check for triple quote
    if '"""' in line and i in triple_lines:
        tq_count = line.count('"""')
        
        if tq_count >= 2:
            # Self-contained docstring: """text"""
            # Replace entirely
            if current_fn in DOCS:
                result.append(DOCS[current_fn])
            else:
                # Keep structure but remove corrupted content
                result.append('    """"""')
            i += 1
            continue
        
        # Single triple-quote opener or closer
        # Check if next function has a docstring defined
        # Find the function name for the docstring starting here
        
        # Look backward for the most recent def
        for j in range(i-1, -1, -1):
            fm = re.search(r'def (\w+)', lines[j])
            if fm:
                current_fn = fm.group(1)
                break
        
        # Look for the closing """
        close_found = False
        for j in range(i+1, min(i+50, len(lines))):
            if '"""' in lines[j]:
                # Found closing - replace entire range with English docstring
                if current_fn in DOCS:
                    result.append(DOCS[current_fn])
                else:
                    result.append('    """"""')
                i = j + 1
                close_found = True
                break
        
        if close_found:
            continue
        
        # No close found - this is probably a corrupted line
        # Remove the corrupted """ line
        i += 1
        continue
    
    # Normal line - check for non-ASCII corruption
    try:
        line.encode('ascii')
        result.append(line)
    except UnicodeEncodeError:
        # Strip corruption
        ascii_part = ''.join(c for c in line if ord(c) < 128 and c != '\ufffd')
        if ascii_part.strip():
            result.append(ascii_part)
        # else: pure corruption, skip
    
    i += 1

output = '\n'.join(result)
output = re.sub(r'\n{3,}', '\n\n', output)

# Step 3: Also handle corrupted comments

# Write
with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
    f.write(output)

rlines = len(output.split('\n'))
print(f'Wrote {rlines} lines')

# Verify
import py_compile
try:
    py_compile.compile('src/surge/engine.py', doraise=True)
    print('*** SYNTAX OK ***')
    sys.exit(0)
except py_compile.PyCompileError as e:
    print(f'SYNTAX ERROR: {e}')
    em = re.search(r'line (\d+)', str(e))
    if em:
        ln = int(em.group(1))
        lst = output.split('\n')
        for j in range(max(0,ln-3), min(len(lst), ln+4)):
            print(f'  L{j+1}: {lst[j][:120]}')
    sys.exit(1)
