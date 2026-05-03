"""Fix engine.py corruption: braces, unterminated strings, docstrings."""
import re
import sys

# Step 1: Copy backup
with open('src/surge/engine.py.bak', 'rb') as f:
    raw = f.read()
if raw[:3] == b'\xef\xbb\xbf':
    raw = raw[3:]
text = raw.decode('utf-8', errors='replace')
lines = text.split('\n')

print(f'Read {len(lines)} lines from backup')

# Step 2: Fix specific f-string brace issues
fixes_applied = 0

for i in range(len(lines)):
    line = lines[i]
    
    # Fix 1: Missing { in "max_amp*100:.0f}%"
    if 'max_amp*100:.0f}%' in line:
        lines[i] = line.replace('max_amp*100:.0f}%', '{max_amp*100:.0f}%')
        fixes_applied += 1
    
    # Fix 2: Missing { in "first_return*100:.1f}%"
    if 'first_return*100:.1f}%' in line:
        lines[i] = line.replace('first_return*100:.1f}%', '{first_return*100:.1f}%')
        fixes_applied += 1
    
    # Fix 3: Missing { in "short_vol*100:.2f}%"
    if 'short_vol*100:.2f}%' in line:
        lines[i] = line.replace('short_vol*100:.2f}%', '{short_vol*100:.2f}%')
        fixes_applied += 1
    
    # Fix 4: Missing { in "acceleration*100:.1f}%"
    if 'acceleration*100:.1f}%' in line:
        lines[i] = line.replace('acceleration*100:.1f}%', '{acceleration*100:.1f}%')
        fixes_applied += 1
    
    # Fix 5: Unterminated opening strings (line ends with " ?)
    if line.strip().endswith('" ?'):
        # These are lines like: result["detail"] = " ?
        # The Chinese text after opening quote was corrupted
        lines[i] = line.rstrip()[:-2] + '""\")' if 'result[' in line else line.rstrip()[:-2] + '.\"\"\"\")'
    
    # Fix 6: Unterminated trend_status
    if 'result["trend_status"] = "?' in line:
        lines[i] = line.replace('?' , 'unknown')
    
    # Fix 7: Missing brace in recent_turnover
    if 'flags.append(f"recent_turnover:.0f}%' in line and 'ValueError' not in lines[i]:
        lines[i] = line.replace('flags.append(f"recent_turnover:.0f}%(-30)')',
                                'flags.append(f"{recent_turnover:.0f}%(-30)")')
        lines[i] = line.replace('"recent_turnover:.0f}%(-30)",' , '"{recent_turnover:.0f}%(-30)",')
        lines[i] = line.replace('"recent_turnover:.0f}%(-30)"' , '"{recent_turnover:.0f}%(-30)"')
    if 'recent_turnover:.0f}%(-30)' in line:
        lines[i] = line.replace('recent_turnover:.0f}%(-30)', '{recent_turnover:.0f}%(-30)')
        fixes_applied += 1
    
    # Fix 8: Missing brace in 60ret_60d
    if '60ret_60d*100:.0f}%(-25)' in line:
        lines[i] = line.replace('60ret_60d*100:.0f}%(-25)', '{60ret_60d*100:.0f}%(-25)')
        fixes_applied += 1
    
    # Fix 9: Unterminated single quotes for pattern labels
    if '"V?,' in line:
        lines[i] = line.replace('"V?,', '"V-reversal",')
        fixes_applied += 1
    if '"W?,' in line:
        lines[i] = line.replace('"W?,', '"W-bottom",')
        fixes_applied += 1
    if '"N?,' in line:
        lines[i] = line.replace('"N?,', '"N-shape",')
        fixes_applied += 1

# Step 3: Fix more complex patterns
# Replace corrupted docstrings with simple English
result = []
in_docstring = False
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
    "load_params":              '    """Load parameters from params.json, fall back to defaults."""',
    "save_params":              '    """Save current parameters to params.json."""',
}

# Additional fix: ensure all result["detail"] assignments have proper closures
# Lines with unterminated detail strings
for i in range(len(lines)):
    line = lines[i]
    stripped = line.strip()
    
    # Fix unterminated result["detail"] assignments
    if 'result["detail"]' in line and '"' in line:
        # Check if the string is properly terminated
        # Find the last quote
        last_quote = line.rfind('"')
        first_quote = line.find('"', line.index('result'))
        if first_quote > 0 and last_quote > first_quote:
            pass  # Has proper quotes
        else:
            # Unterminated
            lines[i] = line.rstrip() + '"'

# Fix the dict literal at line 771 (extra })
for i in range(len(lines)):
    if '"sector_index_perf": "N/A", "sector_index_code": ""}' in lines[i]:
        lines[i] = lines[i].replace('"sector_index_perf": "N/A", "sector_index_code": ""}', 
                                     '"sector_index_perf": "", "sector_index_code": ""')
        fixes_applied += 1

print(f'Specific fixes applied: {fixes_applied}')

# Step 4: Rebuild with clean docstrings, strip corrupted comments, handle encoding
clean_lines = []
current_fn = ''

for line in lines:
    # Track function
    m = re.search(r'def (\w+)', line)
    if m:
        current_fn = m.group(1)
    
    # Check if pure ASCII
    try:
        line.encode('ascii')
        is_ascii = True
    except:
        is_ascii = False
    
    stripped = line.strip()
    
    if is_ascii:
        clean_lines.append(line)
        continue
    
    # Handle docstrings
    triple_quote_count = stripped.count('"""')
    if triple_quote_count > 0:
        if current_fn in DOCS:
            clean_lines.append(DOCS[current_fn])
            if triple_quote_count == 1:
                in_docstring = not in_docstring
            continue
        else:
            clean_lines.append(line)
            continue
    
    # Skip corrupted docstring content
    if in_docstring:
        continue
    
    # Skip pure-corruption lines
    ascii_part = ''.join(c for c in line if ord(c) < 128)
    if not ascii_part.strip():
        continue
    
    # Hash-comment lines with corruption
    if stripped.startswith('#'):
        # Replace with simple English
        fn_comment = {
            "detect_platform_breakout": '# Platform breakout detection',
            "detect_n_shape": '# N-shape detection',
            "detect_vcp": '# VCP pattern',
            "detect_v_reversal": '# V-reversal',
            "detect_w_bottom": '# W-bottom',
            "measure_acceleration": '# Acceleration',
            "score_volume_structure": '# Volume scoring',
            "detect_fake_signal": '# Fake signal filter',
            "score_sector_context": '# Sector context',
            "analyze_stock": '# Analysis',
            "classify_signal": '# Classification',
            "load_params": '# Load params',
            "save_params": '# Save params',
        }
        if current_fn in fn_comment:
            clean_lines.append('    ' + fn_comment[current_fn])
        else:
            clean_lines.append('    # Parameter config')
        continue
    
    # For other mixed lines, keep only ASCII
    if ascii_part.strip():
        clean_lines.append(ascii_part.rstrip())

output = '\n'.join(clean_lines)
output = re.sub(r'\n{3,}', '\n\n', output)

with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
    f.write(output)

out_lines = output.split('\n')
print(f'Wrote {len(out_lines)} lines')

# Verify
import py_compile
try:
    py_compile.compile('src/surge/engine.py', doraise=True)
    print('*** SYNTAX OK ***')
except py_compile.PyCompileError as e:
    print(f'SYNTAX ERROR: {e}')
    em = re.search(r'line (\d+)', str(e))
    if em:
        ln = int(em.group(1))
        for j in range(max(0,ln-3), min(len(out_lines), ln+3)):
            print(f'  L{j+1}: {out_lines[j][:120]}')
