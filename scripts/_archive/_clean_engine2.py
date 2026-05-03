"""Clean corrupted engine.py: strip corrupted Chinese, keep Python code + minimal English docs."""
import re, sys, py_compile

def main():
    with open('src/surge/engine.py.bak', 'rb') as f:
        raw = f.read()
    if raw[:3] == b'\xef\xbb\xbf':
        raw = raw[3:]
    text = raw.decode('utf-8', errors='replace')
    lines = text.split('\n')
    print(f'Read {len(lines)} lines')

    # Minimal English docstrings keyed by function name
    DOCS = {
        "detect_platform_breakout": '    """Detect platform/consolidation breakout pattern."""',
        "detect_n_shape":             '    """Detect N-shape (cup and handle) pattern."""',
        "detect_vcp":                 '    """Detect Volatility Contraction Pattern."""',
        "detect_v_reversal":          '    """Detect V-shaped reversal pattern."""',
        "detect_w_bottom":            '    """Detect W-bottom (double bottom) pattern."""',
        "measure_acceleration":       '    """Measure price acceleration."""',
        "score_volume_structure":     '    """Score volume structure for signal quality."""',
        "detect_fake_signal":         '    """Detect potential fake signals to filter out."""',
        "score_sector_context":       '    """Score sector/market context for signal amplification."""',
        "analyze_stock":              '    """Comprehensive stock analysis: detect patterns, score, classify signal."""',
        "classify_signal":            '    """Classify signal into STRONG / WEAK / NONE based on final score."""',
        "load_params":                '    """Load parameters from params.json, falling back to defaults."""',
        "save_params":                '    """Save current parameters to params.json."""',
    }

    result = []
    in_docstring = False
    current_fn = ''

    for line in lines:
        # Track current function
        m = re.search(r'def (\w+)', line)
        if m:
            current_fn = m.group(1)

        # Check if line is pure ASCII (safe Python code)
        try:
            line.encode('ascii')
            is_ascii = True
        except UnicodeEncodeError:
            is_ascii = False

        if is_ascii:
            result.append(line)
            if '"""' in line:
                in_docstring = not in_docstring
            continue

        stripped = line.strip()

        # Handle docstring opening/closing markers
        if '"""' in stripped:
            if not in_docstring:
                # Opening - replace with English
                if current_fn in DOCS:
                    result.append(DOCS[current_fn])
                else:
                    result.append('    """Docstring."""')
                in_docstring = True
            else:
                # Closing
                result.append('    """')
                in_docstring = False
            continue

        if in_docstring:
            # Skip corrupted docstring content lines
            continue

        # Extract ASCII parts from mixed line
        ascii_part = ''.join(c for c in line if ord(c) < 128)
        if ascii_part.strip():
            result.append(ascii_part.rstrip())
        # else: pure corruption line, drop

    output = '\n'.join(result)
    # Collapse excessive blank lines
    output = re.sub(r'\n{3,}', '\n\n', output)

    with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
        f.write(output)

    out_lines = output.split('\n')
    print(f'Wrote {len(out_lines)} lines')

    # Verify
    try:
        py_compile.compile('src/surge/engine.py', doraise=True)
        print('SYNTAX OK')
        return True
    except py_compile.PyCompileError as e:
        print(f'SYNTAX ERROR: {e}')
        # Show context
        em = re.search(r'line (\d+)', str(e))
        if em:
            ln = int(em.group(1))
            for j in range(max(0,ln-4), min(len(out_lines), ln+3)):
                print(f'  L{j+1}: {out_lines[j][:130]}')
        return False

if __name__ == '__main__':
    sys.exit(0 if main() else 1)
