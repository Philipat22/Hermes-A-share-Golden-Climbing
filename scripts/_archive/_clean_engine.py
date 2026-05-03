"""Clean corrupted Chinese text from engine.py, replacing with English."""

import re
import sys

def main():
    # Read the corrupted file
        import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    with open('src/surge/engine.py.bak', 'rb') as f:
        raw = f.read()
    
    # Remove BOM if present
        if raw[:3] == b'\xef\xbb\xbf':
        raw = raw[3:]
    
    text = raw.decode('utf-8', errors='replace')
    lines = text.split('\n')
    
    print(f"Read {len(lines)} lines from backup")
    
    # Function-based docstring replacements
    DOCS = {
        "detect_platform_breakout": (
            '    """Detect platform/consolidation breakout pattern.\n'
            '    Conditions:\n'
            '    1. Recent N days consolidating (amplitude < max_amp)\n'
            '    2. Price near MA20\n'
            '    3. Price breaks consolidation upper bound\n'
            '    4. Volume confirmation\n'
            '    Returns: {detected, score, detail, consolidation_low, consolidation_high, breakout_price, days, amplitude}\n'
            '    """'
        ),
        "detect_n_shape": (
            '    """Detect N-shape (cup and handle) pattern.\n'
            '    Criteria:\n'
            '    1. First leg up (min return)\n'
            '    2. Pullback (max retrace)\n'
            '    3. Second leg up with volume\n'
            '    Returns: {detected, score, detail, leg1_return, pullback, leg2_return}\n'
            '    """'
        ),
        "detect_vcp": (
            '    """Detect Volatility Contraction Pattern (VCP).\n'
            '    Criteria:\n'
            '    1. Short-term volatility compressed relative to long-term\n'
            '    2. Declining daily ranges\n'
            '    3. Tight closing ranges near highs\n'
            '    Returns: {detected, score, detail, vol_ratio, range_decline}\n'
            '    """'
        ),
        "detect_v_reversal": (
            '    """Detect V-shaped reversal pattern.\n'
            '    Conditions:\n'
            '    1. Sharp recent decline\n'
            '    2. Sudden high-volume reversal upward\n'
            '    3. Follow-through confirmation\n'
            '    Returns: {detected, score, detail, decline_pct, reversal_pct}\n'
            '    """'
        ),
        "detect_w_bottom": (
            '    """Detect W-bottom (double bottom) pattern.\n'
            '    Conditions:\n'
            '    1. First low and bounce\n'
            '    2. Second low at similar level\n'
            '    3. Breakout above middle peak\n'
            '    Returns: {detected, score, detail, bottom1, bottom2, neckline}\n'
            '    """'
        ),
        "measure_acceleration": (
            '    """Measure price acceleration using recent returns.\n'
            '    Computes momentum from multiple windows.\n'
            '    Returns: {acceleration, slope, recent_return, volatility}\n'
            '    """'
        ),
        "score_volume_structure": (
            '    """Score volume structure for signal quality.\n'
            '    Evaluates:\n'
            '    1. Volume ratio vs 20-day average\n'
            '    2. Volume trend consistency\n'
            '    3. Abnormal volume detection\n'
            '    Returns: {score, detail, vol_ratio, trend_score}\n'
            '    """'
        ),
        "detect_fake_signal": (
            '    """Detect potential fake signals to filter out.\n'
            '    Filters:\n'
            '    1. Low-volume breakouts\n'
            '    2. Choppy price action\n'
            '    3. Reversal after signal\n'
            '    4. Gap-based signals\n'
            '    Returns: {is_fake, reason, confidence}\n'
            '    """'
        ),
        "score_sector_context": (
            '    """Score sector/market context for signal amplification.\n'
            '    Part 1: Sector index performance (0-50)\n'
            '    Part 2: Sector signal density (0-50)\n'
            '    O(1) via sector_cache\n'
            '    Returns: {score, detail, index_score, density_score}\n'
            '    """'
        ),
        "analyze_stock": (
            '    """Comprehensive stock analysis: detect patterns, score, classify signal.\n'
            '    Runs all pattern detectors, volume scoring, acceleration,\n'
            '    fake signal filtering, and sector context scoring.\n'
            '    Returns complete signal dictionary.\n'
            '    """'
        ),
        "classify_signal": (
            '    """Classify signal into STRONG / WEAK / NONE based on final score.\n'
            '    Thresholds from params (weak_signal, strong_signal).\n'
            '    """'
        ),
        "load_params": (
            '    """Load parameters from params.json, falling back to defaults."""'
        ),
    }
    
    # Comment line replacements per function
    COMMENTS = {
        "detect_platform_breakout": '# Platform breakout detection',
        "detect_n_shape": '# N-shape detection',
        "detect_vcp": '# VCP pattern detection',
        "detect_v_reversal": '# V-reversal detection',
        "detect_w_bottom": '# W-bottom detection',
        "measure_acceleration": '# Acceleration measurement',
        "score_volume_structure": '# Volume structure scoring',
        "detect_fake_signal": '# Fake signal filtering',
        "score_sector_context": '# Sector context scoring',
        "analyze_stock": '# Stock analysis engine',
        "classify_signal": '# Signal classification',
        "load_params": '# Parameter loading',
        "save_params": '# Parameter saving',
        "": '# Default comment',
    }
    
    # State machine
    result = []
    in_docstring = False
    docstring_lineno = -1
    current_fn = ''
    
    for i, line in enumerate(lines):
        # Detect function name from preceding def line
        m = re.search(r'def (\w+)', line)
        if m:
            current_fn = m.group(1)
        
        # Check for Python string delimiters
        # Count occurrences of triple-quote
        tq_in_line = line.count('"""')
        
        if in_docstring:
            # We're inside a multi-line docstring
            if tq_in_line >= 1:
                # This line closes the docstring
                result.append(line)
                in_docstring = False
                # Now insert replacement docstring content
                if current_fn and current_fn in DOCS:
                    pass  # already handled
                continue
            else:
                # Inside docstring content - skip it
                continue
        
        # Check if this line OPENS a docstring
        if tq_in_line == 1 and '"""' in line:
            # Check if this is a standalone """ opener (not inline comment)
            stripped = line.strip()
            if stripped == '"""':
                # Simple opener - replace docstring
                if current_fn and current_fn in DOCS:
                    # Insert the replacement docstring
                    result.append(DOCS[current_fn])
                    in_docstring = True
                    # The opening """ was processed; 
                    # the closing """ will be in the replacement
                    continue
                else:
                    result.append(line)
                    in_docstring = True
                    continue
        
        # Check if it's a standalone docstring line (both open and close)
        if tq_in_line >= 2 and not in_docstring:
            # Inline docstring like """text"""
            # Strip the content if it's corrupted
            if current_fn and current_fn in DOCS:
                result.append(DOCS[current_fn])
                continue
            else:
                # Keep as-is if it's clean (ASCII only)
                try:
                    line.encode('ascii')
                    result.append(line)
                except:
                    # Corrupted inline docstring - strip content
                    # Keep only the """ markers
                    result.append('    """"""')
                continue
        
        # Check if it's a # comment line with non-ASCII
        try:
            line.encode('ascii')
            is_ascii = True
        except:
            is_ascii = False
        
        if not is_ascii:
            # Check if this is a comment-only line
            stripped = line.strip()
            if stripped.startswith('#'):
                # Replace with generic English comment
                comment_fn = current_fn if current_fn else 'default'
                if comment_fn in COMMENTS:
                    result.append('    ' + COMMENTS[comment_fn])
                else:
                    result.append('    # Parameter/config')
                continue
            
            # Check if it's a standalone non-ASCII content line (not code)
            # If it has only non-ASCII chars and whitespace, skip it
            ascii_chars = [c for c in stripped if ord(c) < 128]
            if len(ascii_chars) <= 2:
                continue  # Skip pure corruption lines
            
            # Try to keep the ASCII part
            ascii_line = ''.join(c for c in line if ord(c) < 128).rstrip()
            if ascii_line.strip():
                result.append(ascii_line)
            continue
        
        # Clean ASCII line - keep as-is
        result.append(line)
    
    output = '\n'.join(result)
    
    # Fix: if docstring was never closed, add closing
    if in_docstring:
        output += '\n    """'
    
    # Write output
    with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"\nWrote {len(result)} lines to engine.py")
    
    # Verify syntax
    import py_compile
    try:
        py_compile.compile('src/surge/engine.py', doraise=True)
        print('SYNTAX OK')
        return True
    except py_compile.PyCompileError as e:
        print(f'\nSyntax error: {e}')
        
        # Show the problematic area
        error_line = None
        m2 = re.search(r'line (\d+)', str(e))
        if m2:
            error_line = int(m2.group(1))
            print(f'\nContext around line {error_line}:')
            start = max(0, error_line - 3)
            end = min(len(result), error_line + 3)
            for j in range(start, end):
                marker = '>>>' if j == error_line - 1 else '   '
                print(f'{marker} L{j+1}: {result[j][:120]}')
        
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
