"""Clean corrupted Chinese text from engine.py, replace with English."""
import re
import sys

with open('src/surge/engine.py.bak', 'rb') as f:
    raw = f.read()

# Remove UTF-8 BOM if present
if raw[:3] == b'\xef\xbb\xbf':
    raw = raw[3:]

# Decode, replacing problematic characters
text = raw.decode('utf-8', errors='replace')

# Strategy: Replace known corrupted comment patterns with clean English equivalents
# based on function names and context

replacements = []
lines = text.split('\n')

# Function-specific docstring replacements
func_repl = {
    'detect_platform_breakout': 'Detect platform/consolidation breakout pattern.\nConditions:\n1. Recent N days consolidating (amplitude < max_amp)\n2. Price near MA20\n3. Price breaks consolidation upper bound\n4. Volume confirmation',
    'detect_n_shape': 'Detect N-shape (cup and handle) pattern.\nCriteria:\n1. First leg up (min return)\n2. Pullback (max retrace)\n3. Second leg up with volume',
    'detect_vcp': 'Detect Volatility Contraction Pattern (VCP).\nCriteria:\n1. Short-term volatility compressed relative to long-term\n2. Declining daily ranges\n3. Tight closing ranges',
    'detect_v_reversal': 'Detect V-shaped reversal pattern.\nConditions:\n1. Sharp decline\n2. Sudden reversal upward\n3. Volume surge on reversal day',
    'detect_w_bottom': 'Detect W-bottom (double bottom) pattern.\nConditions:\n1. First low\n2. Bounce\n3. Second low (similar level)\n4. Breakout above middle peak',
    'measure_acceleration': 'Measure price acceleration using recent returns.',
    'score_volume_structure': 'Score volume structure for signal quality.\nEvaluates:\n1. Volume ratio vs average\n2. Volume trend consistency\n3. Abnormal volume detection',
    'detect_fake_signal': 'Detect potential fake signals to filter out.\nFilters:\n1. Low-volume breakouts\n2. Choppy price action\n3. Reversal after signal\n4. Gap-based signals',
    'score_sector_context': 'Score sector/market context for signal amplification.\nPart 1: Sector index performance (0-50)\nPart 2: Sector signal density (0-50)\nO(1) via sector_cache',
    'analyze_stock': 'Comprehensive stock analysis: detect patterns, score, classify signal.',
    'classify_signal': 'Classify signal into STRONG / WEAK / NONE based on final score.',
    'load_params': 'Load parameters from params.json, falling back to defaults.',
}

# Build clean file
clean_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Check if this line starts a docstring (""" or ''')
    if stripped in ('"""', "'''") or stripped.startswith('"""') or stripped.startswith("'''"):
        # Look at previous non-blank, non-comment line for function name
        prev_line = ''
        for j in range(i-1, -1, -1):
            pl = lines[j].strip()
            if pl and not pl.startswith('#'):
                prev_line = pl
                break
        
        # Determine function name from previous line
        fn_name = ''
        m = re.search(r'def (\w+)', prev_line)
        if m:
            fn_name = m.group(1)
        
        # Find end of docstring
        doc_start = i
        doc_lines = []
        if stripped in ('"""', "'''"):
            # Simple opening
            doc_lines.append(line)
            # Find closing
            i += 1
            while i < len(lines):
                s = lines[i].strip()
                if s in ('"""', "'''"):
                    doc_lines.append(lines[i])
                    break
                doc_lines.append(lines[i])
                i += 1
        else:
            # Inline opening (e.g., """text...)
            # Check if it also closes on same line
            if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                doc_lines.append(line)
            else:
                doc_lines.append(line)
                i += 1
                while i < len(lines):
                    s = lines[i].strip()
                    if s in ('"""', "'''") or s.startswith('"""') or s.startswith("'''"):
                        doc_lines.append(lines[i])
                        break
                    doc_lines.append(lines[i])
                    i += 1
        
        # Replace docstring with clean version
        if fn_name in func_repl:
            clean_doc = f'    """{func_repl[fn_name]}\n    """'
            clean_lines.append(clean_doc)
        else:
            # Keep structure but remove corrupted Chinese
            clean_lines.append(line)  # Keep opening line
            # Skip the corrupted content
            # Keep the closing line
            for dl in doc_lines[1:]:
                s = dl.strip()
                if s in ('"""', "'''") or s.startswith('"""') or s.startswith("'''"):
                    if len(doc_lines) == 2:
                        clean_lines[-1] = clean_lines[-1].rstrip() + ' ' + dl.strip()
                    else:
                        clean_lines.append(dl)
                    break
    else:
        # Handle commented lines - replace corrupted # comments
        if '#' in line:
            hash_pos = line.index('#')
            before_hash = line[:hash_pos]
            comment_text = line[hash_pos+1:]
            
            # Check if comment has non-ASCII corruption
            encoded_before = line.encode('ascii', errors='replace')
            has_non_ascii = b'?' in encoded_before
            
            if has_non_ascii and b'#' in encoded_before:
                # Remove the corrupted comment part
                # Check if this is a section header
                if fn_name:
                    clean_lines.append(before_hash.rstrip())
                else:
                    clean_lines.append(before_hash.rstrip())
            else:
                clean_lines.append(line)
        else:
            clean_lines.append(line)
    
    i += 1

# Write clean file
output = '\n'.join(clean_lines)

with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
    f.write(output)

print(f'Wrote {len(clean_lines)} lines to engine.py')

# Verify syntax
import py_compile
try:
    py_compile.compile('src/surge/engine.py', doraise=True)
    print('SYNTAX OK')
except py_compile.PyCompileError as e:
    print(f'Syntax error: {e}')
