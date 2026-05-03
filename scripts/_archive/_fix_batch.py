"""Find and fix ALL remaining f-string/corruption issues at once."""
import re

with open('src/surge/engine.py', 'r', encoding='utf-8') as f:
    s = f.read()

all_fixes = [
    # Pattern: f"{'' if X else '?}" -> f"{'' if X else ''}"
    (r"""f\"\{\'\'\s*if\s+\w+\s+else\s+'\\?"(?=[\s\}])""", lambda m: m.group(0).rstrip("'" + chr(92) + '?').rstrip("'\\?") + "''}"),
]

# Simpler approach: find all occurrences of unterminated else clauses
import re

fixes_applied = 0

# Fix: f" ... else '?}  -> else ''}
while "' else '?}" in s:
    s = s.replace("' else '?}", "' else ''}")
    fixes_applied += 1

# Fix: f" ... else '? " -> else '' "
while "' else '? " in s:
    s = s.replace("' else '? ", "' else '' ")
    fixes_applied += 1

# Fix: f" ... else '?"  -> else ''
while "' else '?" in s:
    s = s.replace("' else '?", "' else ''")
    fixes_applied += 1

# General: any '? at end of lines (trailing corrupted content)
# But careful not to break legitimate content
# Let's look for '?" or '? followed by closing f-string context
s = re.sub(r"""'\?\)""", "''\\)", s)  # '?) -> '')
s = re.sub(r"""'\?\s""", "'' ", s)    # '? followed by space -> ''
s = re.sub(r"""'\?$""", "''", s, flags=re.MULTILINE)  # '? at end of line -> ''

print(f'Applied {fixes_applied} simple replacements')

# Finally, fix: f"N??{ -> f"N {
s = re.sub(r'f"N\?\?\{', 'f"N {', s)

# Fix: f"?{ -> f"{
s = re.sub(r'f"\?\{', 'f"{', s)

# Fix: stray printed } characters 
# Check for patterns like f"...}" where } is not part of a format expression
# These are single } in f-strings

with open('src/surge/engine.py', 'w', encoding='utf-8') as f:
    f.write(s)

# Verify
import ast
try:
    ast.parse(s)
    print('*** AST OK ***')
except SyntaxError as e:
    print(f'ERROR: L{e.lineno}: {e.msg}')
    lines = s.split('\n')
    for j in range(max(0,e.lineno-2), min(len(lines), e.lineno+2)):
        print(f'  L{j+1}: {lines[j]}')
