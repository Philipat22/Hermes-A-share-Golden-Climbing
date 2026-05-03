import ast

code = """def detect_platform_breakout(
    df: pd.DataFrame,
    params = None,
) -> dict:
    \"\"\"Detect platform/consolidation breakout pattern.\"\"\"
    if params is None:
        pass
"""
try:
    ast.parse(code)
    print('Isolated function OK')
except SyntaxError as e:
    print(f'Error: {e}')
