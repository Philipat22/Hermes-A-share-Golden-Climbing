#!/usr/bin/env python3
"""Scan all Python files for missing functions"""
import os, ast

PROJECT = r'D:\AIHedgeFund\ai-hedge-fund-main'
missing_funcs = ['get_pro_api','get_company_news','get_insider_trades','get_limit_list','get_north_money','get_market_context','get_stock_name_for_ticker','_simple_sentiment']

# Find all files containing each missing function
results = {fn: [] for fn in missing_funcs}
for root, dirs, files in os.walk(PROJECT):
    # Skip __pycache__ and venv
    dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'venv', 'env', '.venv')]
    for fname in files:
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(root, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                src = f.read()
            tree = ast.parse(src)
            for fn_node in tree.body:
                if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if fn_node.name in missing_funcs:
                        rel = os.path.relpath(fpath, PROJECT)
                        results[fn_node.name].append(rel)
        except:
            pass

print('Missing function locations:')
for fn, paths in results.items():
    if paths:
        for p in paths:
            print(f'  {fn}: {p}')
    else:
        print(f'  {fn}: NOT FOUND IN ANY FILE')
