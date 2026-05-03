#!/usr/bin/env python3
"""Analyze surge scan results"""
import sys, json
sys.path.insert(0, r'D:\AIHedgeFund\ai-hedge-fund-main')

with open(r'D:\AIHedgeFund\ai-hedge-fund-main\quant_archive\2026-04\surge_scan_20260430_1132.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
sigs = data['signals']

print("=== Top 20 by score ===")
for s in sorted(sigs, key=lambda x: x['final_score'], reverse=True)[:20]:
    print(f"  {s['ts_code']:>10} {s['name']:<8} score={s['final_score']:3d} grade={s['signal_grade']:>6} type={s.get('pattern_type',''):<8}")

print()
print("=== STRONG signals ===")
for s in sorted([x for x in sigs if x['signal_grade']=='STRONG'], key=lambda x: x['final_score'], reverse=True):
    print(f"  {s['ts_code']:>10} {s['name']:<8} score={s['final_score']:3d} type={s.get('pattern_type','')}")

print()
print("=== Sector distribution ===")
sectors = {}
for s in sigs:
    sec = s.get('industry', '?')
    if sec not in sectors:
        sectors[sec] = {'STRONG':0, 'WEAK':0, 'NONE':0, 'FAKE':0}
    g = s.get('signal_grade','NONE')
    if g in sectors[sec]:
        sectors[sec][g] += 1
    else:
        sectors[sec][g] = 1

for sec in sorted(sectors.keys()):
    d = sectors[sec]
    total = sum(d.values())
    strong = d.get('STRONG',0)
    print(f"  {sec:<12} total={total:3d} strong={strong}")
