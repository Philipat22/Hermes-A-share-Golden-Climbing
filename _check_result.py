import json, os, glob

archives = glob.glob('quant_archive/2026-05/backtest_*.json')
if not archives:
    print("No results found")
    exit()

latest = max(archives, key=os.path.getmtime)
print(f"Latest: {latest}")
d = json.load(open(latest, encoding='utf-8'))
pp = d.get("metrics", {})
print(f"Runtime: {d.get('runtime_s')}s, Picks: {d.get('total_picks')}")
for k in ["5d", "10d", "20d", "60d"]:
    v = pp.get(k, {})
    if v.get("n", 0) > 0:
        print(f"  {k}: wr={v['win_rate']*100:.0f}% ret={v['avg_return']:+.2f}% excess={v['excess']:+.2f}% n={v['n']}")
    else:
        print(f"  {k}: no picks")
