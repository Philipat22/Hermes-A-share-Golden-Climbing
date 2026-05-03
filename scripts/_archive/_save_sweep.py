
import json, os
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
path = os.path.join(ROOT, 'data', 'models', 'phase1d_label_sweep.json')
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('Loaded saved JSON')
    for r in data['rankings'][:10]:
        print(f"  {r['rank']}. {r['config']}: AUC={r['avg_auc']} Excess={r['avg_excess']}% Picks={r['total_picks']} Score={r['score']}")
else:
    print('JSON not saved. Re-saving...')
    rows = [
        ('5d_3%', 0.6000, 12.09, 1755),
        ('5d_5%', 0.6310, 22.06, 408),
        ('5d_7%', 0.6576, 25.89, 62),
        ('5d_10%', 0.6861, 26.97, 130),
        ('10d_5%', 0.5934, 11.03, 1620),
        ('10d_7%', 0.6104, 14.31, 1223),
        ('10d_10%', 0.6319, 21.36, 518),
        ('10d_15%', 0.6541, 42.52, 199),
        ('20d_7%', 0.5696, 12.05, 1405),
        ('20d_10%', 0.5820, 12.28, 2083),
        ('20d_15%', 0.5972, 17.22, 777),
    ]
    # Only include rows with reasonable picks (<5000)
    valid = [(n,a,e,p) for n,a,e,p in rows if p < 5000]
    valid.sort(key=lambda r: -r[1])
    for i, (n, a, e, p) in enumerate(valid, 1):
        print(f"  {i:2d}. {n:<10} AUC={a:.4f} Excess={e:+.2f}%  Picks={p}")
    
    # Save with proper np conversion
    output = {'rankings': [{'rank': i, 'config': n, 'avg_auc': float(a), 'avg_excess': float(e), 'total_picks': int(p)} for i,(n,a,e,p) in enumerate(valid, 1)]}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")
