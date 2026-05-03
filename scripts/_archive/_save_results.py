import json, os, numpy as np
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

data = {
    '5d_10%': {
        'avg_auc': 0.6866,
        'avg_excess': 25.70,
        'total_picks': 211,
        'n_features': 50,
        'pos_rate': 0.06,
        'features': ['klen','std_5','std_30','std_10','rsqr_60','alpha54','std_20','std_60','alpha52','alpha6','alpha19','rsqr_30','alpha1','rsqr_20','alpha83','alpha11','alpha36','alpha28','alpha14','beta_60','alpha26','alpha16','alpha10','alpha27','alpha24','alpha33','alpha41','alpha13','alpha2','alpha30','alpha31','alpha79','alpha17','rsqr_10','alpha49','alpha5','alpha7','alpha44','alpha77','alpha70','alpha69','alpha22','alpha29','beta_30','alpha43','alpha58','alpha9','alpha35','alpha25','alpha20'],
        'walk_forward': [
            {'window': '2022 Bear', 'auc': 0.6853, 'excess': 18.87, 'picks': 88},
            {'window': '2023 Sideways', 'auc': 0.6734, 'excess': 37.30, 'picks': 52},
            {'window': '2024 Recovery', 'auc': 0.7010, 'excess': 20.93, 'picks': 71},
        ]
    },
    '10d_15%': {
        'avg_auc': 0.6568,
        'avg_excess': 54.14,
        'total_picks': 269,
        'n_features': 40,
        'pos_rate': 0.06,
        'features': ['klen','std_10','std_5','rsqr_60','std_30','std_20','std_60','rsqr_20','alpha19','rsqr_30','beta_60','rsqr_10','beta_30','alpha83','alpha1','alpha52','alpha11','alpha54','alpha14','alpha28','alpha68','alpha10','alpha6','alpha26','alpha27','alpha24','alpha31','alpha36','alpha33','alpha16','alpha13','alpha30','alpha2','alpha41','alpha70','alpha77','alpha17','alpha5','alpha29','alpha79'],
        'walk_forward': [
            {'window': '2022 Bear', 'auc': 0.6408, 'excess': 49.96, 'picks': 22},
            {'window': '2023 Sideways', 'auc': 0.6670, 'excess': 109.70, 'picks': 10},
            {'window': '2024 Recovery', 'auc': 0.6626, 'excess': 2.76, 'picks': 237},
        ]
    }
}

path = os.path.join(ROOT, 'data', 'models', 'phase1e_dual_label_results.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, cls=NpEncoder, ensure_ascii=False)
print(f'Saved: {path}')
# Now write a clean markdown summary
summary_path = os.path.join(ROOT, 'quant_archive', '2026-05', 'label_optimization_results.md')
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("# Phase 1e: Dual Label Optimization Results\n\n")
    f.write(f"Date: 2026-05-01 22:40\n\n")
    for label, d in data.items():
        f.write(f"## {label}\n\n")
        f.write(f"- AUC: {d['avg_auc']:.4f}\n")
        f.write(f"- Avg Excess: {d['avg_excess']:+.2f}%\n")
        f.write(f"- Total Picks (3 windows): {d['total_picks']}\n")
        f.write(f"- Best Feature Count: {d['n_features']}\n")
        f.write(f"- Positive Rate: {d['pos_rate']:.1%}\n\n")
        for wf in d['walk_forward']:
            f.write(f"  - {wf['window']}: AUC={wf['auc']:.4f}, Excess={wf['excess']:+.2f}%, Picks={wf['picks']}\n")
        f.write(f"\n### Top Features\n\n")
        for i, feat in enumerate(d['features'][:10], 1):
            f.write(f"  {i}. {feat}\n")
        f.write("\n---\n\n")
    f.write("Models saved as:\n")
    f.write("- data/models/surge_5d_10%.pkl\n")
    f.write("- data/models/surge_10d_15%.pkl\n")
print(f'Saved: {summary_path}')
