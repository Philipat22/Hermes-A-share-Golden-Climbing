"""
Phase 1c: A-Share Specific Feature Integration
==============================================
对比: 原始160因子(基线) vs 原始160因子 + 18个A股特色因子

1. 加载因子缓存
2. 加载CN特色因子 (北向资金/融资融券/股东户数)
3. 合并 → 特征集变大
4. 全数据重训 + Auto Feature Selection
5. Walk-Forward对比: Baseline vs CN+Original

用法: python scripts/phase1c_cn_features.py
"""
import sys, os, pickle, time, json, gc, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

print('=' * 65)
print('Phase 1c: A-Share Specific Feature Integration')
print('=' * 65)

t_start = time.time()

# ── 1. Load factor cache ──────────────────────────────────────────────────
print(f'\n[1/6] Loading factor cache...')
FACTOR_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
all_dfs = []
for fn in sorted(os.listdir(FACTOR_DIR)):
    if fn.endswith('.parquet'):
        all_dfs.append(pd.read_parquet(os.path.join(FACTOR_DIR, fn)))
pdf = pd.concat(all_dfs, ignore_index=True)
print(f'  Rows: {len(pdf):,}  Cols: {len(pdf.columns)}')

# ── 2. Create labels ──────────────────────────────────────────────────────
print(f'\n[2/6] Creating labels...')
HORIZON = 20
pdf = pdf.sort_values(['vt_symbol', 'datetime']).reset_index(drop=True)
pdf['forward_ret'] = np.nan
for name, group in pdf.groupby('vt_symbol', sort=False):
    idx = group.index
    closes = group['close'].values
    for i in range(len(group) - HORIZON):
        pdf.loc[idx[i], 'forward_ret'] = (closes[i+HORIZON] - closes[i]) / closes[i]
pdf['label'] = (pdf['forward_ret'] >= 0.10).astype(int)
pdf['date'] = pd.to_datetime(pdf['datetime'])
pdf['year'] = pdf['date'].dt.year
print(f'  Surge rate: {pdf["label"].mean():.2%}')
print(f'  Date range: {pdf["date"].min().date()} ~ {pdf["date"].max().date()}')

# ── 3. Load CN features ──────────────────────────────────────────────────
print(f'\n[3/6] Loading CN features...')
CN_CACHE = os.path.join(ROOT, 'data', 'cache', 'cn_features', 'cn_features_all.parquet')
if not os.path.exists(CN_CACHE):
    print('  CN features not cached. Fetching...')
    from src.surge.cn_features import fetch_all_cn_features
    stock_codes = pdf['vt_symbol'].unique().tolist()
    print(f'  Stock codes: {len(stock_codes)}')
    cn_df = fetch_all_cn_features(stock_codes)
else:
    cn_df = pd.read_parquet(CN_CACHE)
    print(f'  CN features: {len(cn_df):,} rows, {len(cn_df.columns)-2} features (cached)')

cn_features = [c for c in cn_df.columns if c not in ['vt_symbol', 'date']]
print(f'  CN feature names: {cn_features}')

# ── 4. Merge ──────────────────────────────────────────────────────────────
print(f'\n[4/6] Merging CN features with factor cache...')
t0 = time.time()

# Forward-fill CN features to trading dates
cn_df['date'] = pd.to_datetime(cn_df['date'])
cn_all = pdf[['vt_symbol', 'date']].copy()

# Merge with forward fill (since holder data is quarterly)
cn_merged = cn_all.merge(cn_df, on=['vt_symbol', 'date'], how='left')

# Sort and forward fill per stock
cn_merged = cn_merged.sort_values(['vt_symbol', 'date']).reset_index(drop=True)
for col in cn_features:
    cn_merged[col] = cn_merged.groupby('vt_symbol')[col].transform(lambda x: x.ffill())

cn_merged = cn_merged[['vt_symbol', 'date'] + cn_features]

# Now merge back to main dataframe
pdf = pdf.merge(cn_merged, on=['vt_symbol', 'date'], how='left')
print(f'  Merged: {len(pdf):,} rows, {len(pdf.columns)} cols')
print(f'  {time.time()-t0:.0f}s')

# ── 5. Model Training & Backtest ─────────────────────────────────────────
print(f'\n[5/6] Walk-Forward backtest: Baseline vs CN+Features...')

# Original features (no CN features)
with open(os.path.join(ROOT, 'src', 'surge', 'params.json')) as f:
    params = json.load(f)
ORIG_FEATURES = params['selected_features']  # 40 original
ORIG_FEATURES_ALL = [f for f in ORIG_FEATURES if f in pdf.columns]

# All features (original + CN)
ALL_FEATURES = ORIG_FEATURES_ALL + cn_features
ALL_FEATURES = [f for f in ALL_FEATURES if f in pdf.columns]
print(f'  Original features: {len(ORIG_FEATURES_ALL)}')
print(f'  Total features (orig+CN): {len(ALL_FEATURES)}')

# Clean data
def prepare_data(df, feature_list):
    """Clean Inf and return (X, y, rets, nan_mask).
    LightGBM handles NaN natively, so keep them.
    """
    X_raw = df[feature_list].astype(np.float32).values
    X_raw = np.clip(X_raw, -1e10, 1e10)
    # Replace inf with NaN (LightGBM handles NaN)
    X_raw = np.where(np.isinf(X_raw), np.nan, X_raw)
    # Only drop rows where ALL features are NaN
    nan_mask = ~np.isnan(X_raw).all(axis=1)
    X = X_raw[nan_mask]
    y = df[nan_mask]['label'].values
    rets = df[nan_mask]['forward_ret'].values
    return X, y, rets, nan_mask

# Walk-forward windows
windows = [
    (2019, 2021, 2022, '2022 Bear'),
    (2019, 2022, 2023, '2023 Sideways'),
    (2019, 2023, 2024, '2024 Recovery'),
]

all_results = []

lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'num_leaves': 64,
    'learning_rate': 0.02, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_data_in_leaf': 100, 'max_depth': 10, 'verbosity': -1, 'seed': 42,
}


def run_window(train_df, test_df, feature_set, label, algo_label):
    """Train and test on given feature set"""
    if len(train_df) < 2000 or len(test_df) < 200:
        return None

    X_tr, y_tr, _, _ = prepare_data(train_df, feature_set)
    X_te, y_te, te_rets, te_mask = prepare_data(test_df, feature_set)

    if len(X_tr) < 1000 or len(X_te) < 100 or y_tr.sum() < 30:
        return None

    market_avg = te_rets.mean()
    n_feats = X_tr.shape[1]

    vs = int(len(X_tr) * 0.8)
    model = lgb.train(
        lgb_params,
        lgb.Dataset(X_tr[:vs], y_tr[:vs]),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_tr[vs:], y_tr[vs:])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    scores = model.predict(X_te)
    auc = roc_auc_score(y_te, scores)

    # Find best threshold
    best_excess = -999
    best_th = best_wr = best_n = 0
    for th in [t/100 for t in range(5, 70, 5)]:
        picks = scores >= th
        if picks.sum() < 5:
            continue
        avg_ret = te_rets[picks].mean()
        excess = avg_ret - market_avg
        if excess > best_excess:
            best_excess = excess
            best_th = th
            best_wr = (te_rets[picks] > 0).mean()
            best_n = picks.sum()

    return {
        'window': label,
        'algo': algo_label,
        'auc': round(auc, 4),
        'n_features': n_feats,
        'test_n': len(X_te),
        'best_th': best_th,
        'n_picks': best_n,
        'excess': f'{best_excess*100:+.2f}%',
        'wr': f'{best_wr:.1%}',
    }


for train_start, train_end, test_year, label in windows:
    print(f'\n  === {label} ===')

    train_mask = (pdf['year'] >= train_start) & (pdf['year'] <= train_end)
    test_mask = pdf['year'] == test_year
    train_df = pdf[train_mask]
    test_df = pdf[test_mask]

    # Baseline (original 40 features)
    print(f'  Training baseline (40 features)...')
    r1 = run_window(train_df, test_df, ORIG_FEATURES_ALL, label, 'Baseline')

    # CN+Original (40 + 18 = 58 features)
    print(f'  Training CN+Original ({len(ALL_FEATURES)} features)...')
    r2 = run_window(train_df, test_df, ALL_FEATURES, label, 'CN+Orig')

    for r in [r1, r2]:
        if r:
            all_results.append(r)
            print(f'    {r["algo"]:>10s}: AUC={r["auc"]:.4f} '
                  f'>>th={r["best_th"]:.2f} excess={r["excess"]} '
                  f'WR={r["wr"]} ({r["n_picks"]}picks)')

    gc.collect()

# ── 6. Results ─────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('RESULTS: Baseline (40 factors) vs CN+Original (58 factors)')
print('='*65)

if not all_results:
    print('  No results generated')
    sys.exit(0)

# Print comparison table
baselines = {r['window']: r for r in all_results if r['algo'] == 'Baseline'}
cn_results = {r['window']: r for r in all_results if r['algo'] == 'CN+Orig'}

print(f'\n{"Window":<20} | {"Metric":>10} | {"Baseline":>10} | {"CN+Orig":>10} | {"Delta":>10}')
print(f'{"-"*20}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}')

for w in ['2022 Bear', '2023 Sideways', '2024 Recovery']:
    b = baselines.get(w)
    c = cn_results.get(w)
    if not b or not c:
        continue
    d_auc = c['auc'] - b['auc']
    d_excess = float(c['excess'].replace('%','').replace('+','')) - float(b['excess'].replace('%','').replace('+',''))
    print(f'{w:<20} | {"AUC":>10} | {b["auc"]:>10.4f} | {c["auc"]:>10.4f} | {d_auc*100:>+9.2f}%')
    print(f'{w:<20} | {"Excess":>10} | {b["excess"]:>10} | {c["excess"]:>10} | {d_excess:>+9.2f}%')
    print(f'{w:<20} | {"Picks":>10} | {b["n_picks"]:>10} | {c["n_picks"]:>10} | {c["n_picks"]-b["n_picks"]:>+10}')

# Summary
avg_b_auc = np.mean([r['auc'] for r in all_results if r['algo'] == 'Baseline'])
avg_c_auc = np.mean([r['auc'] for r in all_results if r['algo'] == 'CN+Orig'])
b_excesses = [float(r['excess'].replace('%','').replace('+','')) for r in all_results if r['algo'] == 'Baseline']
c_excesses = [float(r['excess'].replace('%','').replace('+','')) for r in all_results if r['algo'] == 'CN+Orig']

print(f'\n{"-"*65}')
print(f'AVG Baseline AUC:  {avg_b_auc:.4f}')
print(f'AVG CN+Orig AUC:   {avg_c_auc:.4f}')
print(f'AVG AUC Delta:     {(avg_c_auc-avg_b_auc)*100:+.2f}%')
if b_excesses and c_excesses:
    print(f'AVG Baseline Excess: {np.mean(b_excesses):+.2f}%')
    print(f'AVG CN+Orig Excess:  {np.mean(c_excesses):+.2f}%')
    print(f'AVG Excess Delta:    {np.mean(c_excesses)-np.mean(b_excesses):+.2f}%')

# Winner
auc_delta = (avg_c_auc - avg_b_auc) * 100
ex_delta = np.mean(c_excesses) - np.mean(b_excesses) if b_excesses and c_excesses else 0

print(f'\n  CONCLUSION: ', end='')
if auc_delta > 1.0:
    print(f'CN+Original WINS (AUC +{auc_delta:.2f}%)')
elif auc_delta < -1.0:
    print(f'Baseline WINS (CN features +{auc_delta:.2f}% AUC)')
elif ex_delta > 0.5:
    print(f'CN+Original slightly better (excess +{ex_delta:.2f}%)')
else:
    print(f'TIE (within noise: AUC delta {auc_delta:+.2f}%, excess delta {ex_delta:+.2f}%)')

# ── Save ───────────────────────────────────────────────────────────────────
out = {
    'title': 'Phase 1c: A-Share Feature Integration',
    'timestamp': pd.Timestamp.now().isoformat(),
    'cn_features': cn_features,
    'n_cn_features': len(cn_features),
    'n_original_features': len(ORIG_FEATURES_ALL),
    'results': all_results,
}
out_path = os.path.join(ROOT, 'data', 'models', 'phase1c_results.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
print(f'\n  Saved: {out_path}')
print(f'  Total time: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)')
print('='*65)
