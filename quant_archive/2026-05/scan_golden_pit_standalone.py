"""
黄金坑历史全量扫描器
—— 从 Hermes 的 scan_golden_pit 提取的独立版本
   去除所有内部依赖，可直接在 OpenClaw 的 workspace 运行

用法: python scan_golden_pit_standalone.py
输入: data/cache/prices_full.pkl
输出: data/cache/golden_pit_independent_full.pkl
"""

import os, sys, time, pickle
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ═══════════════════════════════════ 配置 ═══════════════════════════════
CACHE = os.environ.get('CACHE_DIR', 'data/cache')
PRICES_PKL = os.path.join(CACHE, 'prices_full.pkl')
SW_PKL = os.path.join(CACHE, 'sw_industry.pkl')
OUTPUT = os.path.join(CACHE, 'golden_pit_independent_full.pkl')

EXCLUDE_SECTORS = {'银行', '港口', '证券', '钢加工', '旅游景点', '火力发电', '出版业', '供气供热'}

# ═══════════════════════════ 扫描单只股票 ═══════════════════════════════

def scan_stock_hermes_identical(code, df, ind_map):
    """
    ⚠️ 这是 Hermes scan_golden_pit 的逐行复刻
    公式必须完全一致 —— 差一个字符都不行
    """
    df = df.copy().sort_values('trade_date').reset_index(drop=True)
    close = df['close'].values.astype(float)
    vol = df['vol'].values.astype(float)
    n = len(df)
    if n < 250:
        return []

    signals = []

    # 逐日扫描 (不是只扫最新一天)
    for end in range(250, n):
        # ── 1. 趋势得分 (5条件) ──
        rolling = pd.Series(close).rolling

        ma20  = rolling(20).mean().values[end]
        ma50  = rolling(50).mean().values[end]
        ma60  = rolling(60).mean().values[end]
        ma120 = rolling(120).mean().values[end]
        ma250 = rolling(250).mean().values[end]

        # t2: 必须用 rolling(50).mean().values[max(0, end-20)]
        #     ← 这是 MA50 在 20天前的值, 不是 raw close 均值
        ma50_20d_ago = rolling(50).mean().values[max(0, end-20)]

        t1 = close[end] > ma250
        t2 = ma50 > ma50_20d_ago
        t3 = ma20 > ma60
        t4 = ma50 > ma120                               # ← 注意: MA50>MA120, 不是 MA60
        t5 = ma120 > ma250

        trend = int(t1) + int(t2) + int(t3) + int(t4) + int(t5)
        if trend < 4:
            continue

        # ── 2. 峰值: 过去120天最高价 ──
        lookback = min(120, end)
        peak_idx = end - lookback + np.argmax(close[end-lookback:end+1])
        peak_price = close[peak_idx]

        # ── 3. 回撤 ──
        current_dd = (close[end] / peak_price - 1) * 100
        if current_dd > -5 or current_dd < -18:
            continue

        # ── 4. 跌速 ──
        days_from_peak = end - peak_idx
        if days_from_peak <= 0:
            continue
        daily_speed = abs(current_dd) / days_from_peak
        if daily_speed < 0.5:
            continue

        # ── 5. 量比 ──
        dv = np.mean(vol[max(peak_idx, end-20):end+1])
        pv = np.mean(vol[max(0, peak_idx-20):peak_idx+1])
        vol_ratio = dv / pv if pv > 0 else 1
        if vol_ratio >= 3.0:
            continue

        # ── 6. -10%触达 (5天内) ──
        target = peak_price * 0.90
        cross_idx = None
        for j in range(peak_idx+1, end+1):
            if close[j] <= target:
                cross_idx = j
                break
        if cross_idx is None:
            continue
        if end - cross_idx > 5:
            continue

        # ── 7. 排除 ──
        if code.startswith('688'):
            continue
        ind = ind_map.get(code, '')
        if ind in EXCLUDE_SECTORS:
            continue

        # ── 8. 记录 ──
        ret120 = (close[end] / close[max(0, end-120)] - 1) * 100

        code_clean = code.split('.')[0]  # 000001.SZ → 000001

        date_int = int(df['trade_date'].iloc[end])

        signals.append({
            'code': code,
            'code_clean': code_clean,
            'date': date_int,         # int 格式: 20260521
            'date_str': str(date_int), # str 格式: '20260521'
            'trend': trend,
            'dd': round(current_dd, 2),
            'speed': round(daily_speed, 3),
            'vol_ratio': round(vol_ratio, 2),
            'ret120': round(ret120, 2),
            'price': round(float(close[end]), 2),
            'peak': round(float(peak_price), 2),
            'peak_idx': int(peak_idx),
            'cross_idx': int(cross_idx),
            'r40': None,  # 事后填充
        })

    return signals


# ════════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # 加载行业映射
    ind_map = {}
    if os.path.exists(SW_PKL):
        sw = pd.read_pickle(SW_PKL)
        ind_map = dict(zip(sw['ts_code'], sw['industry']))
        print(f'行业映射: {len(ind_map)}只')
    else:
        print('⚠ 无行业数据, 跳过板块过滤')

    # 加载价格
    print(f'加载 {PRICES_PKL}...')
    with open(PRICES_PKL, 'rb') as f:
        prices = pickle.load(f)
    print(f'{len(prices)} 只股票')

    # 过滤: 排除 688 + ST + 8板块
    valid_codes = []
    for code in prices.keys():
        if code.startswith('688'):
            continue
        ind = ind_map.get(code, '')
        if ind in EXCLUDE_SECTORS:
            continue
        valid_codes.append(code)

    print(f'有效股票: {len(valid_codes)} (排除688+8板块)')
    print(f'\n开始全量扫描...')
    t1 = time.time()

    # 多进程扫描
    all_signals = []
    MAX_WORKERS = 4

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(scan_stock_hermes_identical, code, prices[code], ind_map): code
            for code in valid_codes
        }

        completed = 0
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                sigs = fut.result()
                all_signals.extend(sigs)
            except Exception as e:
                print(f'  ⚠ {code} 失败: {e}')

            completed += 1
            if completed % 100 == 0:
                elapsed = time.time() - t1
                print(f'  [{completed}/{len(valid_codes)}] {elapsed:.0f}s | 信号: {len(all_signals)}')

    scan_time = time.time() - t1
    print(f'\n扫描完成: {len(all_signals)} 信号 ({scan_time:.0f}s)')

    if not all_signals:
        print('无信号！')
        return

    df = pd.DataFrame(all_signals)
    df = df.sort_values(['date', 'code']).reset_index(drop=True)

    # 保存
    df.to_pickle(OUTPUT)
    print(f'已保存: {OUTPUT} ({len(df)} rows, {len(df.columns)} cols)')

    # 对比 Hermes 缓存
    hermes_path = os.path.join(CACHE, 'golden_pit_signals_all.pkl')
    if os.path.exists(hermes_path):
        hermes = pd.read_pickle(hermes_path)
        print(f'\n{"="*60}')
        print(f'对比 Hermes 缓存')
        print(f'{"="*60}')
        print(f'Hermes 总信号: {len(hermes)} (date列类型: {hermes["date"].apply(type).value_counts().to_dict()})')
        print(f'本扫描结果:   {len(df)}')

        # 用 code + date_int 对比 (hermes用date_int列, 我们用date列=date_int)
        hermes_dates = set()
        for _, row in hermes.iterrows():
            code_clean = row['code'].split('.')[0] if '.' in str(row['code']) else str(row['code'])
            d = row.get('date_int', row['date'])
            if isinstance(d, float):
                d = int(d)
            elif isinstance(d, str):
                d = int(d)
            hermes_dates.add((code_clean, d))

        my_keys = set(zip(df['code_clean'], df['date']))

        intersection = hermes_dates & my_keys
        hermes_only = hermes_dates - my_keys
        my_only = my_keys - hermes_dates

        print(f'\n交集: {len(intersection)} ({100*len(intersection)/max(1,len(hermes_dates)):.1f}% of Hermes)')
        print(f'Hermes独有: {len(hermes_only)}')
        print(f'本扫描独有: {len(my_only)}')

        if hermes_only:
            print(f'\nHermes独有示例 (前10):')
            for c, d in sorted(list(hermes_only))[:10]:
                print(f'  {c} @ {d}')

        if my_only:
            print(f'\n本扫描独有示例 (前10):')
            for c, d in sorted(list(my_only))[:10]:
                print(f'  {c} @ {d}')

    total = time.time() - t0
    print(f'\n总耗时: {total:.0f}s ({total/60:.1f}min)')


if __name__ == '__main__':
    main()
