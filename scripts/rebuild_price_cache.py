#!/usr/bin/env python3
"""
重建前复权价格缓存

用法: python scripts/rebuild_price_cache.py

从 Tushare 重新获取 497 只 A 股日线数据，使用前复权 (adj='qfq')。
复权后价格连续，技术指标不再因除权失真。

输出: data/cache/backtest_prices_extended.pkl (覆盖旧文件)
"""
import os, sys, time, pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, '.env'))

from src.tools.data_fetcher import get_prices, normalize_ts_code, prices_to_df

# 股票池：从旧缓存中获取股票列表
OLD_CACHE = os.path.join(ROOT, 'data', 'cache', 'backtest_prices_extended.pkl')
NEW_CACHE = OLD_CACHE  # 直接覆盖

START_DATE = '2019-01-01'
END_DATE = '2026-04-30'


def main():
    # 读取旧缓存获取股票列表
    if not os.path.exists(OLD_CACHE):
        print(f"ERROR: 旧缓存不存在: {OLD_CACHE}")
        print("请先确认旧缓存路径，或手动提供股票列表")
        sys.exit(1)

    with open(OLD_CACHE, 'rb') as f:
        old_prices = pickle.load(f)

    symbols = sorted(old_prices.keys())
    print(f"股票池: {len(symbols)} 只")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")

    # 逐个获取前复权数据
    new_prices = {}
    failed = []

    print(f"\n开始获取数据 (前复权 adj='qfq')...")
    for sym in tqdm(symbols, desc="Fetching"):
        try:
            prices = get_prices(sym, START_DATE, END_DATE)
            if not prices:
                failed.append(sym)
                continue

            df = prices_to_df(prices)
            if len(df) < 200:
                failed.append(sym)
                continue

            # 确保列正确
            df = df.rename(columns={'date': 'date'})
            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(df.index)

            new_prices[sym] = df

        except Exception as e:
            print(f"  {sym}: ERROR - {e}")
            failed.append(sym)

        # 限速：Tushare 免费版有调用频率限制
        time.sleep(0.3)

    # 保存
    print(f"\n成功: {len(new_prices)} 只, 失败: {len(failed)} 只")
    if failed:
        print(f"失败列表: {failed}")

    with open(NEW_CACHE, 'wb') as f:
        pickle.dump(new_prices, f)

    print(f"缓存已保存: {NEW_CACHE}")
    print(f"文件大小: {os.path.getsize(NEW_CACHE)/1024/1024:.1f} MB")

    # 验证
    print(f"\n验证前复权效果:")
    test_sym = '000858.SZ'
    if test_sym in new_prices:
        df = new_prices[test_sym].sort_values('date').reset_index(drop=True)
        df['pct'] = df['close'].pct_change()
        big_drops = df[df['pct'] < -0.08]
        if len(big_drops) > 0:
            print(f"  {test_sym}: ⚠ 仍有 {len(big_drops)} 次 >8% 大跌 (可能不是复权问题)")
            for _, row in big_drops.head(3).iterrows():
                print(f"    {str(row['date'])[:10]}: {row['pct']:+.1%}")
        else:
            print(f"  {test_sym}: ✓ 无异常大缺口 — 前复权生效")


if __name__ == '__main__':
    main()
