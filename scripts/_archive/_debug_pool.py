#!/usr/bin/env python3
"""Debug 股票池 - 看看 get_stocks_by_sector 为什么只返回这么少"""
import sys, os
sys.path.insert(0, r'D:\AIHedgeFund\ai-hedge-fund-main')

from src.tools.a_stock_api import get_stocks_by_sector, SECTOR_POOL
from src.tools.data_fetcher import get_stock_info

print(f"SECTOR_POOL 共 {len(SECTOR_POOL)} 个板块:")
for sector in SECTOR_POOL:
    stocks = get_stocks_by_sector(sector)
    print(f"  {sector}: {len(stocks)} 只")
    if stocks:
        print(f"    前3: {stocks[:3]}")

total = sum(len(get_stocks_by_sector(s)) for s in SECTOR_POOL)
print(f"\n总计: {total} 只")

# 对比原始 stock_basic 数据
from src.tools.data_fetcher import pro
try:
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry')
    print(f"\nstock_basic 全量: {len(df)} 只")
    if 'industry' in df.columns:
        ind_count = df['industry'].nunique()
        print(f"  unique industries: {ind_count}")
        # 看看有没有空的
        empty_ind = df['industry'].isna().sum()
        print(f"  industry 为空: {empty_ind}")
except Exception as e:
    print(f"stock_basic 错误: {e}")
