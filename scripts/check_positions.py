"""
持仓每日检查: 硬止损 -25% + 40天到期提醒

用法: python check_positions.py
"""

import os, sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle

ROOT = '/mnt/d/AIHedgeFund/ai-hedge-fund-main'
CACHE = os.path.join(ROOT, 'data', 'cache')
POSITIONS = os.path.join(ROOT, 'data', 'positions_golden_pit.csv')

STOP_LOSS_PCT = -0.25
HOLD_DAYS = 40


def code_to_full(code):
    """000001 → 000001.SZ, 600001 → 600001.SH"""
    code = str(code).zfill(6)
    if code.startswith(('000', '001', '002', '003')):
        return f'{code}.SZ'
    return f'{code}.SH'


def load_prices():
    path = os.path.join(CACHE, 'prices_full.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def check_positions():
    if not os.path.exists(POSITIONS):
        print(f'无持仓文件: {POSITIONS}')
        return

    df = pd.read_csv(POSITIONS)
    if len(df) == 0:
        print('无持仓')
        return

    prices = load_prices()
    today = datetime.now()

    print(f'{"="*60}')
    print(f'持仓检查 — {today.strftime("%Y-%m-%d %H:%M")}')
    print(f'{"="*60}')
    print(f'硬止损: {STOP_LOSS_PCT*100:.0f}% | 持有期: {HOLD_DAYS}天')
    print()

    alerts = []
    total_entry = 0
    total_market = 0

    for _, row in df.iterrows():
        code = str(row['code']).zfill(6)
        name = row['name']
        entry_date_str = str(row['entry_date'])
        entry_price = float(row['entry_price'])
        shares = int(row['shares'])

        full_code = code_to_full(code)

        if full_code not in prices:
            print(f'  ⚠ {name}({code}): 无价格数据')
            continue

        pdf = prices[full_code].sort_values('trade_date')
        last_close = float(pdf['close'].iloc[-1])
        last_date = pdf['trade_date'].iloc[-1]
        last_date_str = (last_date.strftime('%Y-%m-%d')
                         if hasattr(last_date, 'strftime')
                         else str(last_date)[:10])

        pnl_pct = (last_close / entry_price - 1) * 100
        pnl_amt = (last_close - entry_price) * shares

        entry_dt = pd.to_datetime(entry_date_str)
        days_held = (today - entry_dt).days

        stop_price = entry_price * 0.75
        stop_triggered = last_close <= stop_price
        expired = days_held >= HOLD_DAYS

        status = '🟢'
        if stop_triggered:
            status = '🔴 STOP'
        elif expired:
            status = '⏰ 到期'

        print(f'  {status} {name}({code}) | '
              f'入场¥{entry_price:.2f} → 现价¥{last_close:.2f} | '
              f'{pnl_pct:+.1f}% (¥{pnl_amt:+.0f}) | '
              f'止损价¥{stop_price:.2f} | '
              f'持有{days_held}天')

        if stop_triggered:
            alerts.append(f'🔴 {name}({code}): 触发-25%硬止损! '
                          f'入场¥{entry_price:.2f}→现价¥{last_close:.2f} ({pnl_pct:+.1f}%)')
        if expired:
            alerts.append(f'⏰ {name}({code}): 40天到期! ({days_held}天)')

        total_entry += entry_price * shares
        total_market += last_close * shares

    if alerts:
        print(f'\n{"="*60}')
        print('⚠ 需要操作:')
        for a in alerts:
            print(f'  {a}')
    else:
        print(f'\n✅ 全部正常')

    if total_entry > 0:
        total_pnl = (total_market / total_entry - 1) * 100
        print(f'\n总市值: ¥{total_market:,.0f} | 成本: ¥{total_entry:,.0f} | '
              f'浮{total_pnl:+.1f}%')


if __name__ == '__main__':
    check_positions()
