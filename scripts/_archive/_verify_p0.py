#!/usr/bin/env python3
"""P0 扫场（16板块全股，行情拉180天）"""
import sys, os
sys.path.insert(0, r'D:\AIHedgeFund\ai-hedge-fund-main')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.surge.scanner import scan_market
from src.surge.feedback import SignalMemory
from src.surge.engine import load_params

params = load_params()
print(f"参数: 强阈值={params['strong_signal']} 弱阈值={params['weak_signal']}")
print(f"权重: P={params['w_price_pattern']} V={params['w_volume']} S={params['w_sector']} A={params['w_acceleration']}")
print()

signals = scan_market(
    stock_pool=None,  # 16板块
    days=180,
    min_price=3.0,
    max_price=200.0,
    save_report=True,
    params=params,
)

print(f"\n结果: {len(signals)} 个信号")
grades = {}
for s in signals:
    g = s.get('signal_grade', 'NONE')
    grades[g] = grades.get(g, 0) + 1
print(f"等级分布: STRONG={grades.get('STRONG',0)} WEAK={grades.get('WEAK',0)} FAKE={grades.get('FAKE',0)} NONE={grades.get('NONE',0)}")

# 检查伪信号触发的统计
fake_count = sum(1 for s in signals if s.get('fake_score', 0) > 50)
high_fake = sum(1 for s in signals if s.get('fake_score', 0) > 80)
print(f"伪信号>50: {fake_count}  >80: {high_fake}")
