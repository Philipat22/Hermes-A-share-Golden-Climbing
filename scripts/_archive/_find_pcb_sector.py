# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())
os.environ["PYTHONIOENCODING"] = "utf-8"

import importlib.util
spec = importlib.util.spec_from_file_location('sector_map', r'D:\AIHedgeFund\ai-hedge-fund-main\src\utils\sector_map.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sm = mod.sector_map_instance

# 搜索 PCB/元件/印制 关键词
print("=== PCB/元件 相关申万行业 ===")
for item in sm.sectors:
    for kw in ['印制电路板', '元件', 'pcb', '印刷电路', '印制板', '电子元件', '元件制造', '被动元件', '印制板']:
        if kw in item['industry']:
            print(f"[{item['industry']}] -> {item['count']}只: {item['sample'][:5]}")
            break

print("\n=== 全部板块列表 ===")
for item in sm.sectors:
    print(f"  {item['industry']}: {item['count']}只")
