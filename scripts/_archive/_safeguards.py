#!/usr/bin/env python3
"""
_safeguards.py — 防翻车工具箱
===============================
覆盖我们踩过的所有坑：
  - 大数据死锁 → 分批+checkpoint
  - 编码损坏 → safe_write
  - inf/nan崩溃 → safe_data
  - JSON序列化失败 → safe_json
  - 0信号bug → mini_test
  - 无声卡死 → ProgressTracker

用法：
  from scripts._safeguards import (
      batch_process,        # 分批处理 + checkpoint续跑
      safe_data,            # 零信任数据清洗（clip/inf/nan）
      safe_json,            # JSON序列化（自动处理numpy类型）
      mini_test,            # 小样本快速验证装饰器
      ProgressTracker,      # 进度追踪器
      safe_write,           # 编码安全的文件写入
      Checkpoint,           # checkpoint管理器
      GuardRail,            # 规则列表 + 自动检查
  )
"""
import os, sys, json, time, hashlib, functools, traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Optional, Union, List, Dict

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# 1. 进度追踪器 — 防止"6小时无声卡死"
# ═══════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """
    任何预计>30秒的任务必须用此追踪器。
    自动打印进度、耗时估算，卡住时从日志知道最后一步。

    用法：
        pt = ProgressTracker(total=100, name="因子计算")
        for i, batch in enumerate(batches):
            pt.tick(f"batch {i+1}/{len(batches)}")
            # ... 干活 ...
        pt.done("完成！")
    """
    def __init__(self, total: int, name: str = "任务"):
        self.total = total
        self.name = name
        self.start = time.time()
        self.last_log = 0
        self.count = 0
        print(f"[{now()}] ▶ {name} 开始  (共{total}步)")

    def tick(self, msg: str = ""):
        self.count += 1
        elapsed = time.time() - self.start
        pct = self.count / self.total * 100
        # 每步都打log太吵，控制频率
        if elapsed - self.last_log < 2 and self.count < self.total:
            return
        self.last_log = elapsed
        remaining = (elapsed / self.count) * (self.total - self.count) if self.count > 0 else 0
        tag = f"  [{self.count}/{self.total} {pct:.0f}%]"
        if msg:
            tag += f"  {msg}"
        if remaining > 120:
            tag += f"  (预计剩余{remaining/60:.0f}min)"
        elif remaining > 10:
            tag += f"  (预计剩余{remaining:.0f}s)"
        print(f"[{now()}]{tag}")

    def done(self, msg: str = "完成"):
        elapsed = time.time() - self.start
        print(f"[{now()}] ✓ {self.name} {msg}  (耗时{elapsed:.0f}s/{elapsed/60:.1f}min)")


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ═══════════════════════════════════════════════════════════════════════
# 2. 分批处理器 + Checkpoint — 防止大数据死锁/断点丢失
# ═══════════════════════════════════════════════════════════════════════

class Checkpoint:
    """
    checkpoint管理器。每批完成写中间结果，重启自动续跑。

    用法：
        cp = Checkpoint("data/cache/factors")
        for i, batch in enumerate(batches):
            if cp.is_done(i):
                continue  # 跳过已完成批次
            result = process(batch)
            cp.save(i, result)

        # 最后合并
        final = cp.merge()
    """
    def __init__(self, dirpath: str, name: str = "checkpoint"):
        self.dir = Path(dirpath)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self._done_file = self.dir / "_done.json"
        self._done: Dict[str, list] = {"done": []}
        if self._done_file.exists():
            try:
                self._done = json.loads(self._done_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    def is_done(self, batch_id: Union[int, str]) -> bool:
        sid = str(batch_id)
        return sid in self._done["done"]

    def save(self, batch_id: Union[int, str], data: Any, fmt: str = "parquet"):
        """保存一批结果 + 标记完成"""
        sid = str(batch_id)
        if isinstance(data, pd.DataFrame):
            path = self.dir / f"batch_{sid:04d}.parquet"
            data.to_parquet(path, index=False)
        elif isinstance(data, dict):
            path = self.dir / f"batch_{sid:04d}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=_json_default)
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")

        # 标记完成（原子写入防损坏）
        if sid not in self._done["done"]:
            self._done["done"].append(sid)
        tmp = self._done_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._done, ensure_ascii=False), encoding="utf-8")
        tmp.rename(self._done_file)

    def list_done(self) -> List[int]:
        return sorted(int(s) for s in self._done["done"] if s.isdigit())

    def merge(self, pattern: str = "batch_*.parquet") -> Optional[pd.DataFrame]:
        """合并所有parquet结果"""
        files = sorted(self.dir.glob(pattern))
        if not files:
            return None
        dfs = []
        for f in files:
            dfs.append(pd.read_parquet(f))
        return pd.concat(dfs, ignore_index=True) if dfs else None

    def skip_done(self, batches: list) -> list:
        """返回未完成的批次列表（过滤掉checkpoint中的）"""
        return [b for i, b in enumerate(batches) if not self.is_done(i)]


def batch_process(
    items: list,
    process_fn: Callable,
    checkpoint_dir: str,
    batch_size: int = 50,
    name: str = "分批处理",
) -> Optional[pd.DataFrame]:
    """
    一键分批处理 + checkpoint。防止大数据死锁。

    参数：
        items: 待处理项列表（如股票代码列表）
        process_fn: 处理函数，接受 batch_items → pd.DataFrame
        checkpoint_dir: checkpoint目录
        batch_size: 每批大小（默认50）
        name: 任务名

    返回：
        合并后的DataFrame

    用法：
        df = batch_process(
            items=stock_list,
            process_fn=my_factor_computer,
            checkpoint_dir="data/cache/my_job",
            batch_size=50,
        )
    """
    cp = Checkpoint(checkpoint_dir, name)
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    pt = ProgressTracker(len(batches), name)

    for i, batch in enumerate(batches):
        if cp.is_done(i):
            pt.tick(f"batch {i+1}/{len(batches)} (已缓存)")
            continue
        result = process_fn(batch)
        cp.save(i, result)
        pt.tick(f"batch {i+1}/{len(batches)} → {len(result)}行")

    pt.done()
    merged = cp.merge()
    print(f"  合并结果: {len(merged):,}行  {len(merged.columns)}列")
    return merged


# ═══════════════════════════════════════════════════════════════════════
# 3. 零信任数据清洗 — 防止inf/nan导致ML崩溃
# ═══════════════════════════════════════════════════════════════════════

def safe_data(
    X: np.ndarray,
    clip_min: float = -1e10,
    clip_max: float = 1e10,
    report: bool = True,
    name: str = "数据",
) -> np.ndarray:
    """
    对数值特征做零信任清洗：
      1. clip极端值
      2. inf → nan
      3. 报告异常行比例

    用法：
        X = safe_data(X, name="features")
        # 然后 dropna
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

    返回：
        清洗后的数组（inf已转nan）
    """
    X = np.asarray(X, dtype=np.float64)

    # clip
    X = np.clip(X, clip_min, clip_max)

    # inf → nan
    n_inf = np.isinf(X).sum()
    if n_inf > 0:
        if report:
            print(f"  ⚠ {name}: 发现{n_inf:,}个inf值 → 转为nan")
        X[np.isinf(X)] = np.nan

    # 统计nan
    n_nan = np.isnan(X).sum()
    if n_nan > 0 and report:
        pct = n_nan / X.size * 100
        print(f"  ⚠ {name}: 发现{n_nan:,}个nan ({pct:.1f}%)")

    # 全nan列检测
    if report:
        all_nan_cols = np.isnan(X).all(axis=0).sum()
        if all_nan_cols:
            print(f"  ❌ {name}: 有{all_nan_cols}列全为nan — 建议删除")

    return X


def safe_split(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    清洗X并同步过滤y。
    返回 (clean_X, clean_y)，并报告丢弃了多少行。
    """
    original = len(X)
    X = safe_data(X)
    mask = ~np.isnan(X).any(axis=1)
    dropped = original - mask.sum()
    if dropped > 0:
        print(f"  丢弃{dropped:,}行 ({dropped/original*100:.1f}%)")
    return X[mask], y[mask]


# ═══════════════════════════════════════════════════════════════════════
# 4. 安全JSON序列化 — 防止int32/float32不可序列化崩溃
# ═══════════════════════════════════════════════════════════════════════

def _json_default(obj):
    """递归将numpy类型转Python原生类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def safe_json(obj: Any, path: Optional[str] = None, **kwargs) -> str:
    """
    安全的JSON序列化 + 可选保存。
    自动处理numpy类型、ensure_ascii=False。

    用法：
        # 只序列化
        s = safe_json(my_dict)

        # 序列化并保存
        safe_json(my_dict, "results/summary.json")
    """
    kw = {"ensure_ascii": False, "default": _json_default, "indent": 2}
    kw.update(kwargs)
    serialized = json.dumps(obj, **kw)

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(serialized)
        print(f"  ✅ 已保存 {path}  ({len(serialized)/1024:.1f}KB)")

    return serialized


# ═══════════════════════════════════════════════════════════════════════
# 5. 小样本验证器 — 防止"跑完发现逻辑错了"
# ═══════════════════════════════════════════════════════════════════════

def mini_test(
    df: pd.DataFrame,
    pct: float = 0.01,
    min_rows: int = 100,
    max_rows: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    从DataFrame中抽取小样本用于快速验证。
    新逻辑上线前，先用小样本跑 → 目视结果 → 再全量。

    参数：
        pct: 采样比例（默认1%）
        min_rows: 最少行数
        max_rows: 最多行数

    返回：
        采样后的DataFrame

    用法：
        test_df = mini_test(full_df)
        result = my_new_logic(test_df)
        print(result.describe())
        # 目视确认OK后：
        result = my_new_logic(full_df)
    """
    n = max(min_rows, min(max_rows, int(len(df) * pct)))
    if n >= len(df):
        print(f"  mini_test: 数据太少 ({len(df)}行)，直接用全量")
        return df
    sampled = df.sample(n=n, random_state=seed).sort_index()
    print(f"  mini_test: {len(df):,}行 → {len(sampled):,}行 ({len(sampled)/len(df)*100:.1f}%)")
    return sampled


def quick_check(condition: bool, msg: str = ""):
    """快速断言，失败时打印详细信息但不抛异常"""
    if not condition:
        print(f"  ❌ 检查失败: {msg}")


# ═══════════════════════════════════════════════════════════════════════
# 6. 编码安全的文件写入
# ═══════════════════════════════════════════════════════════════════════

def safe_write(path: str, content: str, encoding: str = "utf-8", newline: str = None):
    """
    写入文本文件，自动处理编码/换行。
    Windows下写.py文件自动用utf-8，写CSV用utf-8-sig (BOM)。

    参数：
        path: 文件路径
        content: 内容
        encoding: 编码（默认utf-8）
        newline: 换行符（默认None自动检测）
    """
    p = Path(path)

    # 自动选择编码
    if encoding == "auto":
        if p.suffix in {".csv", ".tsv"}:
            encoding = "utf-8-sig"
        else:
            encoding = "utf-8"

    # 自动选择换行
    if newline is None:
        newline = "\r\n" if sys.platform == "win32" else "\n"

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=encoding, newline="")
    # Windows下手动写CRLF
    raw = content.replace("\n", newline) if newline != "\n" else content
    p.write_text(raw, encoding=encoding)
    print(f"  ✅ 写入 {path}  ({len(raw)}字, {encoding})")


# ═══════════════════════════════════════════════════════════════════════
# 7. GuardRail — 执行前自动检查规则列表
# ═══════════════════════════════════════════════════════════════════════

class GuardRail:
    """
    自动化安全检查：运行前检查常见陷阱。

    用法：
        GuardRail([
            ("有nan行", lambda df: df.isna().any().any()),
            ("有inf值", lambda df: np.isinf(df.select_dtypes('number').values).any()),
            ("数据太短", lambda df: len(df) < 1000),
        ]).check(df)
    """
    def __init__(self, rules: List[tuple]):
        self.rules = rules  # [(描述, 断言函数), ...]

    def check(self, *args, **kwargs) -> bool:
        all_pass = True
        for desc, fn in self.rules:
            try:
                if fn(*args, **kwargs):
                    print(f"  ⚠ {desc}")
                    all_pass = False
            except Exception as e:
                print(f"  ❌ 检查异常 [{desc}]: {e}")
                all_pass = False
        if all_pass:
            print("  ✅ 安全检查全部通过")
        return all_pass


# ═══════════════════════════════════════════════════════════════════════
# 8. 计时器装饰器 — 任何函数自动计时
# ═══════════════════════════════════════════════════════════════════════

def timeit(func):
    """函数执行时间自动打印。用法: @timeit"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"  ⏱ {func.__name__}: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        return result
    return wrapper


# ═══════════════════════════════════════════════════════════════════════
# 示例：如果直接运行本脚本，演示所有功能
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  _safeguards.py  功能演示")
    print("=" * 55)

    # ProgressTracker
    pt = ProgressTracker(5, "演示任务")
    for i in range(5):
        time.sleep(0.1)
        pt.tick(f"步骤{i+1}")
    pt.done()

    # safe_json
    data = {"int": np.int32(42), "float": np.float64(3.14), "list": [np.int32(1), np.float32(2.0)]}
    print(f"  safe_json: {safe_json(data)}")

    # safe_data
    X = np.array([[1.0, np.inf], [np.nan, 2.0], [3.0, 4.0]])
    X_clean = safe_data(X)
    print(f"  safe_data: nan修了{np.isnan(X_clean).sum()}个")

    # mini_test
    df_demo = pd.DataFrame({"a": range(10000)})
    test = mini_test(df_demo, pct=0.05)
    print(f"  mini_test: {len(test)}行")

    print("=" * 55)
    print("  所有组件可用！在脚本中 import 即可使用")
    print("=" * 55)
