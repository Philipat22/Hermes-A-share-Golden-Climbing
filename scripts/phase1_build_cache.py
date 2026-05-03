"""Phase 1: Build and cache factor dataset (one-time ~14 min)"""
import sys, os, time, pickle, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

print("=" * 50)
print("Phase 1: Build factor dataset + cache")
print("=" * 50)

t_start = time.time()

# 1. Load prices cache
print("\n[1/3] Loading price data...")
prices_path = os.path.join(ROOT, "data", "cache", "backtest_prices.pkl")
with open(prices_path, "rb") as f:
    prices_cache = pickle.load(f)
print(f"  Loaded: {len(prices_cache)} stocks")

# 2. Get stock pool
print("\n[2/3] Getting stock pool...")
from src.tools.data_fetcher import get_16_sector_stocks
sec = get_16_sector_stocks()
pool = sorted(set(sum((v[:30] for v in sec.values()), [])))
print(f"  Pool: {len(pool)} stocks")

# 3. Build dataset (factor computation)
print("\n[3/3] Computing 163 alpha factors...")
from src.ml.pipeline import SurgeMLPipeline
pln = SurgeMLPipeline()
data = pln.build_and_cache(prices_cache, pool)

elapsed = time.time() - t_start
print(f"\n{'='*50}")
print(f"DONE: {len(data['X'])} samples, {len(data['factor_names'])} factors")
print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Cached: {pln._dataset_cache_path}")
print(f"{'='*50}")
