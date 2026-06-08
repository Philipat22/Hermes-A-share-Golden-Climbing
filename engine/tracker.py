"""
Paper Tracker — Record daily signals, review after holding period.

Usage:
  python engine/tracker.py --record --date 2026-06-02 --days 40
  python engine/tracker.py --review --days 40
  python engine/tracker.py --summary
"""

import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import strategies
strategies.load_all()
from strategies import STRATEGIES
from engine.scanner import Scanner

TRACKING_FILE = PROJECT_ROOT / "data" / "paper_trades.csv"
MAX_POSITIONS = 5


class PaperTracker:
    def __init__(self, holding_days: int = 40):
        self.holding_days = holding_days

    def record(self, date_str: str, strategy_name: str = "north_margin"):
        strat = STRATEGIES.get(strategy_name)
        if strat is None:
            print(f"Unknown strategy: {strategy_name}")
            return

        scanner = Scanner()
        scanner.load()

        class EngineAdapter:
            def __init__(self, s): self.csi300 = s.csi300; self.prices = s.prices
            def _regime_at(self, d): return scanner.regime_at(d)

        signal_filter = strat.create_signal_filter(EngineAdapter(scanner))
        candidates = scanner.scan(date_str, signal_filter)

        if not candidates:
            print(f"No candidates on {date_str}")
            return

        open_positions = self._load_open_positions()
        new_count = 0
        with open(TRACKING_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if TRACKING_FILE.stat().st_size == 0:
                writer.writerow(["entry_date", "exit_date", "code", "strategy",
                                 "entry_price", "exit_price", "return", "status"])

            for c in candidates:
                if c.code in open_positions:
                    continue
                if new_count >= MAX_POSITIONS - len(open_positions):
                    break

                exit_date = (pd.Timestamp(date_str) + timedelta(days=self.holding_days + 10)).strftime("%Y-%m-%d")
                writer.writerow([date_str, exit_date, c.code, strategy_name,
                                 f"{c.close:.2f}", "", "", "open"])
                new_count += 1

        print(f"Recorded {new_count} new positions. "
              f"Open: {len(open_positions) + new_count}/{MAX_POSITIONS}")

    def review(self):
        if not TRACKING_FILE.exists():
            print("No tracking file found.")
            return

        rows = []
        with open(TRACKING_FILE, "r") as f:
            rows = list(csv.DictReader(f))

        with open(PROJECT_ROOT / "data" / "cache" / "prices_full.pkl", "rb") as f:
            prices = pickle.load(f)

        updated = 0
        for row in rows:
            if row["status"] != "open":
                continue
            entry_dt = pd.Timestamp(row["entry_date"])
            if (datetime.now().date() - entry_dt.date()).days < self.holding_days:
                continue

            code = row["code"]
            if code not in prices:
                row["status"] = "no_data"
                updated += 1
                continue

            df = prices[code]
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)

            target = entry_dt + timedelta(days=self.holding_days)
            future = df.index[df.index >= target]
            if len(future) == 0:
                row["status"] = "pending"
                continue

            exit_price = df.loc[future[0], "close"]
            entry_price = float(row["entry_price"])
            ret = (exit_price - entry_price) / entry_price

            row["exit_price"] = f"{exit_price:.2f}"
            row["return"] = f"{ret:.4f}"
            row["exit_date"] = future[0].strftime("%Y-%m-%d")
            row["status"] = "closed"
            updated += 1

        if updated > 0:
            with open(TRACKING_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Updated {updated} positions.")

        self._print_summary(rows)

    def _print_summary(self, rows):
        closed = [r for r in rows if r["status"] == "closed" and r["return"]]
        opened = [r for r in rows if r["status"] == "open"]
        pending = [r for r in rows if r["status"] == "pending"]

        print(f"\n{'='*50}")
        print(f"  PAPER TRACKING (holding={self.holding_days}d)")
        print(f"{'='*50}")
        print(f"  Closed: {len(closed)} | Open: {len(opened)} | Pending: {len(pending)}")

        if closed:
            returns = [float(r["return"]) for r in closed]
            wins = sum(1 for r in returns if r > 0)
            print(f"  Median: {np.median(returns):+.2%}  |  WR: {wins/len(returns):.0%}")
            print(f"  Best: {max(returns):+.1%}  |  Worst: {min(returns):+.1%}")
            for r in closed[-5:]:
                print(f"  {r['entry_date']} {r['code']:<12s} {float(r['return']):>+7.1%}")

        if opened:
            print(f"\n  Open:")
            for r in opened:
                days = (datetime.now().date() - pd.Timestamp(r["entry_date"]).date()).days
                print(f"  {r['entry_date']} {r['code']:<12s} (day {days}/{self.holding_days})")

    def _load_open_positions(self) -> set:
        if not TRACKING_FILE.exists():
            return set()
        open_codes = set()
        with open(TRACKING_FILE, "r") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "open":
                    open_codes.add(row["code"])
        return open_codes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper Tracker")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--strategy", type=str, default="north_margin")
    parser.add_argument("--days", type=int, default=40, help="Holding period in trading days")
    args = parser.parse_args()

    tracker = PaperTracker(holding_days=args.days)

    if args.record:
        tracker.record(args.date, args.strategy)
    elif args.summary:
        tracker._print_summary(
            list(csv.DictReader(open(TRACKING_FILE))) if TRACKING_FILE.exists() else []
        )
    else:
        tracker.review()
