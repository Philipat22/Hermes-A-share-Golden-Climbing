"""
Prediction Log — Record strategy votes, review outcomes.

Usage:
  python engine/prediction_log.py --record    # record today's arbitration
  python engine/prediction_log.py --review    # check past predictions
  python engine/prediction_log.py --stats     # accuracy by strategy
"""

import csv
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = PROJECT_ROOT / "data" / "prediction_log.csv"


def record_prediction(date: str, stock: str, arbitrator_result: dict):
    """Record an arbitration result for future review."""
    exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "date", "stock", "verdict", "position_pct", "confidence",
                "consensus", "buy_votes", "sell_votes", "hold_votes",
                "vote_details", "outcome_40d", "outcome_checked"
            ])
        
        result = arbitrator_result
        votes = result.get("votes", [])
        buy_n = sum(1 for v in votes if v.verdict.value == "buy")
        sell_n = sum(1 for v in votes if v.verdict.value == "sell")
        hold_n = sum(1 for v in votes if v.verdict.value == "hold")
        
        vote_detail = "; ".join(
            f"{v.strategy}={v.verdict.value}(c:{v.confidence:.0%})" 
            for v in votes
        )
        
        writer.writerow([
            date, stock,
            result.get("verdict", "hold"),
            result.get("position_pct", 0),
            result.get("confidence", 0),
            result.get("consensus_level", "unknown"),
            buy_n, sell_n, hold_n,
            vote_detail,
            "",  # outcome_40d — filled later
            "",  # outcome_checked
        ])


def review_predictions():
    """Check past predictions against actual outcomes."""
    if not LOG_FILE.exists():
        print("No prediction log found.")
        return

    rows = []
    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Load prices for outcome checking
    import pickle
    with open(PROJECT_ROOT / "data" / "cache" / "prices_full.pkl", "rb") as f:
        prices = pickle.load(f)

    updated = 0
    for row in rows:
        if row["outcome_checked"] == "yes":
            continue
        
        pred_date = pd.Timestamp(row["date"])
        if (datetime.now() - pred_date).days < 42:
            continue  # Not enough time has passed
        
        stock = row["stock"]
        if not stock or stock not in prices:
            row["outcome_40d"] = "no_data"
            row["outcome_checked"] = "yes"
            updated += 1
            continue
        
        df = prices[stock]
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df.set_index("trade_date", inplace=True)
        
        exit_dt = pred_date + timedelta(days=40)
        future = df.index[df.index >= exit_dt]
        if len(future) == 0:
            continue
        
        entry_price = df.loc[pred_date, "close"] if pred_date in df.index else None
        exit_price = df.loc[future[0], "close"]
        
        if entry_price and entry_price > 0:
            ret = (exit_price - entry_price) / entry_price
            row["outcome_40d"] = f"{ret:.4f}"
        else:
            row["outcome_40d"] = "no_entry_data"
        
        row["outcome_checked"] = "yes"
        updated += 1

    if updated > 0:
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Updated {updated} predictions.")


def show_stats():
    """Show accuracy statistics by strategy."""
    if not LOG_FILE.exists():
        print("No prediction log.")
        return

    rows = []
    with open(LOG_FILE, "r") as f:
        rows = list(csv.DictReader(f))

    checked = [r for r in rows if r["outcome_checked"] == "yes" and r["outcome_40d"]]
    
    if not checked:
        print("No reviewed predictions yet.")
        return

    # Overall
    correct = 0
    total = 0
    for r in checked:
        try:
            ret = float(r["outcome_40d"])
            verdict = r["verdict"]
            if (verdict == "buy" and ret > 0) or (verdict == "sell" and ret < 0):
                correct += 1
            total += 1
        except (ValueError, KeyError):
            pass

    print(f"\nOverall: {correct}/{total} correct ({correct/total*100:.0f}%)" if total > 0 else "No data")

    # By strategy (parse vote_details)
    strat_correct = {}
    strat_total = {}
    for r in checked:
        try:
            ret = float(r["outcome_40d"])
        except (ValueError, KeyError):
            continue
        details = r.get("vote_details", "")
        for part in details.split("; "):
            if "=" not in part:
                continue
            sname, rest = part.split("=", 1)
            svote = rest.split("(")[0] if "(" in rest else rest
            strat_total[sname] = strat_total.get(sname, 0) + 1
            if (svote == "buy" and ret > 0) or (svote == "sell" and ret < 0):
                strat_correct[sname] = strat_correct.get(sname, 0) + 1

    if strat_total:
        print("\nBy strategy:")
        for sname in sorted(strat_total.keys()):
            c = strat_correct.get(sname, 0)
            t = strat_total[sname]
            print(f"  {sname:<25s}: {c}/{t} ({c/t*100:.0f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.review:
        review_predictions()
        show_stats()
    elif args.stats:
        show_stats()
    elif args.record:
        print("Use from arbitrator pipeline to record votes.")
    else:
        review_predictions()
