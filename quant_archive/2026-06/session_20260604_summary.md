# Session Summary — 2026-06-04

## Key Findings

### 1. Cashflow v2 Filter
- ocfps/eps > 0.7 for 2 consecutive quarters
- Result: reduces median but tightens left tail (-36% → -25% at 120d)
- Conclusion: safety airbag, not turbo. Use in fraud filter, not signal generation.

### 2. Trend Pullback Strategy
- Confirmed dead: n=1~4 across all windows
- Sector ETF + top3 enhancement: ETF +0.68%, top3 -1.14% (worse)
- Conclusion: can't chase sector momentum in A-shares

### 3. Sector ETF + Stock Enhancement
- Sector surge → buy ETF (equal weight) + top fundamental stocks
- ETF 20d: +0.68%, Top3 20d: -1.14%
- Conclusion: doesn't work. Top stocks underperform sector average.

### 4. Signal Strictness & YTD Performance
- DD40 <= -15%: 18% of market (955/5315 on 2026-04-27)
- 2026 YTD signals (mean_rev_accel): 425 signals sampled
  - 40d: median -4.6%, WR 35%
  - 60d: median -5.7%, WR 32%
  - 120d: median -5.8%, WR 32%
- Cause: 2026 is bull market. Stocks dropping 15%+ are genuinely bad, not panic-sold.

### 5. 5-Day Confirmation Test (BIGGEST FINDING)
- 87/425 signals (20%) confirmed: 5d return > 5%
- Confirmed entry-day buy: +5.6%, WR 66%
- Confirmed 5-day delay buy: -3.0% → alpha eaten by 5d run-up
- Conclusion: enter at signal day, confirm at day 5, decide to keep or exit

### 6. MA20 Trailing Stop (BIGGEST FINDING)
- Tested on 87 CONFIRMED signals (5d > 5%) in 2026:
  - Fixed 40d: +5.6%, WR 66%
  - Fixed 60d: +5.1%, WR 63%
  - MA20 trailing: +12.6%, WR 76%, avg hold 29d  ← WINNER
  - Trail -10%: +5.8%, WR 75%, avg 24d
- MA20 trailing DOUBLES returns on confirmed signals
- Need to validate on full history (2019-2025)

### 7. Strategy Registration & System Engineering
- Registered: mean_rev_accel (+6.7% 120d), mean_rev_dual (+2.8% 60d)
- T-cost dynamic model: 3-tier slippage (0.1%/0.2%/0.4% by market cap)
- Unified screener: 3-strategy cross-check + tier grading + fraud filter
- Fraud threshold: CF/NP bottom 20%

## State at End of Day

```
Strategies registered: 7 (3 active: mean_rev_accel, mean_rev_dual, north_margin)
Paper trades: 2 closed (median -11.2%), 0 open
System engineering: ~95% complete
Strategy validation: ~80% complete
Distillation: Marks+冯柳 done, 达利欧 pending
Production: 15% (daily report works, paper tracking 2 trades, no automation)
```

## Next Session Priority
1. Validate MA20 trailing stop on FULL HISTORY (2019-2025 confirmed signals)
2. If robust → upgrade mean_rev_accel exit rule from MA60 to MA20
3. Build 3-stage entry: signal → scout(1%) → 5d confirm → concentrate(target%)
