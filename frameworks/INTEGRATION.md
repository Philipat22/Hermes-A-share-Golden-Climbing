# Framework-Funnel Integration

## Mapping: 11 Frameworks → 6-Layer Funnel

```
FRAMEWORK                    FUNNEL LAYER          ROLE
─────────────────────────────────────────────────────────────
01 Physical Layer    ───→    L1 Signal Gen         Early reject: strategy must obey physics
02 Opponent Analysis ───→    L1 Signal Gen         Who's losing money? Are they done?
03 Reflexivity       ───→    L2 Capital Flow       Market state diagnosis (4 indicators)
                             L3 Reflexivity         Distribution drift detection (advanced)

04 Quant Validation  ───→    L0 Actuarial Engine   Run distribution, effect sizes, WF, bootstrap
05 Safety Gate       ───→    L0 Actuarial Engine   8-item checklist, post-distribution
06 Incremental Audit ───→    L0 Actuarial Engine   Test every filter individually vs baseline

07 Position Sizing   ───→    L5 Position/Sizing    Kelly → tail penalty → caps
08 Portfolio Corr    ───→    L5 Position/Sizing    Signal source overlap, sector concentration
09 Tail Risk         ───→    L5 Position/Sizing    Hard stops, black swan survival

10 Actuarial Synth   ───→    L4 Human Confirm      Aggregate all frameworks → confidence score
11 Execution IronLaw ───→    Execution Layer        Final go/no-go. Cannot override.
```

## Pipeline Order

Every trade goes through ALL 11 frameworks, in this exact order:

```
                    ┌─────────────────────────┐
                    │ 01 PHYSICAL LAYER       │  REJECT? → stop
                    └───────────┬─────────────┘
                                │ pass/warn
                    ┌───────────▼─────────────┐
                    │ 02 OPPONENT ANALYSIS    │  REJECT? → stop
                    └───────────┬─────────────┘
                                │ pass/warn
                    ┌───────────▼─────────────┐
                    │ 03 REFLEXIVITY          │  warn = reduce weight
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ LAYER 0 ACTUARIAL       │
                    │  → Find signals         │
                    │  → Compute distribution │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ 04 QUANT VALIDATION     │  REJECT? → stop
                    └───────────┬─────────────┘
                                │ pass
                    ┌───────────▼─────────────┐
                    │ 05 SAFETY GATE          │  REJECT? → stop
                    └───────────┬─────────────┘
                                │ pass/warn
                    ┌───────────▼─────────────┐
                    │ 06 INCREMENTAL AUDIT    │  (for each proposed filter)
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ 07 POSITION SIZING      │  → recommended %
                    │ 08 PORTFOLIO CORR       │  → overlap check
                    │ 09 TAIL RISK            │  → hard stops
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ 10 ACTUARIAL SYNTHESIS  │  aggregate → confidence
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ 11 EXECUTION IRON LAW   │  FINAL: do or don't
                    └─────────────────────────┘
```

## Key Design Rules

1. **Any REJECT = stop the pipeline.** No voting, no override.
2. **WARN = proceed but with reduced confidence/position.**
3. **Frameworks 01-03 run before Layer 0** — if a strategy is physically impossible or the opponent is wrong, don't waste computation.
4. **Frameworks 07-09 run in parallel** — position sizing, portfolio correlation, and tail risk are independent.
5. **Framework 10 only sees verdicts** — it aggregates without re-analyzing raw data.
6. **Framework 11 enforces** — it translates synthesis into action, no exceptions.
