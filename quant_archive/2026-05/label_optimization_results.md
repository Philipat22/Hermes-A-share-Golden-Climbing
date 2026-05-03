# Phase 1e: Dual Label Optimization Results

Date: 2026-05-01 22:40

## 5d_10%

- AUC: 0.6866
- Avg Excess: +25.70%
- Total Picks (3 windows): 211
- Best Feature Count: 50
- Positive Rate: 6.0%

  - 2022 Bear: AUC=0.6853, Excess=+18.87%, Picks=88
  - 2023 Sideways: AUC=0.6734, Excess=+37.30%, Picks=52
  - 2024 Recovery: AUC=0.7010, Excess=+20.93%, Picks=71

### Top Features

  1. klen
  2. std_5
  3. std_30
  4. std_10
  5. rsqr_60
  6. alpha54
  7. std_20
  8. std_60
  9. alpha52
  10. alpha6

---

## 10d_15%

- AUC: 0.6568
- Avg Excess: +54.14%
- Total Picks (3 windows): 269
- Best Feature Count: 40
- Positive Rate: 6.0%

  - 2022 Bear: AUC=0.6408, Excess=+49.96%, Picks=22
  - 2023 Sideways: AUC=0.6670, Excess=+109.70%, Picks=10
  - 2024 Recovery: AUC=0.6626, Excess=+2.76%, Picks=237

### Top Features

  1. klen
  2. std_10
  3. std_5
  4. rsqr_60
  5. std_30
  6. std_20
  7. std_60
  8. rsqr_20
  9. alpha19
  10. rsqr_30

---

Models saved as:
- data/models/surge_5d_10%.pkl
- data/models/surge_10d_15%.pkl
