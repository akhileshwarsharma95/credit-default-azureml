# Model card - credit-default-model v1

## Purpose
Probability of default (PD) for credit-card clients: the probability that a client misses the next monthly payment.
Used to rank and grade the portfolio, steer limits, and trigger early collections or review. Not an automated decline engine:
intervention decisions stay with credit management.

## Training data and lineage
- Source: UCI *Default of Credit Card Clients* (Taiwan, Apr-Sep 2005), data asset version `local`
- Training rows: 23,972 | default rate 22.1%
- Git commit: `local` | trained 2026-10-27T18:53:54+00:00
- Libraries: scikit-learn 1.5.2, lightgbm 4.5.0, pandas 2.2.3, numpy 2.1.3, scipy 1.14.1, mlflow 2.19.0

## Model
- Family: **lightgbm_monotone** (selected out-of-fold against the alternatives below)
- Monotonic constraints: True | calibration: isotonic
- Protected attributes (SEX, MARRIAGE, AGE) excluded from features: True
- Cut-off for intervention: PD >= 0.167 (cost_fp / (cost_fp + cost_fn) on calibrated PD)

| Candidate | CV AUC | CV Gini | Brier |
|---|---|---|---|
| lightgbm_monotone | 0.7893 | 0.5787 | 0.1329 |
| lightgbm_unconstrained | 0.7889 | 0.5778 | 0.1329 |
| logistic_scorecard | 0.7713 | 0.5427 | 0.1373 |

## Test-set performance
- AUC 0.779 (95% CI 0.765-0.792) | Gini 0.557 | KS 0.423
- Brier 0.1351 | ECE 0.0105 | calibration slope 0.94 | Hosmer-Lemeshow p 0.359
- Mean PD 22.0% vs observed default rate 22.1%
- At the cut-off: recall 71.0%, precision 38.9%, flagged 40.4%, expected cost 49% below the no-model baseline

## PD backtest by grade (Jeffreys test)
| Grade | PD range | Clients | Observed DR | Mean PD | Jeffreys p | Light |
|---|---|---|---|---|---|---|
| A | 0%-5% | 305 | 2.0% | 2.7% | 0.784 | green |
| B | 5%-8% | 1,238 | 7.5% | 5.8% | 0.007 | amber |
| C | 8%-12% | 798 | 9.6% | 10.6% | 0.805 | green |
| D | 12%-20% | 1,356 | 16.9% | 15.4% | 0.065 | green |
| E | 20%-35% | 1,122 | 23.0% | 23.9% | 0.767 | green |
| F | 35%-55% | 589 | 42.1% | 44.1% | 0.835 | green |
| G | 55%-100% | 585 | 70.9% | 71.5% | 0.627 | green |

## Main risk drivers (mean |SHAP|)
1. Months since most recent late payment (`months_since_delay`)
2. Repayment status in Sep (`PAY_1`)
3. Longest payment delay (months) (`max_delay`)
4. Months with a late payment (last 3) (`n_delays_recent3`)
5. Bill amount in Sep (`BILL_AMT1`)
6. Credit limit (`LIMIT_BAL`)
7. Months with a late payment (last 6) (`n_delays`)
8. Average repayment ratio (paid / billed) (`pay_ratio_avg`)
9. Amount paid in Sep (`PAY_AMT1`)
10. Amount paid in Aug (`PAY_AMT2`)

## Fairness (attributes not used as features)
- SEX disparate impact ratio: 0.945
- SEX equal opportunity gap: 0.028
- SEX calibration gap: 0.014
- AGE disparate impact ratio: 0.892
- AGE equal opportunity gap: 0.080
- AGE calibration gap: 0.024
- MARRIAGE disparate impact ratio: 0.993
- MARRIAGE equal opportunity gap: 0.048
- MARRIAGE calibration gap: 0.023

## Business assumptions (to be replaced by internal loss data)
- LGD 60%, CCF 50%, cost of a missed default 5 vs a false alarm 1
- IFRS 9 staging is illustrative: 30+ dpd or grade in ['F', 'G'] -> stage 2, 90+ dpd -> stage 3; lifetime 3 years

## Registration decision
- Registered: **True**
- auc_above_minimum: True
- calibration_ece_ok: True
- not_worse_than_champion: True

## Limitations
- Cross-sectional data from 2005 (Taiwan, consumer cards); no out-of-time sample, so backtesting is on a held-out split.
- Target is next-month payment default, not the regulatory 90-dpd definition; PDs need recalibration before IFRS 9/IRB use.
- Exposure, LGD and cost figures are assumptions for illustration.
- Monitor population stability (PSI) monthly; retrain when the score PSI exceeds 0.25 or a grade turns red.
