# Feature analysis (UCI credit card clients, 30,000 rows)

Information value (IV) bands: < 0.02 not predictive, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, > 0.5 very strong.

| Feature | Description | IV | Strength | Risk direction | Default rate, lowest -> highest bin |
|---|---|---|---|---|---|
| `n_delays` | Months with a late payment (last 6) | 0.862 | very strong | increasing | 11.7% -> 70.3% |
| `PAY_1` | Repayment status in Sep | 0.835 | very strong | non-monotonic | 12.3% -> 69.8% |
| `n_delays_recent3` | Months with a late payment (last 3) | 0.823 | very strong | increasing | 12.5% -> 63.8% |
| `months_since_delay` | Months since most recent late payment | 0.797 | very strong | decreasing | 11.7% -> 50.4% |
| `max_delay` | Longest payment delay (months) | 0.778 | very strong | non-monotonic | 11.7% -> 83.6% |
| `PAY_2` | Repayment status in Aug | 0.470 | strong | non-monotonic | 13.8% -> 57.4% |
| `delay_trend` | Delinquency trend (latest vs. 3 months earlier) | 0.364 | strong | non-monotonic | 14.4% -> 51.0% |
| `PAY_3` | Repayment status in Jul | 0.349 | strong | non-monotonic | 15.5% -> 53.2% |
| `PAY_5` | Repayment status in May | 0.340 | strong | increasing | 16.2% -> 100.0% |
| `PAY_4` | Repayment status in Jun | 0.315 | strong | non-monotonic | 16.0% -> 54.1% |
| `PAY_6` | Repayment status in Apr | 0.294 | medium | increasing | 17.0% -> 100.0% |
| `pay_ratio_1` | Latest repayment as share of previous bill | 0.185 | medium | non-monotonic | 10.9% -> 35.4% |
| `pay_ratio_avg` | Average repayment ratio (paid / billed) | 0.185 | medium | decreasing | 12.1% -> 39.9% |
| `LIMIT_BAL` | Credit limit | 0.170 | medium | decreasing | 12.5% -> 35.9% |
| `PAY_AMT1` | Amount paid in Sep | 0.162 | medium | decreasing | 12.8% -> 35.4% |
| `PAY_AMT2` | Amount paid in Aug | 0.147 | medium | decreasing | 11.1% -> 33.2% |
| `util_avg` | Average 6-month utilisation | 0.130 | medium | non-monotonic | 14.0% -> 32.5% |
| `PAY_AMT3` | Amount paid in Jul | 0.123 | medium | decreasing | 12.6% -> 32.5% |
| `PAY_AMT4` | Amount paid in Jun | 0.105 | medium | decreasing | 13.5% -> 31.5% |
| `util_max` | Peak 6-month utilisation | 0.099 | weak | non-monotonic | 14.2% -> 30.9% |
| `PAY_AMT6` | Amount paid in Apr | 0.092 | weak | decreasing | 12.5% -> 29.1% |
| `PAY_AMT5` | Amount paid in May | 0.086 | weak | decreasing | 12.7% -> 29.9% |
| `bill_trend` | Balance growth over 6 months (share of limit) | 0.078 | weak | non-monotonic | 15.9% -> 32.2% |
| `util_1` | Credit utilisation (latest bill / limit) | 0.077 | weak | non-monotonic | 14.8% -> 28.6% |
| `n_paid_full` | Months paid in full | 0.052 | weak | decreasing | 14.3% -> 24.8% |
| `EDUCATION` | Education level | 0.037 | weak | non-monotonic | 7.1% -> 25.2% |
| `n_no_consumption` | Months without card usage | 0.037 | weak | non-monotonic | 13.5% -> 34.4% |
| `AGE` | Age | 0.019 | not predictive | non-monotonic | 19.1% -> 27.1% |
| `BILL_AMT6` | Bill amount in Apr | 0.019 | not predictive | non-monotonic | 17.2% -> 26.0% |
| `BILL_AMT5` | Bill amount in May | 0.014 | not predictive | non-monotonic | 18.9% -> 25.1% |
| `BILL_AMT1` | Bill amount in Sep | 0.012 | not predictive | non-monotonic | 19.7% -> 25.4% |
| `BILL_AMT4` | Bill amount in Jun | 0.011 | not predictive | non-monotonic | 19.5% -> 24.7% |
| `BILL_AMT2` | Bill amount in Aug | 0.011 | not predictive | non-monotonic | 20.0% -> 25.6% |
| `BILL_AMT3` | Bill amount in Jul | 0.011 | not predictive | non-monotonic | 19.9% -> 25.1% |
| `SEX` | Sex | 0.009 | not predictive | non-monotonic | 20.8% -> 24.2% |
| `MARRIAGE` | Marital status | 0.005 | not predictive | non-monotonic | 20.9% -> 23.6% |
