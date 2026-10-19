# PP1 results (simulated data)

100 simulated users, 20 sessions each; train/val/test = 60/20/20 users; mean ± std over 5 random user splits. Early stopping on validation loss.

| Window | System | Accuracy | Precision | Recall | F1 | False-adjust |
|---|---|---|---|---|---|---|
| 3 | GRU | 0.946 ± 0.013 | 0.814 ± 0.056 | 0.839 ± 0.034 | 0.825 ± 0.038 | 0.035 ± 0.014 |
| 3 | Static threshold | 0.893 ± 0.014 | 0.605 ± 0.059 | 0.830 ± 0.042 | 0.699 ± 0.051 | 0.096 ± 0.014 |
| 5 | GRU | 0.948 ± 0.017 | 0.856 ± 0.070 | 0.842 ± 0.040 | 0.847 ± 0.043 | 0.031 ± 0.019 |
| 5 | Static threshold | 0.882 ± 0.020 | 0.613 ± 0.067 | 0.834 ± 0.045 | 0.705 ± 0.057 | 0.108 ± 0.019 |
| 7 | GRU | 0.932 ± 0.015 | 0.825 ± 0.050 | 0.838 ± 0.062 | 0.829 ± 0.031 | 0.045 ± 0.020 |
| 7 | Static threshold | 0.872 ± 0.019 | 0.630 ± 0.054 | 0.837 ± 0.045 | 0.718 ± 0.048 | 0.119 ± 0.018 |

## Signal policy (window 5)

| Version | Setting flips per user | Adjusted sessions when NOT declining | Missed decline sessions |
|---|---|---|---|
| Raw GRU output (p ≥ 0.5) | 0.56 | 0.38 | 0.40 |
| Smoothing + hysteresis + cooldown | 0.56 | 0.43 | 0.70 |

One-off 'blip' users (17 across splits): sessions wrongly adjusted raw = 0, policy = 0.

Limits: all data is simulated by app/simulator.py; real accuracy depends on Components 1-3 data.
