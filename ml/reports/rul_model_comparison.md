# Remaining-life model comparison: gradient boosting vs LSTM

Same 24,498 test rows from pumps neither model saw. **LSTM has the lower overall error.**

| Model | MAE, all (h) | MAE, last 100 h (h) |
| --- | --- | --- |
| Gradient boosting (window features, served live) | 37.5 | 25.4 |
| LSTM (24 h raw sequence) | 34.7 | 23.2 |

## By fault (MAE, h)

| Fault | Gradient boosting | LSTM |
| --- | --- | --- |
| bearing_fault | 17.7 | 16.2 |
| cavitation | 20.5 | 16.5 |
| imbalance | 58.2 | 54.3 |
| misalignment | 32.9 | 30.7 |

## LSTM training

| Epoch | Train MAE (h) | Val MAE (h) |
| --- | --- | --- |
| 1 | 75.5 | 64.8 |
| 2 | 46.1 | 61.4 |
| 3 | 42.3 | 51.1 |
| 4 | 35.8 | 44.7 |
| 5 | 32.6 | 44.6 |
| 6 | 31.3 | 42.7 |
| 7 | 30.3 | 44.1 |
| 8 | 29.2 | 43.0 |
| 9 | 28.6 | 42.3 |
| 10 | 27.5 | 44.3 |
| 11 | 27.0 | 42.8 |
| 12 | 25.9 | 43.0 |
| 13 | 25.1 | 42.5 |
| 14 | 24.4 | 42.0 |
| 15 | 24.0 | 42.2 |
| 16 | 23.7 | 42.3 |

The twin keeps serving the gradient-boosted model: SHAP explains it and it works at any sampling rate.
Figure: `rul_model_comparison.png`.
