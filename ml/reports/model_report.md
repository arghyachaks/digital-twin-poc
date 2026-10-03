# Pump condition models — v1

Trained 2026-10-03T08:46:02 with `xgboost` on 280 pumps; every number below is measured on 60 pumps the models never saw.

## Results

| Model | Metric | ML | Baseline |
| --- | --- | --- | --- |
| Anomaly detection | median warning before failure | 195 h | 88 h (ISO 4.5 mm/s alarm) |
| Anomaly detection | false alarms on healthy pumps | 0.01% | — |
| Fault classifier | macro-F1, all rows | 0.993 | 0.990 (rule-based analysis) |
| Fault classifier | accuracy on early faults (severity < 0.3) | 99.0% | 95.0% |
| Remaining life | mean absolute error | 38 h | — |
| Remaining life | error in the last 100 h | 25 h | — |
| Remaining life | 10–90% interval covers the truth | 81% | 80% target |

## Warning lead time by fault (median hours before failure)

| Fault | ML anomaly model | ISO alarm |
| --- | --- | --- |
| bearing_fault | 191 | 46 |
| cavitation | 96 | 39 |
| imbalance | 337 | 227 |
| misalignment | 366 | 184 |

## Per-class F1

| Class | F1 |
| --- | --- |
| normal | 0.994 |
| imbalance | 0.990 |
| misalignment | 0.995 |
| bearing_fault | 0.990 |
| cavitation | 0.995 |

## Top drivers (SHAP)

| Fault classifier | Remaining life |
| --- | --- |
| 2x vibration (misalignment) | overall vibration |
| 1x vibration (imbalance) | overall vibration trend |
| overall vibration | broadband vibration |
| high-frequency envelope | high-frequency envelope |
| suction pressure | axial vibration |
| broadband vibration | bearing temperature |
| axial vibration | 1x vibration (imbalance) |
| bearing defect band (BPFO) | axial vibration trend |

Figures: `confusion_matrix.png`, `warning_lead_time.png`, `rul_pred_vs_true.png`, `shap_*.png`.
