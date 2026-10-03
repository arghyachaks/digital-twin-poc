"""
Feature engineering shared by training (ml/train.py) and live scoring (ml/predict.py).

Raw sensor readings are turned into time-window features that do not depend on the sampling rate
(training data is every 30 min, the live twin samples every few simulated minutes), and are normalised
to each pump's own healthy baseline (no two pumps have the same baseline vibration).

For every sensor in FEAT_SENSORS:
  <s>__rel      2-hour mean relative to the pump's baseline (ratio - 1; temperature as a difference in C)
  <s>__slope    12-hour trend of that relative value, per hour
and for noisy-fault sensors also
  <s>__std      2-hour standard deviation relative to baseline (cavitation makes pressure/current fluctuate)
"""
import numpy as np

FEAT_SENSORS = ["vibration_mm_s", "vib_1x_mm_s", "vib_2x_mm_s", "vib_axial_mm_s", "vib_bpfo_mm_s", "vib_hf_env_g",
                "vib_broadband_mm_s", "bearing_temp_c", "motor_current_a", "suction_bar", "discharge_bar", "flow_m3h"]
STD_SENSORS = ["vib_broadband_mm_s", "motor_current_a", "suction_bar", "discharge_bar"]
DIFF_SENSORS = {"bearing_temp_c"}          # compared as a difference (C), not a ratio
MEAN_WIN_H, SLOPE_WIN_H, BASELINE_H = 2.0, 12.0, 12.0
MIN_HISTORY_H = BASELINE_H                 # features are only defined once a baseline exists

FEATURE_NAMES = ([f"{s}__rel" for s in FEAT_SENSORS] + [f"{s}__slope" for s in FEAT_SENSORS]
                 + [f"{s}__std" for s in STD_SENSORS])

LABELS = {"vibration_mm_s": "overall vibration", "vib_1x_mm_s": "1x vibration (imbalance)",
          "vib_2x_mm_s": "2x vibration (misalignment)", "vib_axial_mm_s": "axial vibration",
          "vib_bpfo_mm_s": "bearing defect band (BPFO)", "vib_hf_env_g": "high-frequency envelope",
          "vib_broadband_mm_s": "broadband vibration", "bearing_temp_c": "bearing temperature",
          "motor_current_a": "motor current", "suction_bar": "suction pressure", "discharge_bar": "discharge pressure",
          "flow_m3h": "flow"}


def ffill(X):
    """Forward-fill NaNs per column (sensor dropouts), back-fill leading NaNs."""
    X = np.array(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = np.isnan(col)
        if bad.all():
            col[:] = 0.0
            continue
        if bad.any():
            idx = np.where(~bad, np.arange(col.size), 0)
            np.maximum.accumulate(idx, out=idx)
            col[:] = col[idx]
            first = np.flatnonzero(~bad)[0]
            col[:first] = col[first]
    return X


def baseline(t, X):
    """Mean of each sensor over the first BASELINE_H hours of running."""
    t = np.asarray(t, float)
    m = t <= t[0] + BASELINE_H
    return np.nanmean(np.asarray(X, float)[m], axis=0)


def _relative(X, base):
    R = np.empty_like(X)
    for j, s in enumerate(FEAT_SENSORS):
        if s in DIFF_SENSORS:
            R[:, j] = X[:, j] - base[j]
        else:
            R[:, j] = X[:, j] / max(abs(base[j]), 1e-6) - 1.0
    return R


def _window_sums(t, A, win):
    """Per row: count and sums of A over rows with time in (t_i - win, t_i]."""
    start = np.searchsorted(t, t - win, side="right")
    cs = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, axis=0)])
    idx = np.arange(t.size)
    n = (idx - start + 1).astype(float)
    return n, cs[idx + 1] - cs[start], start


def compute(t, X, base):
    """Feature matrix (rows x FEATURE_NAMES) for one pump's history.
    t: hours (sorted), X: rows x FEAT_SENSORS raw values, base: baseline per sensor."""
    t = np.asarray(t, float)
    X = ffill(X)
    R = _relative(X, np.asarray(base, float))

    n, s1, _ = _window_sums(t, R, MEAN_WIN_H)
    mean_rel = s1 / n[:, None]

    # 12 h least-squares slope of the relative series (per hour)
    tc = (t - t[0])[:, None]
    nS, sR, _ = _window_sums(t, R, SLOPE_WIN_H)
    _, sT, _ = _window_sums(t, np.repeat(tc, 1, axis=1), SLOPE_WIN_H)
    _, sTT, _ = _window_sums(t, tc ** 2, SLOPE_WIN_H)
    _, sTR, _ = _window_sums(t, tc * R, SLOPE_WIN_H)
    den = nS[:, None] * sTT - sT ** 2
    slope = np.where((nS[:, None] >= 3) & (np.abs(den) > 1e-9), (nS[:, None] * sTR - sT * sR) / np.where(den == 0, 1, den), 0.0)

    cols = [FEAT_SENSORS.index(s) for s in STD_SENSORS]
    Xs = X[:, cols] / np.maximum(np.abs(np.asarray(base, float)[cols]), 1e-6)
    nW, q1, _ = _window_sums(t, Xs, MEAN_WIN_H)
    _, q2, _ = _window_sums(t, Xs ** 2, MEAN_WIN_H)
    var = np.maximum(q2 / nW[:, None] - (q1 / nW[:, None]) ** 2, 0.0)
    std = np.sqrt(var)
    return np.hstack([mean_rel, slope, std])


def describe(name, value):
    """Human wording for a feature and its current value, used for SHAP explanations."""
    sensor, kind = name.split("__")
    label = LABELS.get(sensor, sensor)
    if kind == "rel":
        txt = f"{value:+.1f} °C vs baseline" if sensor in DIFF_SENSORS else f"{value * 100:+.0f}% vs baseline"
        return label, txt
    if kind == "slope":
        txt = f"{value:+.2f} °C/h trend" if sensor in DIFF_SENSORS else f"{value * 100:+.1f}%/h trend"
        return f"{label} trend", txt
    return f"{label} fluctuation", f"{value * 100:.1f}% std"
