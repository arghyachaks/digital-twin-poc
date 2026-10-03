"""
Train the CDU-100 pump condition models and log everything to MLflow
====================================================================
    python ml/train.py                   (or scripts\\6_train_models.bat)

Models (all on ml/features.py features):
  1. Anomaly detection  - Isolation Forest trained on healthy data only -> "something's off" score
  2. Fault classifier   - XGBoost multi-class: normal / imbalance / misalignment / bearing_fault / cavitation
  3. Remaining life     - XGBoost quantile regressors (10th / 50th / 90th percentile) -> "~72 h (55-90)"
  4. Explainability     - SHAP values (XGBoost TreeSHAP) -> global importance + per-prediction reasons

Evaluated on pumps never seen in training (split by pump), against two baselines:
  - ISO 10816 vibration alarm (4.5 mm/s), for detection lead time
  - the rule-based vibration analysis the assistant used before (app/agent.py), for fault classification

Outputs
  ml/models/      the production bundle the FastAPI app loads (manifest.json + model files)
  ml/reports/     model_report.md + figures
  MLflow          params, metrics, figures, model files: ml/mlflow.db (+ ml/mlartifacts); browse with
                  scripts\\7_mlflow_ui.bat -> http://localhost:5000
If XGBoost is not installed it falls back to scikit-learn gradient boosting (no SHAP).
"""
import datetime as dt
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import features as FE  # noqa: E402

DATA = os.path.join(HERE, "data", "pump_runs.csv.gz")
MODELS = os.path.join(HERE, "models")
REPORTS = os.path.join(HERE, "reports")
CLASSES = ["normal", "imbalance", "misalignment", "bearing_fault", "cavitation"]
SEED = 7

try:
    import xgboost as xgb
    BACKEND = "xgboost"
except ImportError:
    xgb = None
    BACKEND = "sklearn"

try:
    import mlflow
except ImportError:
    mlflow = None


def log(msg):
    print(f"[train] {msg}", flush=True)


# ============================================================================ data -> features
def build_features(df):
    """Per pump: baseline from its first 12 h, features for every row after that."""
    parts = []
    for uid, g in df.groupby("unit_id", sort=False):
        g = g.sort_values("t_h")
        t = g["t_h"].to_numpy()
        X = FE.ffill(g[FE.FEAT_SENSORS].to_numpy())
        base = FE.baseline(t, X)
        F = FE.compute(t, X, base)
        keep = t >= t[0] + FE.MIN_HISTORY_H
        meta = g.loc[keep, ["unit_id", "split", "fault_mode", "t_h", "label", "severity", "rul_h", "vibration_mm_s"]]
        parts.append(pd.concat([meta.reset_index(drop=True), pd.DataFrame(F[keep], columns=FE.FEATURE_NAMES)], axis=1))
    return pd.concat(parts, ignore_index=True)


# ============================================================================ baselines
def rules_predict(F):
    """The pre-ML rule-based vibration analysis, expressed on the same features."""
    rel = {s: F[f"{s}__rel"].to_numpy() for s in FE.FEAT_SENSORS}
    score = np.vstack([
        rel["vib_1x_mm_s"],
        (rel["vib_2x_mm_s"] + rel["vib_axial_mm_s"]) / 2,
        rel["vib_bpfo_mm_s"] + 0.1 * rel["vib_hf_env_g"],
        np.maximum(rel["vib_broadband_mm_s"], np.maximum(0, -rel["suction_bar"]) * 8),
    ]).T
    best = score.argmax(1)
    return np.where(score.max(1) > 0.6, best + 1, 0)


def sustained_first(t, flag, k=3):
    """Time of the first run of k consecutive True values (or None)."""
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run >= k:
            return t[i - k + 1]
    return None


# ============================================================================ models
def make_classifier():
    if BACKEND == "xgboost":
        return xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                                 tree_method="hist", objective="multi:softprob", eval_metric="mlogloss",
                                 early_stopping_rounds=30, n_jobs=-1, random_state=SEED)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_depth=8, random_state=SEED)


def make_quantile(alpha):
    if BACKEND == "xgboost":
        return xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", objective="reg:quantileerror", quantile_alpha=alpha,
                                early_stopping_rounds=40, n_jobs=-1, random_state=SEED)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=300, learning_rate=0.08, random_state=SEED)


def fit(model, X, y, Xv, yv, w=None):
    if BACKEND == "xgboost":
        model.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], verbose=False)
    else:
        model.fit(X, y, sample_weight=w)
    return model


def shap_values(model, X):
    """TreeSHAP contributions from XGBoost: (rows, [classes,] features + bias)."""
    if BACKEND != "xgboost":
        return None
    return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)


# ============================================================================ main
def main():
    t0 = time.time()
    if not os.path.exists(DATA):
        sys.exit(f"{DATA} not found - run scripts\\5_generate_dataset.bat first")
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    log(f"backend: {BACKEND}; mlflow: {'yes' if mlflow else 'not installed'}")

    df = pd.read_csv(DATA)
    log(f"loaded {len(df):,} rows / {df.unit_id.nunique()} pumps")
    D = build_features(df)
    log(f"features: {len(FE.FEATURE_NAMES)} columns, {len(D):,} rows ({time.time() - t0:.0f}s)")
    tr, va, te = (D[D.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    feats = FE.FEATURE_NAMES
    metrics, figs = {}, []

    # ---------------------------------------------------------------- 1. anomaly detection
    healthy = tr[tr.label == "normal"]
    iso = IsolationForest(n_estimators=300, max_samples=1024, random_state=SEED, n_jobs=-1)
    iso.fit(healthy[feats].sample(min(80000, len(healthy)), random_state=SEED))
    s_va = -iso.score_samples(va[feats])
    s_norm = s_va[va.label.to_numpy() == "normal"]
    center, thr = float(np.median(s_norm)), float(np.quantile(s_norm, 0.995))

    def anomaly(F):
        return np.clip((-iso.score_samples(F[feats]) - center) / (thr - center), 0, None)

    te_a = anomaly(te)
    te["anomaly"] = te_a
    normal_te = te.label.to_numpy() == "normal"
    healthy_units_te = te.fault_mode.to_numpy() == "normal"
    metrics["anomaly_false_alarm_rate"] = float((te_a[healthy_units_te] >= 1).mean())
    for lo, hi in [(0.05, 0.2), (0.2, 0.5), (0.5, 1.01)]:
        m = (~normal_te) & (te.severity >= lo) & (te.severity < hi)
        metrics[f"anomaly_detect_rate_sev_{lo:.2f}_{min(hi, 1):.2f}"] = float((te_a[m] >= 1).mean())
    # detection lead time vs ISO 4.5 mm/s, per failing test pump
    lead = []
    for uid, g in te[te.fault_mode != "normal"].groupby("unit_id"):
        g = g.sort_values("t_h")
        t, end = g.t_h.to_numpy(), g.t_h.max()
        ml_t = sustained_first(t, g.anomaly.to_numpy() >= 1)
        iso_t = sustained_first(t, g.vibration_mm_s.to_numpy() >= 4.5)
        lead.append({"unit": uid, "fault": g.fault_mode.iloc[0], "ml_lead_h": end - ml_t if ml_t is not None else 0.0,
                     "iso_lead_h": end - iso_t if iso_t is not None else 0.0})
    lead = pd.DataFrame(lead)
    metrics["median_lead_h_ml"] = float(lead.ml_lead_h.median())
    metrics["median_lead_h_iso"] = float(lead.iso_lead_h.median())
    lead_by_fault = lead.groupby("fault")[["ml_lead_h", "iso_lead_h"]].median().round(1)
    log(f"anomaly: false alarms {metrics['anomaly_false_alarm_rate']:.2%}, median warning lead "
        f"{metrics['median_lead_h_ml']:.0f} h (ISO alarm {metrics['median_lead_h_iso']:.0f} h)")

    # ---------------------------------------------------------------- 2. fault classifier
    ytr = tr.label.map(CLASSES.index).to_numpy()
    yva = va.label.map(CLASSES.index).to_numpy()
    yte = te.label.map(CLASSES.index).to_numpy()
    sub = tr.index[(tr.label != "normal") | (np.arange(len(tr)) % 2 == 0)]       # thin out the many normal rows
    counts = np.bincount(ytr[sub], minlength=len(CLASSES))
    w = (len(sub) / (len(CLASSES) * counts))[ytr[sub]]
    clf = fit(make_classifier(), tr.loc[sub, feats], ytr[sub], va[feats], yva, w)
    p_te = clf.predict_proba(te[feats])
    pred = p_te.argmax(1)
    rules = rules_predict(te)
    metrics["clf_accuracy"] = float(accuracy_score(yte, pred))
    metrics["clf_macro_f1"] = float(f1_score(yte, pred, average="macro"))
    metrics["rules_macro_f1"] = float(f1_score(yte, rules, average="macro"))
    early = (yte > 0) & (te.severity.to_numpy() < 0.3)
    metrics["clf_early_accuracy"] = float((pred[early] == yte[early]).mean())
    metrics["rules_early_accuracy"] = float((rules[early] == yte[early]).mean())
    for i, c in enumerate(CLASSES):
        metrics[f"clf_f1_{c}"] = float(f1_score(yte == i, pred == i))
    cm = confusion_matrix(yte, pred, labels=range(len(CLASSES)))
    log(f"classifier: macro-F1 {metrics['clf_macro_f1']:.3f} (rules {metrics['rules_macro_f1']:.3f}); "
        f"early-stage accuracy {metrics['clf_early_accuracy']:.1%} (rules {metrics['rules_early_accuracy']:.1%})")

    # ---------------------------------------------------------------- 3. remaining useful life (quantiles)
    def rul_rows(F):
        return F[(F.label != "normal") & F.rul_h.notna()]
    rtr, rva, rte = rul_rows(tr), rul_rows(va), rul_rows(te)
    rul = {}
    for q in (0.1, 0.5, 0.9):
        rul[q] = fit(make_quantile(q), rtr[feats], rtr.rul_h, rva[feats], rva.rul_h)
    P = np.sort(np.vstack([np.clip(rul[q].predict(rte[feats]), 0, None) for q in (0.1, 0.5, 0.9)]).T, axis=1)
    y = rte.rul_h.to_numpy()
    metrics["rul_mae_h"] = float(mean_absolute_error(y, P[:, 1]))
    near = y <= 100
    metrics["rul_mae_last100h"] = float(mean_absolute_error(y[near], P[near, 1]))
    metrics["rul_interval_coverage"] = float(((y >= P[:, 0]) & (y <= P[:, 2])).mean())
    log(f"RUL: MAE {metrics['rul_mae_h']:.1f} h overall, {metrics['rul_mae_last100h']:.1f} h in the last 100 h; "
        f"10-90% interval covers {metrics['rul_interval_coverage']:.0%}")
    pd.DataFrame({"unit_id": rte.unit_id, "t_h": rte.t_h, "fault": rte.label, "rul_true": y,
                  "xgb_p10": P[:, 0], "xgb_p50": P[:, 1], "xgb_p90": P[:, 2]}).to_csv(
        os.path.join(REPORTS, "rul_test_predictions.csv"), index=False)

    # ---------------------------------------------------------------- 4. SHAP (global importance)
    importance = {}
    sv = shap_values(clf, te[feats].sample(min(4000, len(te)), random_state=SEED))
    if sv is not None:
        imp = np.abs(sv[:, :, :-1]).mean(axis=(0, 1))
        importance["classifier"] = dict(sorted(zip(feats, imp.round(4).tolist()), key=lambda kv: -kv[1])[:15])
        svr = shap_values(rul[0.5], rte[feats].sample(min(4000, len(rte)), random_state=SEED))
        impr = np.abs(svr[:, :-1]).mean(0)
        importance["rul"] = dict(sorted(zip(feats, impr.round(3).tolist()), key=lambda kv: -kv[1])[:15])

    # ---------------------------------------------------------------- figures
    figs = make_figures(cm, lead_by_fault, P, y, rte, importance, metrics)

    # ---------------------------------------------------------------- save production bundle
    version = 1
    old = os.path.join(MODELS, "manifest.json")
    if os.path.exists(old):
        version = json.load(open(old)).get("version", 0) + 1
    joblib.dump(iso, os.path.join(MODELS, "anomaly_iforest.joblib"))
    files = {"anomaly": "anomaly_iforest.joblib"}
    if BACKEND == "xgboost":
        clf.save_model(os.path.join(MODELS, "fault_classifier.json"))
        files["classifier"] = "fault_classifier.json"
        for q in (0.1, 0.5, 0.9):
            fn = f"rul_q{int(q * 100):02d}.json"
            rul[q].save_model(os.path.join(MODELS, fn))
            files[f"rul_q{int(q * 100):02d}"] = fn
    else:
        joblib.dump(clf, os.path.join(MODELS, "fault_classifier.joblib"))
        files["classifier"] = "fault_classifier.joblib"
        for q in (0.1, 0.5, 0.9):
            fn = f"rul_q{int(q * 100):02d}.joblib"
            joblib.dump(rul[q], os.path.join(MODELS, fn))
            files[f"rul_q{int(q * 100):02d}"] = fn
    params = {"backend": BACKEND, "features": len(feats), "baseline_h": FE.BASELINE_H, "mean_window_h": FE.MEAN_WIN_H,
              "slope_window_h": FE.SLOPE_WIN_H, "iforest_trees": 300, "anomaly_quantile": 0.995,
              "train_pumps": int(tr.unit_id.nunique()), "test_pumps": int(te.unit_id.nunique()), "train_rows": len(tr)}
    try:
        run_id = log_mlflow(params, metrics, figs, importance, lead_by_fault, version)
    except Exception as e:  # tracking must never stop a good model from being saved
        log(f"MLflow logging skipped: {type(e).__name__}: {e}")
        run_id = None
    manifest = {"version": version, "trained_at": dt.datetime.now().isoformat(timespec="seconds"), "backend": BACKEND,
                "mlflow_run_id": run_id, "features": feats, "classes": CLASSES, "files": files,
                "anomaly": {"center": center, "threshold": thr}, "metrics": {k: round(v, 4) for k, v in metrics.items()},
                "importance": importance}
    json.dump(manifest, open(old, "w"), indent=2)
    write_report(manifest, lead_by_fault, cm, params)
    log(f"saved model bundle v{version} to ml/models ({time.time() - t0:.0f}s total)")


# ============================================================================ reporting
def make_figures(cm, lead_by_fault, P, y, rte, importance, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), [c.replace("_", "\n") for c in CLASSES], fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Fault classifier on unseen pumps (macro-F1 {metrics['clf_macro_f1']:.2f})", fontsize=10)
    fig.tight_layout()
    out.append(_save(fig, "confusion_matrix.png"))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(lead_by_fault))
    ax.bar(x - 0.2, lead_by_fault.ml_lead_h, 0.4, label="ML anomaly model", color="#2b6cb0")
    ax.bar(x + 0.2, lead_by_fault.iso_lead_h, 0.4, label="ISO 4.5 mm/s alarm", color="#a0aec0")
    for i, (a, b) in enumerate(zip(lead_by_fault.ml_lead_h, lead_by_fault.iso_lead_h)):
        ax.text(i - 0.2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f.replace("_", " ") for f in lead_by_fault.index])
    ax.set_ylabel("median warning before failure (h)")
    ax.set_title("Warning lead time on unseen pumps: ML vs ISO alarm", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.append(_save(fig, "warning_lead_time.png"))

    fig, ax = plt.subplots(figsize=(6.4, 5))
    idx = np.random.default_rng(0).choice(len(y), min(4000, len(y)), replace=False)
    ax.scatter(y[idx], P[idx, 1], s=3, alpha=0.3, color="#2b6cb0")
    lim = max(y.max(), P[:, 1].max())
    ax.plot([0, lim], [0, lim], color="#e53e3e", lw=1)
    ax.set_xlabel("true remaining life (h)")
    ax.set_ylabel("predicted median (h)")
    ax.set_title(f"Remaining useful life on unseen pumps (MAE {metrics['rul_mae_h']:.0f} h)", fontsize=10)
    fig.tight_layout()
    out.append(_save(fig, "rul_pred_vs_true.png"))

    for key, title in (("classifier", "fault classifier"), ("rul", "remaining-life model")):
        if key not in importance:
            continue
        items = list(importance[key].items())[:12][::-1]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh([FE.describe(k, 0)[0] for k, _ in items], [v for _, v in items], color="#2b6cb0")
        ax.set_xlabel("mean |SHAP value|")
        ax.set_title(f"What drives the {title} (SHAP, unseen pumps)", fontsize=10)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        out.append(_save(fig, f"shap_{key}.png"))
    return out


def _save(fig, name):
    import matplotlib.pyplot as plt
    p = os.path.join(REPORTS, name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def log_mlflow(params, metrics, figs, importance, lead_by_fault, version):
    if mlflow is None:
        return None
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(HERE, "mlflow.db").replace("\\", "/"))
    name = "cdu100-pump-condition"
    if mlflow.get_experiment_by_name(name) is None:
        art = os.path.join(HERE, "mlartifacts")
        os.makedirs(art, exist_ok=True)
        from pathlib import Path
        mlflow.create_experiment(name, artifact_location=Path(art).resolve().as_uri())
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"gbm-v{version}") as run:
        mlflow.set_tags({"model_family": "gradient_boosting", "rul_model": f"{BACKEND} quantile", "bundle_version": version,
                         "stage": "production"})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for f in figs:
            mlflow.log_artifact(f, "figures")
        mlflow.log_dict(importance, "shap_importance.json")
        mlflow.log_text(lead_by_fault.to_csv(), "warning_lead_by_fault.csv")
        mlflow.log_artifacts(MODELS, "model_bundle")
        log(f"MLflow run {run.info.run_id} logged")
        return run.info.run_id


def write_report(m, lead_by_fault, cm, params):
    k = m["metrics"]
    lines = [
        f"# Pump condition models — v{m['version']}",
        "",
        f"Trained {m['trained_at']} with `{m['backend']}` on {params['train_pumps']} pumps; every number below is measured "
        f"on {params['test_pumps']} pumps the models never saw.",
        "",
        "## Results",
        "",
        "| Model | Metric | ML | Baseline |",
        "| --- | --- | --- | --- |",
        f"| Anomaly detection | median warning before failure | {k['median_lead_h_ml']:.0f} h | {k['median_lead_h_iso']:.0f} h (ISO 4.5 mm/s alarm) |",
        f"| Anomaly detection | false alarms on healthy pumps | {k['anomaly_false_alarm_rate']:.2%} | — |",
        f"| Fault classifier | macro-F1, all rows | {k['clf_macro_f1']:.3f} | {k['rules_macro_f1']:.3f} (rule-based analysis) |",
        f"| Fault classifier | accuracy on early faults (severity < 0.3) | {k['clf_early_accuracy']:.1%} | {k['rules_early_accuracy']:.1%} |",
        f"| Remaining life | mean absolute error | {k['rul_mae_h']:.0f} h | — |",
        f"| Remaining life | error in the last 100 h | {k['rul_mae_last100h']:.0f} h | — |",
        f"| Remaining life | 10–90% interval covers the truth | {k['rul_interval_coverage']:.0%} | 80% target |",
        "",
        "## Warning lead time by fault (median hours before failure)",
        "",
        "| Fault | ML anomaly model | ISO alarm |",
        "| --- | --- | --- |",
    ]
    for f, r in lead_by_fault.iterrows():
        lines.append(f"| {f} | {r.ml_lead_h:.0f} | {r.iso_lead_h:.0f} |")
    lines += ["", "## Per-class F1", "", "| Class | F1 |", "| --- | --- |"]
    for c in CLASSES:
        lines.append(f"| {c} | {k['clf_f1_' + c]:.3f} |")
    if m.get("importance"):
        lines += ["", "## Top drivers (SHAP)", "", "| Fault classifier | Remaining life |", "| --- | --- |"]
        a, b = list(m["importance"]["classifier"]), list(m["importance"]["rul"])
        for i in range(8):
            lines.append(f"| {FE.describe(a[i], 0)[0]} | {FE.describe(b[i], 0)[0]} |")
    lines += ["", "Figures: `confusion_matrix.png`, `warning_lead_time.png`, `rul_pred_vs_true.png`, `shap_*.png`."]
    open(os.path.join(REPORTS, "model_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
