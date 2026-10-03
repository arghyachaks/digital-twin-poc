"""
Step 5: LSTM remaining-useful-life model, compared head-to-head with the gradient-boosted model
==============================================================================================
    python ml/train_lstm.py              (or scripts\\8_train_lstm.bat; run ml/train.py first)

The boosted model sees hand-made window features. The LSTM instead reads the last 24 hours of raw sensor
history (48 steps x 12 sensors, each relative to the pump's own baseline) and learns its own features.
Both are scored on exactly the same test rows (pumps never seen in training); results, the comparison
figure and the model are logged to the same MLflow experiment as run "lstm-vN".

The live twin keeps serving the boosted model: it is explainable with SHAP and does not depend on the
sampling rate. This script answers "would a deep sequence model do better?" with numbers.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as FE  # noqa: E402

DATA = os.path.join(HERE, "data", "pump_runs.csv.gz")
REPORTS = os.path.join(HERE, "reports")
MODELS = os.path.join(HERE, "models")
SEQ = 48            # 48 x 30 min = 24 h of history
SEED = 7
EPOCHS = 16


def log(m):
    print(f"[lstm] {m}", flush=True)


def relative(X, base):
    """Per-step sensor values relative to baseline (temperature as C difference / 10)."""
    R = np.empty_like(X)
    for j, s in enumerate(FE.FEAT_SENSORS):
        R[:, j] = (X[:, j] - base[j]) / 10.0 if s in FE.DIFF_SENSORS else X[:, j] / max(abs(base[j]), 1e-6) - 1.0
    # signed log compression: a failing pump's vibration can reach 10-20x baseline, which saturates LSTM gates
    return (np.sign(R) * np.log1p(np.abs(R))).astype(np.float32)


def prepare(df):
    """Sequences ending at every RUL-labelled row (fault detected, after the 12 h baseline)."""
    seqs = {"train": [], "val": [], "test": []}
    for uid, g in df.groupby("unit_id", sort=False):
        g = g.sort_values("t_h")
        t = g.t_h.to_numpy()
        X = FE.ffill(g[FE.FEAT_SENSORS].to_numpy())
        R = relative(X, FE.baseline(t, X))
        ok = np.flatnonzero((g.label.to_numpy() != "normal") & g.rul_h.notna().to_numpy() & (t >= t[0] + FE.MIN_HISTORY_H))
        split = g.split.iloc[0]
        for i in ok:
            lo = i - SEQ + 1
            s = R[max(lo, 0): i + 1]
            if lo < 0:                                   # pad the start with the first value
                s = np.vstack([np.repeat(s[:1], -lo, axis=0), s])
            seqs[split].append((uid, t[i], s, g.rul_h.iloc[i]))
    out = {}
    for k, v in seqs.items():
        out[k] = {"unit": np.array([x[0] for x in v]), "t": np.array([x[1] for x in v]),
                  "X": np.stack([x[2] for x in v]) if v else np.zeros((0, SEQ, len(FE.FEAT_SENSORS)), np.float32),
                  "y": np.array([x[3] for x in v], dtype=np.float32)}
    return out


def main():
    import torch
    from torch import nn

    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    xgb_path = os.path.join(REPORTS, "rul_test_predictions.csv")
    if not os.path.exists(xgb_path):
        sys.exit("run ml/train.py first (needs reports/rul_test_predictions.csv for the comparison)")
    df = pd.read_csv(DATA)
    D = prepare(df)
    log(f"sequences: train {len(D['train']['y']):,}, val {len(D['val']['y']):,}, test {len(D['test']['y']):,} "
        f"({time.time() - t0:.0f}s)")

    class RULNet(nn.Module):
        def __init__(self, n_in, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, dropout=0.1)
            # last state + mean over the window: the trend and the current level both matter for remaining life
            self.head = nn.Sequential(nn.Linear(2 * hidden + n_in, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            z = torch.cat([out[:, -1], out.mean(dim=1), x[:, -1]], dim=1)
            return nn.functional.softplus(self.head(z)).squeeze(-1)   # remaining life is never negative

    scale = 100.0                                         # predict RUL in units of 100 h
    # every 2nd training sequence: neighbouring 30-min windows overlap almost entirely, so this halves CPU time
    Xtr, ytr = torch.from_numpy(D["train"]["X"][::2].copy()), torch.from_numpy(D["train"]["y"][::2] / scale)
    Xva, yva = torch.from_numpy(D["val"]["X"]), D["val"]["y"]
    # per-channel standardisation from the training set
    mu, sd = Xtr.mean(dim=(0, 1), keepdim=True), Xtr.std(dim=(0, 1), keepdim=True).clamp_min(1e-3)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    model = RULNet(Xtr.shape[2])
    epochs, batch, best, best_state, history = EPOCHS, 256, 1e9, None, []
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=epochs, steps_per_epoch=(len(Xtr) + batch - 1) // batch)
    loss_fn = nn.SmoothL1Loss(beta=0.1)        # ~L1 (median), smooth near zero for stable gradients

    def predict(X):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(X[i:i + 4096]).numpy() for i in range(0, len(X), 4096)]) * scale

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        val_mae = float(np.mean(np.abs(np.clip(predict(Xva), 0, None) - yva)))
        history.append({"epoch": ep + 1, "train_mae_h": tot / len(perm) * scale, "val_mae_h": val_mae})
        log(f"epoch {ep + 1}/{epochs}: train MAE {tot / len(perm) * scale:.1f} h, val MAE {val_mae:.1f} h")
        if val_mae < best:
            best, best_state = val_mae, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # ---- head-to-head on the same test rows
    te = D["test"]
    xg = pd.read_csv(xgb_path)
    pred = np.clip(predict((torch.from_numpy(te["X"]) - mu) / sd), 0, None)
    # both test sets are built from the same rows in the same order: align by position, and verify it
    if len(xg) != len(pred) or not (xg.unit_id.to_numpy() == te["unit"]).all() or not np.allclose(xg.rul_true, te["y"], atol=0.05):
        sys.exit("test rows of ml/train.py and this script do not line up - re-run ml/train.py on the same dataset")
    cmp = xg.assign(lstm_p50=pred)
    y = cmp.rul_true.to_numpy()
    near = y <= 100

    def mae(col, m=slice(None)):
        return float(np.mean(np.abs(cmp[col].to_numpy()[m] - y[m])))

    metrics = {"rul_mae_h": mae("lstm_p50"), "rul_mae_last100h": mae("lstm_p50", near),
               "xgb_rul_mae_h": mae("xgb_p50"), "xgb_rul_mae_last100h": mae("xgb_p50", near), "compared_rows": int(len(cmp))}
    by_fault = cmp.assign(err_lstm=np.abs(cmp.lstm_p50 - y), err_xgb=np.abs(cmp.xgb_p50 - y)).groupby("fault")[["err_xgb", "err_lstm"]].mean().round(1)
    winner = "LSTM" if metrics["rul_mae_h"] < metrics["xgb_rul_mae_h"] else "gradient boosting"
    log(f"test MAE: LSTM {metrics['rul_mae_h']:.1f} h vs boosted {metrics['xgb_rul_mae_h']:.1f} h "
        f"(last 100 h: {metrics['rul_mae_last100h']:.1f} vs {metrics['xgb_rul_mae_last100h']:.1f}) -> {winner} wins")

    os.makedirs(MODELS, exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "seq": SEQ, "sensors": FE.FEAT_SENSORS},
               os.path.join(MODELS, "rul_lstm.pt"))
    fig = comparison_figure(cmp, y, metrics)
    write_report(metrics, by_fault, history, winner)
    try:
        log_mlflow(metrics, history, fig, by_fault)
    except Exception as e:  # tracking must never lose a finished comparison
        log(f"MLflow logging skipped: {type(e).__name__}: {e}")
    log(f"done in {time.time() - t0:.0f}s")


def comparison_figure(cmp, y, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = [0, 25, 50, 100, 200, 400, 1000]
    cut = pd.cut(y, bins)
    e = pd.DataFrame({"bin": cut, "xgb": np.abs(cmp.xgb_p50 - y), "lstm": np.abs(cmp.lstm_p50 - y)}).groupby("bin", observed=True).mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(e))
    axes[0].bar(x - 0.2, e.xgb, 0.4, label=f"Gradient boosting (MAE {m['xgb_rul_mae_h']:.0f} h)", color="#2b6cb0")
    axes[0].bar(x + 0.2, e.lstm, 0.4, label=f"LSTM (MAE {m['rul_mae_h']:.0f} h)", color="#d69e2e")
    axes[0].set_xticks(x, [f"{int(b.left)}-{int(b.right)}" for b in e.index], fontsize=8)
    axes[0].set_xlabel("true remaining life (h)")
    axes[0].set_ylabel("mean absolute error (h)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Error by remaining life: lower is better", fontsize=10)
    idx = np.random.default_rng(0).choice(len(y), min(3000, len(y)), replace=False)
    axes[1].scatter(y[idx], cmp.xgb_p50.to_numpy()[idx], s=3, alpha=0.3, label="Gradient boosting", color="#2b6cb0")
    axes[1].scatter(y[idx], cmp.lstm_p50.to_numpy()[idx], s=3, alpha=0.3, label="LSTM", color="#d69e2e")
    lim = y.max()
    axes[1].plot([0, lim], [0, lim], color="#e53e3e", lw=1)
    axes[1].set_xlabel("true remaining life (h)")
    axes[1].set_ylabel("predicted (h)")
    axes[1].legend(fontsize=8, markerscale=4)
    axes[1].set_title("Predictions on unseen pumps", fontsize=10)
    fig.tight_layout()
    p = os.path.join(REPORTS, "rul_model_comparison.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def write_report(m, by_fault, history, winner):
    lines = ["# Remaining-life model comparison: gradient boosting vs LSTM", "",
             f"Same {m['compared_rows']:,} test rows from pumps neither model saw. **{winner} has the lower overall error.**", "",
             "| Model | MAE, all (h) | MAE, last 100 h (h) |", "| --- | --- | --- |",
             f"| Gradient boosting (window features, served live) | {m['xgb_rul_mae_h']:.1f} | {m['xgb_rul_mae_last100h']:.1f} |",
             f"| LSTM (24 h raw sequence) | {m['rul_mae_h']:.1f} | {m['rul_mae_last100h']:.1f} |", "",
             "## By fault (MAE, h)", "", "| Fault | Gradient boosting | LSTM |", "| --- | --- | --- |"]
    for f, r in by_fault.iterrows():
        lines.append(f"| {f} | {r.err_xgb} | {r.err_lstm} |")
    lines += ["", "## LSTM training", "", "| Epoch | Train MAE (h) | Val MAE (h) |", "| --- | --- | --- |"]
    for h in history:
        lines.append(f"| {h['epoch']} | {h['train_mae_h']:.1f} | {h['val_mae_h']:.1f} |")
    lines += ["", "The twin keeps serving the gradient-boosted model: SHAP explains it and it works at any sampling rate.",
              "Figure: `rul_model_comparison.png`."]
    open(os.path.join(REPORTS, "rul_model_comparison.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def log_mlflow(metrics, history, fig, by_fault):
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(HERE, "mlflow.db").replace("\\", "/"))
    mlflow.set_experiment("cdu100-pump-condition")
    version = json.load(open(os.path.join(MODELS, "manifest.json"))).get("version", 0) if os.path.exists(os.path.join(MODELS, "manifest.json")) else 0
    with mlflow.start_run(run_name=f"lstm-v{version}"):
        mlflow.set_tags({"model_family": "lstm", "rul_model": "pytorch lstm", "stage": "challenger"})
        mlflow.log_params({"seq_len_steps": SEQ, "seq_hours": SEQ / 2, "hidden": 64, "layers": 2, "loss": "SmoothL1",
                           "epochs": EPOCHS, "optimizer": "AdamW one-cycle 3e-3",
                           "inputs": "12 sensors relative to baseline, signed log, standardised"})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        for h in history:
            mlflow.log_metric("val_mae_h_epoch", h["val_mae_h"], step=h["epoch"])
            mlflow.log_metric("train_mae_h_epoch", h["train_mae_h"], step=h["epoch"])
        mlflow.log_artifact(fig, "figures")
        mlflow.log_artifact(os.path.join(REPORTS, "rul_model_comparison.md"))
        mlflow.log_artifact(os.path.join(MODELS, "rul_lstm.pt"), "model")
        mlflow.log_text(by_fault.to_csv(), "mae_by_fault.csv")


if __name__ == "__main__":
    main()
