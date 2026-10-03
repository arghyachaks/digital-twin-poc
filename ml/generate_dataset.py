"""
Run-to-failure training dataset for the CDU-100 pump models
===========================================================
Simulates a fleet of centrifugal pumps with the SAME physics the live twin uses (backend/pump_physics.py):
each pump runs healthy for a while, then (except the 'normal' group) a fault starts and grows until failure.

    python ml/generate_dataset.py                      # 80 pumps per group -> ~400 histories
    python ml/generate_dataset.py --units-per-group 150 --step-min 15

Outputs (ml/data/):
  pump_runs.csv.gz   one row per pump per time step: sensors + labels (fault label, severity, RUL)
  units.csv          one row per pump: class, fault mode, onset/failure hours, progression shape, split
  dataset_card.md    what is in the data, how it was made, column dictionary, counts
  plots/*.png        example histories and fault signatures (sanity check that the physics separates faults)

Labels
  label      what is wrong at that moment: 'normal' until the fault is detectable (severity >= 0.05),
             then the fault mode — this is the classifier target
  severity   0..1 fault severity (1 = failure)
  rul_h      hours until this pump fails — the RUL target; empty for pumps that never fail (censored)
  split      train / val / test, assigned PER PUMP so no pump's history leaks across splits
"""
import argparse
import csv
import datetime as dt
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import pump_physics as P  # noqa: E402

SENSORS = list(P.SENSOR_UNITS)
DETECT_SEVERITY = 0.05


def iso_zone(v):
    return np.where(v >= 7.1, "ALARM", np.where(v >= 4.5, "WARN", "OK"))


def simulate_unit(rng, uid, group, step_h):
    pump_class = rng.choice(list(P.PUMP_CLASSES))
    unit = P.make_unit(rng, pump_class)
    if group == "normal":
        onset, fail_after, shape = None, None, None
        total = rng.uniform(300, 900)
    else:
        prog = P.PROGRESSION[group]
        onset = rng.uniform(24, 300)                         # healthy running before the fault starts
        fail_after = rng.uniform(*prog["hours"])             # onset -> failure
        shape = rng.uniform(*prog["shape"])                  # how sharply it accelerates at the end
        total = onset + fail_after
    t = np.arange(0.0, total + 1e-9, step_h)
    if group != "normal" and t[-1] < total:                  # always include the failure moment
        t = np.append(t, total)
    sev = np.zeros_like(t) if group == "normal" else np.clip((np.maximum(t - onset, 0) / fail_after) ** shape, 0, 1)
    load = P.load_profile(t, rng, unit["phase"])
    amb = P.ambient_profile(t)
    s = P.sensors(rng, unit, "normal" if group == "normal" else group, sev, load, amb)

    # realistic data quality: rare sensor dropouts (blank values)
    for k in SENSORS:
        drop = rng.random(t.shape) < 0.002
        s[k] = np.where(drop, np.nan, s[k])

    label = np.where(sev >= DETECT_SEVERITY, group, "normal") if group != "normal" else np.full(t.shape, "normal")
    rul = (total - t) if group != "normal" else np.full(t.shape, np.nan)
    meta = {"unit_id": uid, "pump_class": pump_class, "group": group, "onset_h": onset, "fail_h": total if group != "normal" else None,
            "fail_after_h": fail_after, "shape": shape, "rows": int(t.size),
            **{k: round(v, 4) for k, v in unit.items() if isinstance(v, float)}}
    return t, sev, load, amb, s, label, rul, meta


def assign_splits(units, rng):
    """70/15/15 by pump, stratified by group."""
    for g in sorted({u["group"] for u in units}):
        ids = [u for u in units if u["group"] == g]
        rng.shuffle(ids)
        n = len(ids)
        n_tr, n_va = int(round(n * 0.70)), int(round(n * 0.15))
        for i, u in enumerate(ids):
            u["split"] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.4g}"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-per-group", type=int, default=80)
    ap.add_argument("--step-min", type=float, default=30.0, help="sampling interval in simulated minutes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    step_h = args.step_min / 60.0
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)

    groups = P.LABELS                     # normal + 4 fault modes
    plan = [(g, i) for g in groups for i in range(args.units_per_group)]
    units, runs = [], []
    for n, (g, _) in enumerate(plan):
        uid = f"U{n + 1:04d}"
        r = simulate_unit(rng, uid, g, step_h)
        units.append(r[-1])
        runs.append(r)
    assign_splits(units, rng)
    split_of = {u["unit_id"]: u["split"] for u in units}

    # ---- pump_runs.csv.gz
    header = ["unit_id", "split", "pump_class", "fault_mode", "t_h", "label", "severity", "rul_h", "iso_zone",
              "load_frac", "ambient_c"] + SENSORS
    path = os.path.join(args.out, "pump_runs.csv.gz")
    total_rows = 0
    with gzip.open(path, "wt", newline="", compresslevel=6) as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for t, sev, load, amb, s, label, rul, meta in runs:
            zone = iso_zone(np.nan_to_num(s["vibration_mm_s"]))
            cols = [s[k] for k in SENSORS]
            for i in range(t.size):
                w.writerow([meta["unit_id"], split_of[meta["unit_id"]], meta["pump_class"], meta["group"], f"{t[i]:.2f}",
                            label[i], f"{sev[i]:.4f}", fmt(rul[i]), zone[i], f"{load[i]:.3f}", f"{amb[i]:.1f}"]
                           + [fmt(c[i]) for c in cols])
            total_rows += t.size

    # ---- units.csv
    ucols = ["unit_id", "split", "pump_class", "group", "onset_h", "fail_after_h", "fail_h", "shape", "rows",
             "k_base", "k_fault", "noise", "temp_off"]
    with open(os.path.join(args.out, "units.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ucols)
        for u in units:
            w.writerow([fmt(u.get(c)) if u.get(c) is not None else "" for c in ucols])

    stats = summarise(runs, units, total_rows, args)
    write_card(args, stats, path)
    make_plots(runs, args.out)
    print(json.dumps(stats["headline"], indent=2))
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB), units.csv, dataset_card.md, plots/")


def summarise(runs, units, total_rows, args):
    by = {}
    for t, sev, load, amb, s, label, rul, meta in runs:
        g = meta["group"]
        b = by.setdefault(g, {"pumps": 0, "rows": 0, "hours": 0.0, "fail_after": [], "warn_lead": [], "env_lead": []})
        b["pumps"] += 1
        b["rows"] += t.size
        b["hours"] += float(t[-1])
        if g != "normal":
            b["fail_after"].append(meta["fail_after_h"])
            v = np.nan_to_num(s["vibration_mm_s"])
            # lead time an ISO 4.5 mm/s warning gives before failure (first sustained crossing)
            idx = np.flatnonzero(np.convolve(v >= 4.5, np.ones(4), "same") >= 4)
            if idx.size:
                b["warn_lead"].append(float(t[-1] - t[idx[0]]))
    rows = []
    for g in P.LABELS:
        b = by[g]
        rows.append({"group": g, "pumps": b["pumps"], "rows": b["rows"], "run_hours": round(b["hours"]),
                     "median_onset_to_failure_h": round(float(np.median(b["fail_after"])), 1) if b["fail_after"] else None,
                     "median_iso_warning_lead_h": round(float(np.median(b["warn_lead"])), 1) if b["warn_lead"] else None})
    splits = {s: sum(1 for u in units if u["split"] == s) for s in ("train", "val", "test")}
    labels = {}
    for t, sev, load, amb, s, label, rul, meta in runs:
        for lab, c in zip(*np.unique(label, return_counts=True)):
            labels[str(lab)] = labels.get(str(lab), 0) + int(c)
    return {"groups": rows, "splits": splits, "labels": labels,
            "headline": {"pumps": len(units), "rows": total_rows, "splits": splits, "label_rows": labels}}


def write_card(args, stats, path):
    units = P.SENSOR_UNITS
    lines = [
        "# Pump run-to-failure dataset",
        "",
        f"Generated {dt.date.today().isoformat()} by `ml/generate_dataset.py` (seed {args.seed}, one row every "
        f"{args.step_min:g} simulated minutes). Physics: `backend/pump_physics.py`, the same model the live twin uses.",
        "",
        f"**{stats['headline']['pumps']} pumps, {stats['headline']['rows']:,} rows.** Re-create it any time with the same seed; "
        "the CSV is not stored in git.",
        "",
        "## Groups",
        "",
        "| Group | Pumps | Rows | Run hours | Median onset→failure (h) | Median ISO warning lead (h) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in stats["groups"]:
        lines.append(f"| {r['group']} | {r['pumps']} | {r['rows']:,} | {r['run_hours']:,} | "
                     f"{r['median_onset_to_failure_h'] or '—'} | {r['median_iso_warning_lead_h'] or '—'} |")
    lines += [
        "",
        "*ISO warning lead* = hours between overall vibration first staying above 4.5 mm/s and failure: the baseline a model must beat.",
        "",
        f"Splits by pump (no leakage): train {stats['splits']['train']}, val {stats['splits']['val']}, test {stats['splits']['test']}.",
        "",
        "Rows per label (classifier target): " + ", ".join(f"{k} {v:,}" for k, v in sorted(stats["labels"].items())) + ".",
        "",
        "## Columns",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        "| unit_id | pump id (U0001…) |",
        "| split | train / val / test, assigned per pump |",
        "| pump_class | service: P-101 crude charge, P-102 reflux, P-103 bottoms |",
        "| fault_mode | the fault this pump eventually develops (normal = never fails) |",
        "| t_h | hours since the pump's history starts |",
        "| label | what is wrong now: normal until severity ≥ 0.05, then the fault mode — **classifier target** |",
        "| severity | fault severity 0–1 (1 = failure) |",
        "| rul_h | hours until failure — **RUL target**; empty for pumps that never fail |",
        "| iso_zone | OK / WARN / ALARM from overall vibration (ISO 10816 4.5 / 7.1 mm/s) |",
        "| load_frac, ambient_c | operating conditions (throughput fraction, ambient °C) |",
    ]
    desc = {"vibration_mm_s": "overall radial vibration velocity RMS", "vib_1x_mm_s": "1× running speed component (imbalance)",
            "vib_2x_mm_s": "2× running speed component (misalignment)", "vib_axial_mm_s": "axial vibration (misalignment)",
            "vib_bpfo_mm_s": "bearing outer-race defect band (bearing fault)", "vib_hf_env_g": "high-frequency envelope, early bearing damage / cavitation",
            "vib_broadband_mm_s": "broadband noise floor (cavitation)", "bearing_temp_c": "bearing temperature",
            "motor_current_a": "motor current", "suction_bar": "suction pressure (falls with cavitation)",
            "discharge_bar": "discharge pressure", "flow_m3h": "flow", "speed_rpm": "shaft speed"}
    for k in SENSORS:
        lines.append(f"| {k} | {desc[k]} ({units[k]}) |")
    lines += [
        "",
        "## Fault signatures",
        "",
        "| Fault | Main evidence | Secondary |",
        "| --- | --- | --- |",
        "| imbalance | vib_1x_mm_s rises | small bearing temperature rise |",
        "| misalignment | vib_2x_mm_s and vib_axial_mm_s rise | bearing temperature and motor current rise |",
        "| bearing_fault | vib_hf_env_g rises first, then vib_bpfo_mm_s | bearing temperature climbs steeply near failure |",
        "| cavitation | vib_broadband_mm_s and vib_hf_env_g rise | suction and discharge pressure fall and fluctuate |",
        "",
        "Realism built in: unit-to-unit variation in baseline and fault sensitivity, daily load and ambient cycles, "
        "slow load drift, sensor noise, 0.2 % random sensor dropouts (blank cells), and faults that accelerate near the end.",
        "",
        "Plots: `plots/example_runs.png` (one pump per fault, onset to failure), `plots/signatures.png` (feature distributions by label).",
    ]
    with open(os.path.join(args.out, "dataset_card.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def make_plots(runs, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return
    groups = P.FAULT_MODES
    show = ["vibration_mm_s", "vib_1x_mm_s", "vib_2x_mm_s", "vib_bpfo_mm_s", "vib_hf_env_g", "vib_broadband_mm_s",
            "bearing_temp_c", "suction_bar"]
    fig, axes = plt.subplots(len(show), len(groups), figsize=(16, 15), sharex="col")
    for j, g in enumerate(groups):
        run = next(r for r in runs if r[-1]["group"] == g)
        t, sev, load, amb, s, label, rul, meta = run
        for i, k in enumerate(show):
            ax = axes[i, j]
            ax.plot(t, s[k], lw=0.8, color="#2b6cb0")
            if meta["onset_h"]:
                ax.axvline(meta["onset_h"], color="#d69e2e", lw=1, ls="--")
            if k == "vibration_mm_s":
                ax.axhline(4.5, color="#d69e2e", lw=0.8, ls=":")
                ax.axhline(7.1, color="#e53e3e", lw=0.8, ls=":")
            if j == 0:
                ax.set_ylabel(k.replace("_", " "), fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"{g}  ({meta['unit_id']}, fails at {meta['fail_h']:.0f} h)", fontsize=9)
        axes[-1, j].set_xlabel("hours", fontsize=8)
    fig.suptitle("One pump per fault mode: dashed line = fault onset; ISO 4.5 / 7.1 mm/s on overall vibration", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "example_runs.png"), dpi=110)
    plt.close(fig)

    # signatures: distribution of key features by label, late-stage rows (severity > 0.5) vs normal
    feats = ["vib_1x_mm_s", "vib_2x_mm_s", "vib_axial_mm_s", "vib_bpfo_mm_s", "vib_hf_env_g", "vib_broadband_mm_s",
             "bearing_temp_c", "suction_bar"]
    data = {lab: {f: [] for f in feats} for lab in P.LABELS}
    for t, sev, load, amb, s, label, rul, meta in runs:
        g = meta["group"]
        mask = (sev > 0.5) if g != "normal" else np.ones_like(sev, bool)
        idx = np.flatnonzero(mask)[::5]
        for f in feats:
            vals = s[f][idx]
            data[g][f].extend(vals[~np.isnan(vals)].tolist())
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, f in zip(axes.ravel(), feats):
        ax.boxplot([data[l][f] for l in P.LABELS], showfliers=False)
        ax.set_xticks(range(1, len(P.LABELS) + 1), [l.replace("_", "\n") for l in P.LABELS], fontsize=7)
        ax.set_title(f, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Fault fingerprints: each feature by label (fault rows with severity > 0.5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "signatures.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
