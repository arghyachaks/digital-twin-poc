"""
Live scoring for the twin: loads the production bundle in ml/models and assesses one pump from its recent history.

    models = PumpModels.load()            # None if no trained bundle exists
    result = models.assess(t_hours, X_rows_by_FEAT_SENSORS, baseline)

result = {
  "anomaly": 1.7, "alert": True,                                # score >= 1 means "outside healthy behaviour"
  "fault": "bearing_fault", "confidence": 0.93, "probs": {...}, # classifier (only meaningful when alerted)
  "rul_h": {"p10": 120, "p50": 165, "p90": 210},                # only when a fault is identified
  "why_fault": [{"feature", "label", "value", "impact"}, ...],   # SHAP: what pushed towards this fault
  "why_rul":   [...],                                           # SHAP: what shortens the remaining life
  "model_version": 3
}
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from . import features as FE

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")


class PumpModels:
    def __init__(self, manifest):
        self.m = manifest
        self.version = manifest["version"]
        self.classes = manifest["classes"]
        self.feats = manifest["features"]
        f = manifest["files"]
        self.iforest = joblib.load(os.path.join(MODELS, f["anomaly"]))
        self.xgb = None
        if manifest["backend"] == "xgboost":
            import xgboost as xgb
            self.xgb = xgb
            self.clf = xgb.XGBClassifier()
            self.clf.load_model(os.path.join(MODELS, f["classifier"]))
            self.rul = {}
            for q in ("10", "50", "90"):
                r = xgb.XGBRegressor()
                r.load_model(os.path.join(MODELS, f[f"rul_q{q}"]))
                self.rul[q] = r
        else:
            self.clf = joblib.load(os.path.join(MODELS, f["classifier"]))
            self.rul = {q: joblib.load(os.path.join(MODELS, f[f"rul_q{q}"])) for q in ("10", "50", "90")}

    @classmethod
    def load(cls):
        path = os.path.join(MODELS, "manifest.json")
        if not os.path.exists(path):
            return None
        try:
            return cls(json.load(open(path)))
        except Exception as e:  # missing backend package, corrupt files ...
            print(f"[ml] could not load models: {e}")
            return None

    # ------------------------------------------------------------------ scoring
    def assess(self, t, X, base):
        F = FE.compute(np.asarray(t, float), np.asarray(X, float), np.asarray(base, float))[-1:]
        row = pd.DataFrame(F, columns=self.feats)
        a = self.m["anomaly"]
        score = float(max(0.0, (-self.iforest.score_samples(row)[0] - a["center"]) / (a["threshold"] - a["center"])))
        probs = self.clf.predict_proba(row)[0]
        k = int(np.argmax(probs))
        fault = self.classes[k]
        out = {"anomaly": round(score, 2), "alert": score >= 1.0, "fault": fault, "confidence": round(float(probs[k]), 3),
               "probs": {c: round(float(p), 3) for c, p in zip(self.classes, probs)}, "rul_h": None,
               "why_fault": [], "why_rul": [], "model_version": self.version}
        if fault != "normal":
            q = sorted(float(max(0.0, self.rul[x].predict(row)[0])) for x in ("10", "50", "90"))
            out["rul_h"] = {"p10": round(q[0]), "p50": round(q[1]), "p90": round(q[2])}
        if self.xgb is not None:
            d = self.xgb.DMatrix(row)
            if fault != "normal":
                c = self.clf.get_booster().predict(d, pred_contribs=True)[0][k][:-1]
                out["why_fault"] = self._reasons(c, F[0], positive=True)
                r = self.rul["50"].get_booster().predict(d, pred_contribs=True)[0][:-1]
                out["why_rul"] = self._reasons(r, F[0], positive=False)
        return out

    def _reasons(self, contrib, values, positive, n=4):
        """Top SHAP contributions: towards the fault (positive) or shortening remaining life (negative)."""
        order = np.argsort(-contrib if positive else contrib)
        res = []
        for i in order[:n]:
            if (contrib[i] <= 0) if positive else (contrib[i] >= 0):
                break
            label, txt = FE.describe(self.feats[i], float(values[i]))
            res.append({"feature": self.feats[i], "label": label, "value": txt,
                        "impact": round(float(contrib[i]), 3 if positive else 1)})
        return res
