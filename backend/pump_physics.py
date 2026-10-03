"""
Centrifugal pump physics for the CDU-100 twin
=============================================
One model shared by the live twin (backend/simulator.py) and the ML dataset generator
(ml/generate_dataset.py), so models are trained on exactly the signals they will see live.

Signals (what a real online condition-monitoring system on a pump provides)
  vibration_mm_s   overall radial velocity RMS, 10-1000 Hz (ISO 10816 zones: 4.5 warn, 7.1 alarm)
  vib_1x_mm_s      component at 1x running speed (~49.6 Hz)         -> imbalance
  vib_2x_mm_s      component at 2x running speed                     -> misalignment
  vib_axial_mm_s   axial overall velocity                            -> misalignment
  vib_bpfo_mm_s    bearing outer-race defect band (~3.58x)           -> bearing fault
  vib_hf_env_g     high-frequency acceleration envelope (2-10 kHz)   -> early bearing damage, cavitation
  vib_broadband_mm_s  broadband noise floor                          -> cavitation
  bearing_temp_c, motor_current_a, suction_bar, discharge_bar, flow_m3h, speed_rpm

Fault modes (labels): normal, imbalance, misalignment, bearing_fault, cavitation.
Severity d in [0, 1]; d = 1 is functional failure (the pump trips).
Works on Python floats (live twin) and on numpy arrays (vectorised dataset generation).
"""
import math

import numpy as np

FAULT_MODES = ["imbalance", "misalignment", "bearing_fault", "cavitation"]
LABELS = ["normal"] + FAULT_MODES
ALIASES = {"bearing_wear": "bearing_fault", "bearing": "bearing_fault", "unbalance": "imbalance"}

RPM_NOMINAL = 2975.0
SHAFT_HZ = RPM_NOMINAL / 60.0          # 1x ~ 49.6 Hz
BPFO_ORDER = 3.58                       # outer-race defect frequency as a multiple of 1x (typical 7-ball bearing)

# How long each fault takes from onset to failure (hours) and how fast it accelerates (d = (t/T)^shape)
PROGRESSION = {
    "imbalance":     {"hours": (150, 700), "shape": (1.0, 1.4)},
    "misalignment":  {"hours": (120, 600), "shape": (1.1, 1.6)},
    "bearing_fault": {"hours": (60, 450),  "shape": (1.8, 3.2)},   # slow start, rapid end: classic bearing wear
    "cavitation":    {"hours": (24, 220),  "shape": (1.0, 1.8)},
}

PUMP_CLASSES = {"P-101": 420.0, "P-102": 160.0, "P-103": 230.0}   # nominal flow m3/h per service


def canonical(mode):
    if not mode:
        return "normal"
    return ALIASES.get(mode, mode)


def make_unit(rng, pump_class="P-101"):
    """Unit-to-unit variation: no two pumps have identical baselines or fault sensitivities."""
    g = rng.normal
    return {
        "pump_class": pump_class,
        "flow_nom": PUMP_CLASSES.get(pump_class, 220.0),
        "k_base": float(np.clip(g(1.0, 0.12), 0.7, 1.4)),     # baseline vibration level
        "temp_off": float(g(0.0, 2.0)),                       # bearing temperature offset, C
        "k_fault": float(np.clip(g(1.0, 0.15), 0.6, 1.5)),    # how strongly faults show in vibration
        "noise": float(np.clip(g(1.0, 0.2), 0.6, 1.6)),       # sensor noise level
        "phase": float(rng.uniform(0, 2 * math.pi)),          # load cycle phase
    }


def load_profile(t_h, rng, phase):
    """Throughput fraction: daily cycle + slow random drift (vectorised over t_h)."""
    t_h = np.asarray(t_h, dtype=float)
    drift = np.cumsum(rng.normal(0, 0.004, t_h.shape)) if t_h.ndim else 0.0
    return np.clip(0.88 + 0.06 * np.sin(2 * np.pi * t_h / 24 + phase) + drift, 0.62, 1.05)


def ambient_profile(t_h):
    """Gulf-coast style ambient temperature, C (peak mid-afternoon)."""
    return 32.0 + 6.0 * np.sin(2 * np.pi * (np.asarray(t_h, dtype=float) - 9.0) / 24.0)


def sensors(rng, unit, mode, d, load=0.9, ambient=32.0):
    """Sensor readings for one pump. `mode` canonical fault name or 'normal'; d severity 0..1.
    Scalars in -> floats out; arrays in -> arrays out."""
    mode = canonical(mode)
    d = np.asarray(d, dtype=float)
    load = np.asarray(load, dtype=float)
    ambient = np.asarray(ambient, dtype=float)
    shape = np.broadcast(d, load, ambient).shape
    kb, kf, nz = unit["k_base"], unit["k_fault"], unit["noise"]

    def on(m):
        return d * kf if mode == m else np.zeros(shape)

    imb, mis, brg, cav = on("imbalance"), on("misalignment"), on("bearing_fault"), on("cavitation")

    def noisy(x, rel=0.04, add=0.02):
        return np.maximum(x * (1 + rng.normal(0, rel * nz, shape)) + rng.normal(0, add * nz, shape), 0.0)

    # vibration components (velocity RMS mm/s unless noted)
    v1x = noisy(1.10 * kb * (0.85 + 0.15 * load) + 9.0 * imb + 2.2 * mis + 0.6 * brg)
    v2x = noisy(0.45 * kb + 0.4 * imb + 7.5 * mis + 0.3 * brg)
    vax = noisy(0.70 * kb + 0.5 * imb + 6.5 * mis)
    vbp = noisy(0.25 * kb + 8.0 * brg ** 1.5 + 0.4 * cav)
    vbb = noisy(0.55 * kb + 0.6 * brg + 7.5 * cav * (0.7 + 0.3 * np.abs(rng.normal(0, 1, shape))))
    henv = noisy(0.18 * kb + 3.5 * brg ** 0.8 + 2.2 * cav, rel=0.08, add=0.02)   # g, rises EARLY for bearings
    overall = np.sqrt(v1x ** 2 + v2x ** 2 + vbp ** 2 + vbb ** 2)

    # process & thermal
    btemp = 52 + 9 * load + 0.35 * (ambient - 32) + unit["temp_off"] + 34 * brg ** 1.5 + 11 * mis + 4 * imb
    current = 140 + 45 * load + 9 * mis + 6 * brg + 3 * imb - 6 * cav
    suction = 2.3 - 1.7 * cav + 0.15 * (1 - load)
    flow = unit["flow_nom"] * load * (1 - 0.10 * cav - 0.02 * brg)
    discharge = 18.6 - 2.2 * (load - 0.88) - 3.2 * cav - 0.6 * brg
    speed = RPM_NOMINAL + 12 - 18 * load

    out = {
        "vibration_mm_s": overall,
        "vib_1x_mm_s": v1x,
        "vib_2x_mm_s": v2x,
        "vib_axial_mm_s": vax,
        "vib_bpfo_mm_s": vbp,
        "vib_hf_env_g": henv,
        "vib_broadband_mm_s": vbb,
        "bearing_temp_c": btemp + rng.normal(0, 0.4 * nz, shape),
        "motor_current_a": current + rng.normal(0, 1.2 * nz, shape) + rng.normal(0, 4.0, shape) * cav,
        "suction_bar": np.maximum(suction + rng.normal(0, 0.03 * nz, shape) + rng.normal(0, 0.25, shape) * cav, 0.05),
        "discharge_bar": discharge + rng.normal(0, 0.08 * nz, shape) + rng.normal(0, 0.6, shape) * cav,
        "flow_m3h": flow + rng.normal(0, 2.0 * nz, shape),
        "speed_rpm": speed + rng.normal(0, 2.0, shape),
    }
    if not shape:
        return {k: float(v) for k, v in out.items()}
    return out


def standby_sensors(rng):
    """Spare pump not running."""
    return {"vibration_mm_s": abs(rng.normal(0, 0.05)), "vib_1x_mm_s": 0.0, "vib_2x_mm_s": 0.0, "vib_axial_mm_s": 0.0,
            "vib_bpfo_mm_s": 0.0, "vib_hf_env_g": abs(rng.normal(0, 0.01)), "vib_broadband_mm_s": 0.0,
            "bearing_temp_c": 34 + rng.normal(0, 0.3), "motor_current_a": 0.0, "suction_bar": 2.4,
            "discharge_bar": 0.4 + rng.normal(0, 0.02), "flow_m3h": 0.0, "speed_rpm": 0.0}


SENSOR_UNITS = {"vibration_mm_s": "mm/s", "vib_1x_mm_s": "mm/s", "vib_2x_mm_s": "mm/s", "vib_axial_mm_s": "mm/s",
                "vib_bpfo_mm_s": "mm/s", "vib_hf_env_g": "g", "vib_broadband_mm_s": "mm/s", "bearing_temp_c": "°C",
                "motor_current_a": "A", "suction_bar": "bar", "discharge_bar": "bar", "flow_m3h": "m³/h",
                "speed_rpm": "rpm"}
