"""
Twin engine: runs the CDU-100 plant simulation inside the FastAPI process, keeps a rolling
history per sensor, and pushes telemetry to every connected browser over WebSocket.
"""
import asyncio
import json
import os
import re
import sys
from collections import deque

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))
from simulator import LAYOUT, Plant  # noqa: E402

UNITS = {"vibration_mm_s": "mm/s", "vib_1x_mm_s": "mm/s", "vib_2x_mm_s": "mm/s", "vib_axial_mm_s": "mm/s",
         "vib_bpfo_mm_s": "mm/s", "vib_hf_env_g": "g", "vib_broadband_mm_s": "mm/s", "suction_bar": "bar", "bearing_temp_c": "°C", "motor_current_a": "A", "discharge_bar": "bar",
         "flow_m3h": "m³/h", "speed_rpm": "rpm", "fouling_m2k_kw": "m²K/kW", "shell_dp_bar": "bar",
         "crude_out_c": "°C", "hot_in_c": "°C", "duty_mw": "MW", "coil_outlet_c": "°C", "tube_skin_c": "°C",
         "stack_c": "°C", "o2_pct": "%", "fuel_gas_t_h": "t/h", "top_c": "°C", "flash_zone_c": "°C",
         "top_bar": "bar", "bottom_level_pct": "%", "tray_dp_mbar": "mbar", "level_pct": "%", "temp_c": "°C",
         "level_m": "m"}


class Twin:
    def __init__(self, speed=300.0, hz=2.0, history_len=1440):
        self.plant = Plant(speed)
        self.hz = hz
        self.latest = {}
        self.history = {}          # tag -> sensor -> deque[(sim_hours, value)]
        self.base = {}             # tag -> sensor -> first (healthy) value
        self.history_len = history_len
        self.clients = set()
        self.tags = list(self.plant.assets)
        self.spec = {e["tag"]: e for e in LAYOUT["equipment"]}

    # ------------------------------------------------------------------ loop
    async def run(self):
        period = 1.0 / self.hz
        while True:
            frames = self.plant.step(period)
            t = self.plant.t_h
            for tag, f in frames.items():
                h = self.history.setdefault(tag, {})
                for k, v in f["sensors"].items():
                    h.setdefault(k, deque(maxlen=self.history_len)).append((t, v))
                    if f["status"] == "OK" and f["degradation"] < 0.1:
                        self.base.setdefault(tag, {}).setdefault(k, v)
            self.latest = {"type": "telemetry", "sim_time": self.plant.sim_time.isoformat(),
                           "speed": self.plant.speed, "assets": frames, "events": self.plant.pending[:]}
            self.plant.pending.clear()
            await self.broadcast(self.latest)
            await asyncio.sleep(period)

    async def broadcast(self, msg):
        if not self.clients:
            return
        data = json.dumps(msg)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    # ------------------------------------------------------------------ queries
    def norm_tag(self, text):
        """'p101a', 'P 101 A', 'p-101a' -> 'P-101A'; returns None if unknown."""
        if not text:
            return None
        s = re.sub(r"[^A-Z0-9]", "", str(text).upper())
        for tag in self.tags:
            if re.sub(r"[^A-Z0-9]", "", tag) == s:
                return tag
        return None

    def asset(self, tag):
        tag = self.norm_tag(tag)
        if not tag or not self.latest:
            return None, None
        return tag, self.latest["assets"].get(tag)

    def trend(self, tag, sensor=None, hours=12.0):
        tag = self.norm_tag(tag)
        h = self.history.get(tag, {})
        now = self.plant.t_h
        out = {}
        for k, series in h.items():
            if sensor and k != sensor:
                continue
            pts = [(t, v) for t, v in series if t >= now - hours]
            if len(pts) < 2:
                continue
            vals = [v for _, v in pts]
            dt_h = pts[-1][0] - pts[0][0] or 1e-9
            out[k] = {"unit": UNITS.get(k, ""), "first": round(vals[0], 3), "last": round(vals[-1], 3),
                      "min": round(min(vals), 3), "max": round(max(vals), 3),
                      "mean": round(sum(vals) / len(vals), 3),
                      "change_per_hour": round((vals[-1] - vals[0]) / dt_h, 4), "window_h": round(dt_h, 1)}
        return out

    def series(self, tag, sensor, points=300):
        tag = self.norm_tag(tag)
        s = list(self.history.get(tag, {}).get(sensor, []))
        step = max(1, len(s) // points)
        return [{"t_h": round(t, 3), "v": round(v, 4)} for t, v in s[::step]]

    def baseline(self, tag):
        """Earliest recorded values (healthy reference) per sensor."""
        return dict(self.base.get(self.norm_tag(tag), {}))

    def reset_baseline(self, tag=None):
        if tag:
            self.base.pop(self.norm_tag(tag), None)
        else:
            self.base.clear()

    def command(self, msg):
        if "tag" in msg:
            msg = {**msg, "tag": self.norm_tag(msg["tag"]) or msg["tag"]}
        return self.plant.command(msg)
