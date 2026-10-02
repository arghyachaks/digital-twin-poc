"""
CDU-100 sensor simulator + WebSocket hub
========================================
Streams physics-flavoured telemetry for every monitored asset in config/plant_layout.json.

    pip install websockets numpy
    python backend/simulator.py                 # ws://localhost:8765, 300x real time

Messages OUT (JSON, ~2 Hz):
  {"type":"telemetry","sim_time":"2026-10-01T06:15:00","speed":300,
   "assets":{"P-101A":{"kind":"pump","status":"WARN","health":71.2,"rul_h":38.5,"anomaly":3.1,
                       "color":[1,0.55,0],"glow":2.5,"pulse":4,"sensors":{...}}, ...},
   "events":[{"ts":..., "tag":"P-101A","level":"WARN","msg":"Vibration 5.1 mm/s exceeds ISO 10816 zone B"}]}

Commands IN (JSON):
  {"cmd":"inject_fault","tag":"E-102","mode":"fouling","severity":0.4}
  {"cmd":"reset","tag":"P-101A"}   {"cmd":"reset_all"}
  {"cmd":"speed","value":600}      {"cmd":"snapshot"}
Every tick is also appended to data/telemetry_<date>.csv for ML training.
"""
import argparse
import asyncio
import csv
import datetime as dt
import json
import math
import os
import random
import threading
import webbrowser
from collections import deque
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

try:  # only needed for the standalone server below; the FastAPI app imports Plant directly
    from websockets.asyncio.server import broadcast, serve
except ImportError:
    try:
        from websockets import broadcast, serve  # type: ignore
    except ImportError:
        broadcast = serve = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAYOUT = json.load(open(os.path.join(ROOT, "config", "plant_layout.json")))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

STATUS_STYLE = {  # colour, glow, pulse speed -> drives M_TwinMaster HealthColor/HealthGlow/PulseSpeed
    "OK":      ([0.10, 1.00, 0.25], 0.0, 1.0),
    "STANDBY": ([0.20, 0.50, 1.00], 0.6, 1.0),
    "WARN":    ([1.00, 0.55, 0.00], 2.5, 4.0),
    "ALARM":   ([1.00, 0.05, 0.02], 6.0, 9.0),
    "TRIPPED": ([1.00, 0.00, 0.00], 4.0, 2.0),
}


def kind_of(mesh):
    return {"SM_Pump_Centrifugal": "pump", "SM_HeatExchanger": "exchanger", "SM_DistillationColumn": "column",
            "SM_Furnace": "furnace", "SM_StorageTank": "tank"}.get(mesh, "other")


# =========================================================================== asset models

class Asset:
    def __init__(self, spec, rng):
        self.tag, self.name, self.kind = spec["tag"], spec["name"], kind_of(spec["mesh"])
        self.rng = rng
        self.d = 0.0            # degradation 0..1 (1 = functional failure)
        self.rate = 0.0         # degradation per sim hour
        self.mode = None
        self.running = True
        self.status = "OK"
        self.hist = deque(maxlen=240)   # (sim_hours, key indicator) for trend / RUL
        self.base = deque(maxlen=600)   # healthy baseline for anomaly z-score
        self.sensors = {}

    def n(self, s):
        return self.rng.gauss(0, s)

    def inject(self, mode, severity=0.3, hours_to_fail=None):
        self.mode = mode
        self.d = max(self.d, float(severity))
        self.rate = 1.0 / hours_to_fail if hours_to_fail else 1.0 / 24.0

    def reset(self):
        self.d, self.rate, self.mode, self.running = 0.0, 0.0, None, True
        self.hist.clear()

    def step(self, hours, t_h):
        if self.running:
            self.d = min(1.0, self.d + self.rate * hours * (0.6 + 0.8 * self.rng.random()))
        self.sensors = self.read(t_h)
        key = self.indicator()
        self.hist.append((t_h, key))
        if self.d < 0.05 and self.running:
            self.base.append(key)

    # -- overridden per kind
    def read(self, t_h):
        return {}

    def indicator(self):
        return 0.0

    def limits(self):
        return (1.0, 2.0, 3.0)  # warn, alarm, fail thresholds for indicator

    # -- analytics shared by all assets
    def anomaly(self):
        if len(self.base) < 30:
            return 0.0
        b = np.array(self.base)
        return float(abs(self.indicator() - b.mean()) / (b.std() + 1e-6))

    def rul_hours(self):
        """Linear trend of the key indicator extrapolated to the failure limit."""
        if len(self.hist) < 20 or not self.running:
            return None
        t, y = np.array(self.hist).T
        slope, icpt = np.polyfit(t - t[-1], y, 1)
        fail = self.limits()[2]
        if slope <= 1e-6:
            return None
        return float(max(0.0, (fail - icpt) / slope))

    def evaluate(self):
        if not self.running:
            return "TRIPPED" if self.d >= 1.0 else "STANDBY", 0.0
        w, a, f = self.limits()
        k = self.indicator()
        health = float(np.clip(100 * (1 - (k - self.limits_base()) / (f - self.limits_base())), 0, 100))
        status = "ALARM" if k >= a else "WARN" if k >= w else "OK"
        # hysteresis: only step down once the indicator is 8 % below the threshold
        rank = {"OK": 0, "WARN": 1, "ALARM": 2}
        if self.status in rank and rank[status] < rank[self.status]:
            thr = a if self.status == "ALARM" else w
            if k > thr * 0.92:
                status = self.status
        return status, health

    def limits_base(self):
        return self.base_value

    def frame(self):
        status, health = self.evaluate()
        self.status = status
        color, glow, pulse = STATUS_STYLE[status]
        rul = self.rul_hours() if (status in ("WARN", "ALARM") or self.anomaly() > 4) else None
        return {"kind": self.kind, "name": self.name, "status": status, "health": round(health, 1),
                "rul_h": None if rul is None or rul > 2000 else round(rul, 1),
                "anomaly": round(self.anomaly(), 2), "degradation": round(self.d, 3), "fault": self.mode,
                "color": color, "glow": glow, "pulse": pulse,
                "sensors": {k: round(v, 3) for k, v in self.sensors.items()}}


class Pump(Asset):
    """Centrifugal pump: bearing wear -> vibration (ISO 10816 zones), temperature, current."""
    base_value = 1.8

    def __init__(self, spec, rng):
        super().__init__(spec, rng)
        self.spare = spec["tag"].endswith("B")
        self.running = not self.spare
        self.flow_nom = {"P-101": 420.0, "P-102": 160.0, "P-103": 230.0}.get(spec["tag"][:5], 200.0)

    def read(self, t_h):
        if not self.running:
            return {"vibration_mm_s": abs(self.n(0.05)), "bearing_temp_c": 34 + self.n(0.3),
                    "motor_current_a": 0.0, "discharge_bar": 0.4 + self.n(0.02), "flow_m3h": 0.0, "speed_rpm": 0.0}
        d = self.d
        amb = 4 * math.sin(2 * math.pi * (t_h % 24) / 24)
        vib = self.base_value + 9.5 * d ** 2 + abs(self.n(0.12 + 0.6 * d))
        if self.mode == "cavitation":
            vib += 2.5 * d + abs(self.n(1.0 * d))
        return {
            "vibration_mm_s": vib,
            "bearing_temp_c": 62 + amb + 32 * d ** 1.5 + self.n(0.4),
            "motor_current_a": 182 + 16 * d + self.n(1.5),
            "discharge_bar": 18.2 - (2.5 if self.mode == "cavitation" else 1.2) * d + self.n(0.08),
            "flow_m3h": self.flow_nom * (1 - 0.08 * d) + self.n(2.0),
            "speed_rpm": 2975 + self.n(3),
        }

    def indicator(self):
        return self.sensors.get("vibration_mm_s", 0.0)

    def limits(self):
        return (4.5, 7.1, 11.2)  # ISO 10816-3 group 2 (rigid): B/C, C/D, failure


class Exchanger(Asset):
    """Shell & tube: fouling resistance grows -> duty and outlet temperature fall, dP rises."""
    base_value = 0.0

    def read(self, t_h):
        d = self.d
        return {"fouling_m2k_kw": 0.00035 * d / 0.6 + abs(self.n(0.00001)),
                "shell_dp_bar": 0.65 + 0.9 * d + self.n(0.01),
                "crude_out_c": 245 - 38 * d + self.n(0.6),
                "hot_in_c": 310 + self.n(0.8),
                "duty_mw": 14.5 * (1 - 0.35 * d) + self.n(0.08)}

    def indicator(self):
        return self.d * 3.0 + self.n(0.01)

    def limits(self):
        return (1.0, 1.8, 3.0)


class Furnace(Asset):
    base_value = 0.0

    def read(self, t_h):
        d = self.d
        return {"coil_outlet_c": 362 - 6 * d + self.n(0.5), "tube_skin_c": 470 + 85 * d + self.n(1.5),
                "stack_c": 185 + 40 * d + self.n(1.0), "o2_pct": 3.2 + self.n(0.08) - 1.2 * d,
                "fuel_gas_t_h": 6.1 + 0.9 * d + self.n(0.03)}

    def indicator(self):
        return (self.sensors.get("tube_skin_c", 470) - 470) / 30.0

    def limits(self):
        return (1.0, 2.0, 3.0)


class Column(Asset):
    base_value = 0.0

    def read(self, t_h):
        d = self.d
        return {"top_c": 118 + self.n(0.4) + 6 * d, "flash_zone_c": 355 + self.n(0.6),
                "top_bar": 1.45 + self.n(0.01) + 0.2 * d, "bottom_level_pct": 52 + self.n(0.8) + 25 * d,
                "tray_dp_mbar": 140 + 120 * d + self.n(2)}

    def indicator(self):
        return (self.sensors.get("tray_dp_mbar", 140) - 140) / 40.0

    def limits(self):
        return (1.0, 2.0, 3.0)


class Tank(Asset):
    base_value = 0.0

    def __init__(self, spec, rng):
        super().__init__(spec, rng)
        self.level = 65.0 if "Feed" in spec["name"] else 40.0
        self.dir = -1 if "Feed" in spec["name"] else 1

    def read(self, t_h):
        self.level = float(np.clip(self.level + self.dir * 0.002 + self.n(0.001), 5, 95))
        if self.level in (5, 95):
            self.dir *= -1
        return {"level_pct": self.level, "temp_c": 38 + self.n(0.2), "level_m": self.level / 100 * 13}

    def indicator(self):
        return 0.0 + self.d * 3

    def limits(self):
        return (1.0, 2.0, 3.0)


MODELS = {"pump": Pump, "exchanger": Exchanger, "furnace": Furnace, "column": Column, "tank": Tank}


# =========================================================================== plant / scenario

class Plant:
    def __init__(self, speed, seed=7):
        self.rng = random.Random(seed)
        self.speed = speed
        self.sim_time = dt.datetime.now().replace(microsecond=0)
        self.t_h = 0.0
        self.assets = {}
        for e in LAYOUT["equipment"]:
            if e.get("monitored"):
                k = kind_of(e["mesh"])
                self.assets[e["tag"]] = MODELS.get(k, Asset)(e, self.rng)
        self.events = deque(maxlen=50)
        self.pending = []
        # Demo scenario: hero pump P-101A has early bearing wear, fails in ~70 sim hours
        if "P-101A" in self.assets:
            self.assets["P-101A"].inject("bearing_wear", severity=0.05, hours_to_fail=70)
        self._csv = None

    def event(self, tag, level, msg):
        ev = {"ts": self.sim_time.isoformat(), "tag": tag, "level": level, "msg": msg}
        self.events.append(ev)
        self.pending.append(ev)
        print(f"  [{level:7}] {self.sim_time:%d %b %H:%M}  {tag:7} {msg}")

    def step(self, dt_real):
        hours = dt_real * self.speed / 3600.0
        self.t_h += hours
        self.sim_time += dt.timedelta(hours=hours)
        frames = {}
        for tag, a in self.assets.items():
            prev = a.status
            a.step(hours, self.t_h)
            f = a.frame()
            frames[tag] = f
            if f["status"] != prev:
                self.on_status_change(a, prev, f)
            if a.kind == "pump" and a.running and a.d >= 1.0:
                self.trip(a)
        self.log_csv(frames)
        return frames

    def on_status_change(self, a, prev, f):
        s = f["sensors"]
        if a.kind == "pump" and f["status"] in ("WARN", "ALARM"):
            self.event(a.tag, f["status"], f"Vibration {s['vibration_mm_s']:.1f} mm/s, bearing {s['bearing_temp_c']:.0f} C"
                                           + (f", RUL ~{f['rul_h']:.0f} h" if f["rul_h"] else ""))
        elif f["status"] in ("WARN", "ALARM"):
            self.event(a.tag, f["status"], f"{a.kind} condition degraded (degradation {a.d:.0%})")

    def trip(self, a):
        a.running = False
        a.status = "TRIPPED"
        self.event(a.tag, "TRIP", "Pump tripped on high vibration (ISO 10816 zone D)")
        spare = self.assets.get(a.tag[:-1] + "B")
        if spare and not spare.running:
            spare.running = True
            self.event(spare.tag, "INFO", f"Auto-start: spare took over from {a.tag}")

    def log_csv(self, frames):
        path = os.path.join(DATA_DIR, f"telemetry_{dt.date.today():%Y%m%d}.csv")
        new = not os.path.exists(path)
        if self._csv is None or self._csv.name != path:
            self._csv = open(path, "a", newline="")
            self._w = csv.writer(self._csv)
        if new:
            self._w.writerow(["sim_time", "tag", "status", "health", "degradation", "sensor", "value"])
        for tag, f in frames.items():
            for k, v in f["sensors"].items():
                self._w.writerow([self.sim_time.isoformat(), tag, f["status"], f["health"], f["degradation"], k, v])
        self._csv.flush()

    def command(self, msg):
        c = msg.get("cmd")
        a = self.assets.get(msg.get("tag", ""))
        if c == "inject_fault" and a:
            mode = msg.get("mode", {"pump": "bearing_wear", "exchanger": "fouling"}.get(a.kind, "degradation"))
            a.inject(mode, float(msg.get("severity", 0.3)), msg.get("hours_to_fail"))
            if not a.running and a.kind == "pump" and a.d < 1:
                a.running = True
            self.event(a.tag, "INFO", f"Fault injected: {mode} @ {a.d:.0%}")
        elif c == "reset" and a:
            a.reset()
            self.event(a.tag, "INFO", "Reset to healthy")
        elif c == "reset_all":
            for x in self.assets.values():
                x.reset()
                if isinstance(x, Pump):
                    x.running = not x.spare
            self.event("PLANT", "INFO", "All assets reset")
        elif c == "speed":
            self.speed = float(msg.get("value", self.speed))
            self.event("PLANT", "INFO", f"Simulation speed x{self.speed:g}")
        return {"type": "ack", "cmd": c, "ok": True}


# =========================================================================== server

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--speed", type=float, default=300.0, help="sim seconds per real second")
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--web", type=int, default=8080, help="serve the 3D web viewer on this port (0 = off)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if args.web:
        class Quiet(SimpleHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def end_headers(self):
                self.send_header("Cache-Control", "no-cache")
                super().end_headers()
        Quiet.extensions_map.update({".js": "text/javascript", ".glb": "model/gltf-binary"})
        httpd = ThreadingHTTPServer(("0.0.0.0", args.web), partial(Quiet, directory=ROOT))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://localhost:{args.web}/web/"
        print(f"3D web twin on {url}")
        if not args.no_browser:
            webbrowser.open(url)

    plant = Plant(args.speed)
    clients = set()
    last = {}

    async def handler(ws, *_):
        clients.add(ws)
        print(f"  client connected ({len(clients)})")
        try:
            if last:
                await ws.send(json.dumps(last))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if msg.get("cmd") == "snapshot":
                    await ws.send(json.dumps(last))
                else:
                    await ws.send(json.dumps(plant.command(msg)))
        finally:
            clients.discard(ws)
            print(f"  client left ({len(clients)})")

    async with serve(handler, args.host, args.port, compression=None):
        print(f"CDU-100 simulator on ws://localhost:{args.port}  speed x{plant.speed:g}  "
              f"({len(plant.assets)} assets, logging to data/)")
        period = 1.0 / args.hz
        tick = 0
        while True:
            frames = plant.step(period)
            last = {"type": "telemetry", "sim_time": plant.sim_time.isoformat(), "speed": plant.speed,
                    "assets": frames, "events": plant.pending[:]}
            plant.pending.clear()
            if clients:
                broadcast(clients, json.dumps(last))
            tick += 1
            if tick % int(args.hz * 10) == 0:
                p = frames.get("P-101A", {})
                print(f"{plant.sim_time:%d %b %H:%M}  P-101A {p.get('status')} health {p.get('health')}%  "
                      f"vib {p.get('sensors', {}).get('vibration_mm_s', 0):.2f} mm/s  RUL {p.get('rul_h')} h  "
                      f"clients {len(clients)}")
            await asyncio.sleep(period)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
