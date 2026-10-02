"""
Twin Live Link - connects the open L_Refinery level to the simulator WebSocket.
Run in the Unreal Editor:  Output Log -> Cmd (Python) ->
    py "C:/Users/Arghya Chakraborty/Desktop/DigitalTwin/unreal/twin_live_link.py"
Run again to restart; run twin_live_link_stop.py to stop. Works in the editor viewport and during Play-In-Editor (it follows the active world).

For every actor tagged "TwinAsset" + <TAG>:
  * dynamic material instances drive M_TwinMaster HealthColor / HealthGlow / PulseSpeed
  * the floating TextRender label shows  TAG / health % / status / key reading
Events (WARN/ALARM/TRIP) are printed to the Output Log.
"""
import builtins
import os
import sys
import time

import unreal

HOST, PORT = "127.0.0.1", 8765
APPLY_EVERY = 0.25  # seconds

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import importlib  # noqa: E402

import ws_client  # noqa: E402

importlib.reload(ws_client)

KEY_READING = {"pump": ("vibration_mm_s", "mm/s"), "exchanger": ("crude_out_c", "C"),
               "furnace": ("tube_skin_c", "C skin"), "column": ("tray_dp_mbar", "mbar dP"),
               "tank": ("level_pct", "% level")}


def _world():
    try:
        pie = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
        if pie:
            return pie
        return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    except Exception:
        return unreal.EditorLevelLibrary.get_editor_world()


class LiveLink:
    def __init__(self):
        self.client = None
        self.world = None
        self.bind = {}       # tag -> [MID,...]
        self.labels = {}     # tag -> TextRenderComponent
        self.latest = None
        self.last_apply = 0.0
        self.last_try = 0.0
        self.handle = unreal.register_slate_post_tick_callback(self.tick)
        unreal.log("[TwinLink] started - waiting for simulator on ws://%s:%d" % (HOST, PORT))

    # ------------------------------------------------------------------ binding
    def rebind(self, world):
        self.world, self.bind, self.labels = world, {}, {}
        actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
        for a in actors:
            tags = [str(t) for t in a.tags]
            if "TwinAsset" in tags and len(tags) > 1:
                tag = tags[1]
                comp = a.get_component_by_class(unreal.StaticMeshComponent)
                if not comp:
                    continue
                mids = []
                for i in range(comp.get_num_materials()):
                    m = comp.get_material(i)
                    if isinstance(m, unreal.MaterialInstanceDynamic):
                        mids.append(m)
                    elif m:
                        mids.append(comp.create_dynamic_material_instance(i, m))
                self.bind[tag] = mids
            elif "TwinLabel" in tags and len(tags) > 1:
                c = a.get_component_by_class(unreal.TextRenderComponent)
                if c:
                    self.labels[tags[1]] = c
        unreal.log(f"[TwinLink] bound {len(self.bind)} assets, {len(self.labels)} labels in {world.get_name()}")

    # ------------------------------------------------------------------ tick
    def tick(self, dt):
        now = time.time()
        if self.client is None or self.client.closed:
            if now - self.last_try > 3.0:
                self.last_try = now
                try:
                    self.client = ws_client.WSClient(HOST, PORT, timeout=0.5)
                    unreal.log("[TwinLink] connected to simulator")
                except OSError:
                    self.client = None
            return
        for m in self.client.poll():
            if isinstance(m, dict) and m.get("type") == "telemetry":
                self.latest = m
                for ev in m.get("events", []):
                    fn = unreal.log_warning if ev["level"] in ("WARN", "ALARM", "TRIP") else unreal.log
                    fn(f"[TwinLink] {ev['ts'][5:16]} {ev['tag']} {ev['level']}: {ev['msg']}")
        if self.latest and now - self.last_apply >= APPLY_EVERY:
            self.last_apply = now
            w = _world()
            if w is None:
                return
            if w != self.world:
                self.rebind(w)
            self.apply(self.latest)

    def apply(self, msg):
        for tag, f in msg["assets"].items():
            r, g, b = f["color"]
            col = unreal.LinearColor(r, g, b, 1.0)
            for mid in self.bind.get(tag, []):
                try:
                    mid.set_vector_parameter_value("HealthColor", col)
                    mid.set_scalar_parameter_value("HealthGlow", float(f["glow"]))
                    mid.set_scalar_parameter_value("PulseSpeed", float(f["pulse"]))
                except Exception:
                    pass
            lab = self.labels.get(tag)
            if lab:
                key, unit = KEY_READING.get(f["kind"], (None, ""))
                val = f["sensors"].get(key) if key else None
                line2 = f"{f['health']:.0f}%  {f['status']}"
                if val is not None:
                    line2 += f"  |  {val:.1f} {unit}"
                if f.get("rul_h") and f["status"] in ("WARN", "ALARM"):
                    line2 += f"  |  RUL {f['rul_h']:.0f} h"
                lab.set_text(f"{tag}\n{line2}")
                lab.set_editor_property("text_render_color",
                                        unreal.Color(r=int(r * 255), g=int(g * 255), b=int(b * 255), a=255))

    def stop(self):
        unreal.unregister_slate_post_tick_callback(self.handle)
        if self.client:
            self.client.close()
        for mids in self.bind.values():
            for mid in mids:
                try:
                    mid.set_scalar_parameter_value("HealthGlow", 0.0)
                except Exception:
                    pass
        unreal.log("[TwinLink] stopped")


# toggle: running the script again stops the previous link and starts a fresh one
prev = getattr(builtins, "_twin_live_link", None)
if prev is not None:
    prev.stop()
    builtins._twin_live_link = None
builtins._twin_live_link = LiveLink()
