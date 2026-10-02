"""
Plant assistant: a LangGraph ReAct agent whose tools read the live twin.
The same tool functions power an offline (no-LLM) mode so the chat always works.
Tools can also return UI actions (e.g. fly the 3D camera to an asset) that the browser executes.
"""
import contextvars
import re

from .twin import UNITS

TWIN = None                                   # set by main.py
_actions = contextvars.ContextVar("ui_actions", default=None)

FAULT_HINTS = {
    "pump": {
        "bearing_wear": "Rising vibration together with rising bearing temperature is the classic bearing-wear signature. "
                        "Plan a bearing replacement; check lubrication and alignment; switch to the spare pump before zone D.",
        "cavitation": "Noisy vibration with falling discharge pressure points to cavitation. Check suction strainer, "
                      "suction pressure / NPSH margin and tank level.",
    },
    "exchanger": {"fouling": "Falling crude outlet temperature with rising shell-side pressure drop indicates fouling. "
                             "Schedule cleaning; meanwhile the fired heater must add the lost duty (more fuel)."},
    "furnace": {"degradation": "Rising tube-skin and stack temperature suggests coking or poor heat transfer. "
                               "Watch tube-skin limits; consider reducing firing or decoking."},
    "column": {"degradation": "Rising tray pressure drop and bottom level suggests flooding/fouled trays. Reduce feed or "
                              "check reflux and bottoms pump."},
}


def _act(**a):
    lst = _actions.get()
    if lst is not None:
        lst.append(a)


def _fmt(k, v):
    return f"{v:.3g} {UNITS.get(k, '')}".strip()


# =========================================================================== tool functions

def get_plant_overview() -> str:
    """Summary of the whole CDU-100 unit: simulated time, counts per status, and every asset that is not healthy
    (with health %, remaining useful life and key reading). Use this first for general questions."""
    m = TWIN.latest
    if not m:
        return "Telemetry not available yet."
    counts, issues = {}, []
    for tag, f in m["assets"].items():
        counts[f["status"]] = counts.get(f["status"], 0) + 1
        if f["status"] not in ("OK", "STANDBY"):
            rul = f", RUL ~{f['rul_h']:.0f} h" if f.get("rul_h") else ""
            issues.append(f"- {tag} ({f['name']}): {f['status']}, health {f['health']:.0f}%{rul}, anomaly {f['anomaly']:.1f}σ")
    lines = [f"Sim time {m['sim_time'][:16]}, speed x{m['speed']:g}. Status counts: "
             + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
    lines += issues or ["All running assets are healthy."]
    return "\n".join(lines)


def get_asset_status(tag: str) -> str:
    """Live status of one asset (e.g. 'P-101A', 'E-102', 'C-101', 'H-101', 'T-101'): status, health %, remaining
    useful life, anomaly score and every current sensor reading with units."""
    t, f = TWIN.asset(tag)
    if not f:
        return f"Unknown asset '{tag}'. Known tags: {', '.join(TWIN.tags)}"
    rul = f"{f['rul_h']:.0f} h" if f.get("rul_h") else "n/a"
    s = "\n".join(f"  - {k}: {_fmt(k, v)}" for k, v in f["sensors"].items())
    return (f"{t} — {f['name']} ({f['kind']})\nstatus {f['status']}, health {f['health']:.0f}%, RUL {rul}, "
            f"anomaly {f['anomaly']:.1f}σ, active fault: {f.get('fault') or 'none'}\nsensors:\n{s}")


def get_sensor_trend(tag: str, sensor: str = "", hours: float = 12.0) -> str:
    """Trend statistics over the last N simulated hours for one asset: first/last/min/max/mean and change per hour.
    Leave sensor empty for all sensors. Sensor names look like 'vibration_mm_s', 'bearing_temp_c', 'crude_out_c'."""
    t = TWIN.norm_tag(tag)
    if not t:
        return f"Unknown asset '{tag}'."
    tr = TWIN.trend(t, sensor or None, float(hours))
    if not tr:
        return f"Not enough history for {t} yet."
    _act(type="chart", tag=t, sensor=sensor or next(iter(tr)))
    return f"{t} trend over ~{hours:g} h:\n" + "\n".join(
        f"  - {k}: {d['first']} → {d['last']} {d['unit']} (min {d['min']}, max {d['max']}, "
        f"{d['change_per_hour']:+.3g}/h)" for k, d in tr.items())


def diagnose_asset(tag: str) -> str:
    """Root-cause style diagnosis for one asset: compares current sensors with its healthy baseline, lists the
    biggest deviations, the likely failure mode and the recommended maintenance action."""
    t, f = TWIN.asset(tag)
    if not f:
        return f"Unknown asset '{tag}'."
    base = TWIN.baseline(t)
    devs = []
    for k, v in f["sensors"].items():
        b = base.get(k)
        if b is None or abs(b) < 1e-9:
            continue
        devs.append((abs(v - b) / abs(b), k, b, v))
    devs.sort(reverse=True)
    lines = [f"{t}: {f['status']}, health {f['health']:.0f}%"
             + (f", RUL ~{f['rul_h']:.0f} h" if f.get("rul_h") else "")]
    if devs:
        lines.append("Largest deviations from healthy baseline:")
        lines += [f"  - {k}: {_fmt(k, b)} → {_fmt(k, v)} ({(v - b) / b:+.0%})" for _, k, b, v in devs[:4]]
    mode = f.get("fault")
    if not mode and f["kind"] == "pump" and devs and devs[0][1] in ("vibration_mm_s", "bearing_temp_c") and devs[0][0] > 0.3:
        mode = "bearing_wear"
    significant = f["status"] not in ("OK", "STANDBY") or (devs and devs[0][0] > 0.15)
    hint = FAULT_HINTS.get(f["kind"], {}).get(mode or "", "") if significant else ""
    if hint:
        lines.append(f"Likely failure mode: {mode.replace('_', ' ')}. {hint}")
    elif f["status"] == "OK":
        lines.append("No significant deviation — asset is operating normally.")
    _act(type="focus", tag=t)
    return "\n".join(lines)


def list_recent_events(limit: int = 10) -> str:
    """Most recent plant events and alarms (WARN / ALARM / TRIP / INFO), newest first."""
    ev = list(TWIN.plant.events)[-int(limit):][::-1]
    if not ev:
        return "No events yet."
    return "\n".join(f"- {e['ts'][5:16]} {e['level']} {e['tag']}: {e['msg']}" for e in ev)


def focus_asset(tag: str) -> str:
    """Fly the 3D digital-twin camera to an asset and open its detail panel. Call this whenever the conversation
    is about a specific asset so the operator can see it."""
    t = TWIN.norm_tag(tag)
    if not t:
        return f"Unknown asset '{tag}'."
    _act(type="focus", tag=t)
    return f"Camera moved to {t}."


def simulate_fault(tag: str, mode: str = "", severity: float = 0.4, hours_to_fail: float = 12.0) -> str:
    """SCENARIO SANDBOX ONLY: inject a simulated fault into the twin (never the real plant). Only use when the user
    explicitly asks for a what-if / simulation. Modes: pumps 'bearing_wear' or 'cavitation'; exchangers 'fouling';
    furnace/column 'degradation'. severity 0-1."""
    t = TWIN.norm_tag(tag)
    if not t:
        return f"Unknown asset '{tag}'."
    msg = {"cmd": "inject_fault", "tag": t, "severity": float(severity), "hours_to_fail": float(hours_to_fail)}
    if mode:
        msg["mode"] = mode
    TWIN.command(msg)
    _act(type="focus", tag=t)
    return f"Simulated {mode or 'default'} fault injected on {t} at severity {severity:.0%} (twin only)."


def reset_asset(tag: str) -> str:
    """SCENARIO SANDBOX ONLY: reset a simulated asset to healthy. Use 'ALL' to reset the whole unit."""
    if str(tag).upper() == "ALL":
        TWIN.command({"cmd": "reset_all"})
        return "All simulated assets reset to healthy."
    t = TWIN.norm_tag(tag)
    if not t:
        return f"Unknown asset '{tag}'."
    TWIN.command({"cmd": "reset", "tag": t})
    return f"{t} reset to healthy (simulation)."


def set_simulation_speed(multiplier: float) -> str:
    """Change how fast simulated time runs (e.g. 60, 300, 1200, 3600 sim-seconds per real second)."""
    TWIN.command({"cmd": "speed", "value": float(multiplier)})
    return f"Simulation speed set to x{float(multiplier):g}."


TOOLS = [get_plant_overview, get_asset_status, get_sensor_trend, diagnose_asset, list_recent_events,
         focus_asset, simulate_fault, reset_asset, set_simulation_speed]

SYSTEM = """You are the CDU-100 plant assistant inside a refinery digital twin (fictional crude distillation unit).
Equipment: C-101 atmospheric column, H-101 fired heater, E-101..E-104 heat exchangers,
pumps P-101A/B (crude charge), P-102A/B (reflux), P-103A/B (bottoms) — the B pumps are spares —
and tanks T-101 (crude feed), T-102 (diesel).
Rules:
- Every number you state must come from a tool call in this turn. Never invent readings.
- When you talk about a specific asset, call focus_asset (or diagnose_asset) so the 3D view shows it.
- simulate_fault / reset_asset / set_simulation_speed change the SIMULATION only; use them only when the user asks.
- Answer like a reliability engineer: short, concrete, prioritised. Use bullet points for lists. Mention ISO 10816
  vibration zones for pumps (B/C at 4.5 mm/s, C/D at 7.1 mm/s) when relevant, and give a recommended action."""


# =========================================================================== agent

class Assistant:
    def __init__(self):
        self.graph = None
        self.desc = "starting"
        self.error = None

    def init(self):
        from .llm import make_llm
        llm, self.desc = make_llm()
        if llm is None:
            return
        try:
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            tools = [tool(fn) for fn in TOOLS]
            try:
                self.graph = create_react_agent(llm, tools, prompt=SYSTEM, checkpointer=MemorySaver())
            except TypeError:  # older langgraph
                self.graph = create_react_agent(llm, tools, state_modifier=SYSTEM, checkpointer=MemorySaver())
        except Exception as e:  # missing packages etc.
            self.graph, self.error = None, str(e)
            self.desc += f" (agent unavailable: {e})"

    async def chat(self, text, session="default"):
        acts = []
        token = _actions.set(acts)
        used = []
        try:
            if self.graph is None:
                reply = offline_answer(text)
                used = ["offline"]
            else:
                try:
                    res = await self.graph.ainvoke({"messages": [("user", text)]},
                                                   config={"configurable": {"thread_id": session},
                                                           "recursion_limit": 14})
                    msgs = res["messages"]
                    # tools used in this turn = tool messages after the last human message
                    last_h = max(i for i, m in enumerate(msgs) if m.type == "human")
                    used = [m.name for m in msgs[last_h:] if m.type == "tool"]
                    reply = _text(msgs[-1].content)
                except Exception as e:
                    reply = (f"_The language model failed ({type(e).__name__}: {str(e)[:160]}). "
                             f"Answering from plant data instead:_\n\n" + offline_answer(text))
                    used = ["offline"]
        finally:
            _actions.reset(token)
        return {"reply": reply, "tools": used, "actions": acts, "llm": self.desc}


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def offline_answer(text):
    """Deterministic answers from the same tools when no LLM is available."""
    q = text.lower()
    tags = [t for t in TWIN.tags if re.sub(r"[^a-z0-9]", "", t.lower()) in re.sub(r"[^a-z0-9]", "", q)]
    parts = []
    if tags:
        for t in tags[:2]:
            parts.append(diagnose_asset(t))
            if any(w in q for w in ("trend", "history", "last", "over")):
                parts.append(get_sensor_trend(t, "", 24))
            else:
                parts.append(get_asset_status(t))
    elif any(w in q for w in ("event", "alarm", "log", "happened")):
        parts.append(list_recent_events(10))
    else:
        parts.append(get_plant_overview())
    return "\n\n".join(parts)
