"""
CDU-100 Digital Twin — FastAPI application
==========================================
    python -m uvicorn app.main:app --port 8000        (or double-click START.bat)
    open http://localhost:8000

Serves:
  /                      web UI (3D twin + asset panel + AI assistant)
  /ws                    live telemetry (WebSocket, JSON ~2 Hz) and scenario commands
  /api/assets            all assets, latest frame          /api/assets/{tag}   one asset
  /api/assets/{tag}/history?sensor=vibration_mm_s           time series for charts
  /api/events            recent events                     /api/command         scenario commands
  /api/chat              LangGraph assistant               /api/llm             which LLM is active
  /docs                  interactive API documentation (Swagger)
"""
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from . import agent as agent_mod  # noqa: E402
from .twin import LAYOUT, Twin  # noqa: E402

twin = Twin(speed=float(os.getenv("SIM_SPEED", "300")))
agent_mod.TWIN = twin
assistant = agent_mod.Assistant()


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(twin.run())
    await asyncio.to_thread(assistant.init)
    print(f"\n  CDU-100 Digital Twin ready ->  http://localhost:{os.getenv('PORT', '8000')}"
          f"\n  Assistant: {assistant.desc}\n")
    yield
    task.cancel()


app = FastAPI(title="CDU-100 Digital Twin", version="0.2", lifespan=lifespan)


class ChatIn(BaseModel):
    message: str
    session_id: str = "default"


class CommandIn(BaseModel):
    cmd: str
    tag: str | None = None
    mode: str | None = None
    severity: float | None = None
    hours_to_fail: float | None = None
    value: float | None = None


# --------------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return {"ok": True, "sim_time": twin.latest.get("sim_time"), "clients": len(twin.clients), "llm": assistant.desc}


@app.get("/api/llm")
def llm():
    return {"llm": assistant.desc, "agent": assistant.graph is not None}


@app.get("/api/layout")
def layout():
    return LAYOUT


@app.get("/api/assets")
def assets():
    return twin.latest.get("assets", {})


@app.get("/api/assets/{tag}")
def asset(tag: str):
    t, f = twin.asset(tag)
    if not f:
        raise HTTPException(404, f"unknown asset {tag}")
    return {"tag": t, **f}


@app.get("/api/assets/{tag}/history")
def history(tag: str, sensor: str, points: int = 300):
    if not twin.norm_tag(tag):
        raise HTTPException(404, f"unknown asset {tag}")
    return {"tag": twin.norm_tag(tag), "sensor": sensor, "now_h": twin.plant.t_h, "points": twin.series(tag, sensor, points)}


@app.get("/api/events")
def events(limit: int = 30):
    return list(twin.plant.events)[-limit:][::-1]


@app.post("/api/command")
def command(c: CommandIn):
    return twin.command({k: v for k, v in c.model_dump().items() if v is not None})


@app.post("/api/chat")
async def chat(c: ChatIn):
    return await assistant.chat(c.message.strip()[:2000], c.session_id)


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    twin.clients.add(sock)
    try:
        if twin.latest:
            await sock.send_json(twin.latest)
        while True:
            msg = await sock.receive_json()
            if isinstance(msg, dict) and msg.get("cmd"):
                await sock.send_json(twin.command(msg))
    except WebSocketDisconnect:
        pass
    finally:
        twin.clients.discard(sock)


# --------------------------------------------------------------------------- static (3D assets + UI)
app.mount("/content", StaticFiles(directory=os.path.join(ROOT, "Content_Source")), name="content")
app.mount("/config", StaticFiles(directory=os.path.join(ROOT, "config")), name="config")
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "web")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"), headers={"Cache-Control": "no-cache"})
