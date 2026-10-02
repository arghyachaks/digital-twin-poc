"""
Control the running simulator (also the tool surface the LangGraph agent will call).

  python backend/simctl.py status                 # one snapshot, all assets
  python backend/simctl.py fault P-103A --severity 0.5 --hours 6
  python backend/simctl.py fault E-102 --mode fouling --severity 0.6
  python backend/simctl.py reset P-101A | reset-all
  python backend/simctl.py speed 1200
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_client import WSClient  # noqa: E402


def wait_for(c, kind, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        for m in c.poll():
            if isinstance(m, dict) and m.get("type") == kind:
                return m
        time.sleep(0.05)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["status", "fault", "reset", "reset-all", "speed"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--mode")
    ap.add_argument("--severity", type=float, default=0.3)
    ap.add_argument("--hours", type=float)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    c = WSClient(port=a.port)
    if a.action == "status":
        c.send({"cmd": "snapshot"})
        m = wait_for(c, "telemetry")
        print(f"sim time {m['sim_time']}  speed x{m['speed']:g}")
        for tag, f in m["assets"].items():
            rul = f"RUL {f['rul_h']:.0f}h" if f["rul_h"] else ""
            print(f"  {tag:7} {f['status']:8} {f['health']:5.1f}%  anomaly {f['anomaly']:5.2f}  {rul}")
    else:
        msg = {"fault": {"cmd": "inject_fault", "tag": a.arg, "mode": a.mode, "severity": a.severity,
                         "hours_to_fail": a.hours},
               "reset": {"cmd": "reset", "tag": a.arg}, "reset-all": {"cmd": "reset_all"},
               "speed": {"cmd": "speed", "value": float(a.arg or 300)}}[a.action]
        c.send({k: v for k, v in msg.items() if v is not None})
        print(wait_for(c, "ack"))
    c.close()


if __name__ == "__main__":
    main()
