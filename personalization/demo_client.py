"""
Live demo client: plays the role of Components 1-3 (run:  python demo_client.py)

1. Start the API in another terminal:
       uvicorn app.api:create_app --factory --port 8000
2. Run this script. It sends a simulated 'gradual decline' user's sessions one
   by one and prints the personalization signal after each session.
"""

import json
import sys
import urllib.request

from app.simulator import simulate_user

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


events, labels = simulate_user("demo-live-user", 16, "gradual", seed=11)
uid = events[0]["user_id"]
by_session = {}
for e in events:
    by_session.setdefault(e["session_id"], []).append(e)

print(f"user {uid}: sending {len(by_session)} sessions\n")
for idx, (sid, evs) in enumerate(sorted(by_session.items())):
    call("POST", "/events", evs)
    out = call("GET", f"/signal/{uid}")
    s = out["signal"]
    note = "" if out["status"] == "ok" else f" ({out['status']}: {out['sessions_seen']}/{out['sessions_needed']})"
    prob = out.get("raw_probability", "-")
    print(f"session {idx:2d} | true decline={labels[idx]} | P={prob!s:>5} | gesture x{s['gesture_tolerance_multiplier']}"
          f" | speech x{s['speech_timeout_multiplier']} | UI {s['ui_shortcut_mode']}{note}")
