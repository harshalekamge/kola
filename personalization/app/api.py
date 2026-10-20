"""
Local HTTP API for Components 1-3 and the AI Assistant Engine.

Run:  uvicorn app.api:create_app --factory --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs   (auto-generated, good for the demo)

    POST /events            body = one event OR a list of events (see docs/INTERFACE_CONTRACT.md)
    GET  /signal/{user_id}  personalization signal for that user
    GET  /health
Everything stays on the local machine: no cloud.
"""

import threading

from fastapi import Body, FastAPI, HTTPException

from app.schema import validate_event
from app.service import PersonalizationService
from app.store import ContextStore


def create_app(db_path="context_store.db", model_dir="models"):
    app = FastAPI(title="Component 4 - Personalization Module", version="1.0")
    store = ContextStore(db_path)
    service = PersonalizationService(store, model_dir)
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "events": store.count("events")}

    @app.post("/events")
    def post_events(payload=Body(...)):
        events = payload if isinstance(payload, list) else [payload]
        for i, e in enumerate(events):
            try:
                validate_event(e)
            except ValueError as err:
                raise HTTPException(status_code=422, detail=f"event {i}: {err}")
        with lock:
            n = store.add_events(events)
        return {"accepted": n}

    @app.get("/signal/{user_id}")
    def get_signal(user_id: str):
        with lock:
            if user_id not in store.list_users():
                raise HTTPException(status_code=404, detail="unknown user_id")
            return service.signal_for(user_id)

    return app
