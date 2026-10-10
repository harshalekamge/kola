"""
Event schema (the data contract between Components 1-3 and Component 4).

Every component sends ONE JSON object per user action ("event"):

    {
      "schema_version": "1.0",
      "user_id": "u_3f9a1c2b",          # pseudonymous hash, never a real name
      "session_id": "S-u_3f9a1c2b-004",  # one app-usage period
      "timestamp": "2026-10-12T09:15:03.120Z",
      "source": "voice" | "gesture" | "screen",
      "payload": { ...fields depend on source... }
    }
"""

import hashlib
from datetime import datetime

SCHEMA_VERSION = "1.0"
SOURCES = ("voice", "gesture", "screen")

# field name -> (accepted python types, optional range/choices)
PAYLOAD_FIELDS = {
    "voice": {
        "latency_ms": ((int, float), (0, 60000)),
        "asr_confidence": ((int, float), (0.0, 1.0)),
        "language": (str, ("si", "en", "singlish")),
        "intent": (str, None),
        "was_corrected": (bool, None),
        "success": (bool, None),
    },
    "gesture": {
        "gesture_name": (str, None),
        "confidence": ((int, float), (0.0, 1.0)),
        "stability_score": ((int, float), (0.0, 1.0)),
        "was_retried": (bool, None),
        "success": (bool, None),
    },
    "screen": {
        "action": (str, None),
        "element_found": (bool, None),
        "ocr_confidence": ((int, float), (0.0, 1.0)),
        "click_success": (bool, None),
        "retries": (int, (0, 20)),
    },
}


def hash_user_id(raw_id: str) -> str:
    """Pseudonymise a raw user identifier (privacy by design)."""
    return "u_" + hashlib.sha256(raw_id.encode()).hexdigest()[:8]


def parse_timestamp(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def validate_event(event: dict) -> dict:
    """Raise ValueError if the event breaks the contract; return it if fine."""
    if not isinstance(event, dict):
        raise ValueError("event must be a JSON object")
    for key in ("schema_version", "user_id", "session_id", "timestamp", "source", "payload"):
        if key not in event:
            raise ValueError(f"missing field: {key}")
    if event["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {event['schema_version']}")
    if event["source"] not in SOURCES:
        raise ValueError(f"unknown source: {event['source']}")
    try:
        parse_timestamp(event["timestamp"])
    except (ValueError, TypeError, AttributeError):
        raise ValueError("timestamp must be ISO-8601")
    payload = event["payload"]
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    for name, (types, rule) in PAYLOAD_FIELDS[event["source"]].items():
        if name not in payload:
            raise ValueError(f"payload missing field: {name}")
        value = payload[name]
        if types is not bool and isinstance(value, bool):
            raise ValueError(f"{name} must not be a boolean")
        if not isinstance(value, types):
            raise ValueError(f"{name} has wrong type")
        if isinstance(rule, tuple) and len(rule) == 2 and not isinstance(rule[0], str):
            if not (rule[0] <= value <= rule[1]):
                raise ValueError(f"{name} out of range {rule}")
        elif isinstance(rule, tuple) and value not in rule:
            raise ValueError(f"{name} must be one of {rule}")
    return event
