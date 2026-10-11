"""
Aggregator: turns the raw events of ONE session into the 8 session features
the GRU consumes.

    voice_command_latency_ms  mean latency_ms of voice events
    gesture_stability         mean stability_score of gesture events
    click_error_rate          failed screen actions / all screen actions
    correction_frequency      (corrected voice + retried gesture) / (voice + gesture events)
    modality_share_*          share of events from each source
    completion_accuracy       successful events / all events

If a modality was never used in a session we fall back to neutral defaults
(and the share is simply 0) so the model never receives NaN.
"""

from .store import FEATURE_COLUMNS, ContextStore

DEFAULTS = {"voice_command_latency_ms": 0.0, "gesture_stability": 0.0, "click_error_rate": 0.0}


def _mean(xs, default):
    return sum(xs) / len(xs) if xs else default


def aggregate_session(events):
    if not events:
        raise ValueError("cannot aggregate an empty session")
    voice = [e["payload"] for e in events if e["source"] == "voice"]
    gesture = [e["payload"] for e in events if e["source"] == "gesture"]
    screen = [e["payload"] for e in events if e["source"] == "screen"]
    n = len(events)

    successes = (sum(p["success"] for p in voice) + sum(p["success"] for p in gesture)
                 + sum(p["click_success"] for p in screen))
    vg = len(voice) + len(gesture)
    corrections = sum(p["was_corrected"] for p in voice) + sum(p["was_retried"] for p in gesture)

    return {
        "click_error_rate": (sum(not p["click_success"] for p in screen) / len(screen)) if screen else 0.0,
        "voice_command_latency_ms": _mean([p["latency_ms"] for p in voice], DEFAULTS["voice_command_latency_ms"]),
        "correction_frequency": (corrections / vg) if vg else 0.0,
        "gesture_stability": _mean([p["stability_score"] for p in gesture], DEFAULTS["gesture_stability"]),
        "modality_share_voice": len(voice) / n,
        "modality_share_gesture": len(gesture) / n,
        "modality_share_screen": len(screen) / n,
        "completion_accuracy": successes / n,
    }


def aggregate_user(store: ContextStore, user_id, labels=None):
    """Aggregate every session of a user into the sessions table."""
    out = []
    for idx, (session_id, start_ts) in enumerate(store.list_session_ids(user_id)):
        events = store.get_events(user_id, session_id)
        feats = aggregate_session(events)
        label = labels[idx] if labels is not None and idx < len(labels) else None
        store.upsert_session(user_id, session_id, idx, start_ts, len(events), feats, label)
        out.append(feats)
    return out


def aggregate_all(store: ContextStore, labels_by_user=None):
    for uid in store.list_users():
        lab = labels_by_user[uid]["labels"] if labels_by_user and uid in labels_by_user else None
        aggregate_user(store, uid, lab)
    return store.count("sessions")


assert len(FEATURE_COLUMNS) == 8
