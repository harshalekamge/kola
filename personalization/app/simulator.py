"""
Event simulator: stands in for Components 1-3 until they exist.

It emits RAW events (the contract in schema.py), not ready-made features,
so the real aggregation code is exercised. Users have different baselines
and follow different trajectories, which keeps the learning task honest:

    stable     - no decline, small random noise
    gradual    - noisy gradual decline starting partway through
    recovering - declines, then recovers (should stop being flagged)
    blip       - ONE bad session then back to normal (must NOT be flagged)

Ground-truth labels (severity > 0.5 => decline) are returned separately and
are never put inside the events.
"""

import random
from datetime import datetime, timedelta, timezone

from .schema import SCHEMA_VERSION, hash_user_id, validate_event

PROFILES = ("stable", "gradual", "recovering", "blip")
LANGS = ("si", "en", "singlish")
INTENTS = ("open_app", "scroll_down", "click_button", "read_text", "go_back")
GESTURES = ("point", "pinch", "swipe", "open_palm")
ACTIONS = ("click", "type", "scroll", "select")


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def severity_curve(profile: str, n_sessions: int, rng: random.Random):
    """Hidden 'how bad is the user's state' per session, in [0, 1]."""
    sev = [0.0] * n_sessions
    if profile == "gradual":
        onset = int(n_sessions * rng.uniform(0.35, 0.6))
        for i in range(onset, n_sessions):
            sev[i] = _clip((i - onset + 1) / max(1, n_sessions - onset) * rng.uniform(0.9, 1.2))
    elif profile == "recovering":
        start = int(n_sessions * rng.uniform(0.25, 0.4))
        peak = start + int(n_sessions * 0.25)
        end = peak + int(n_sessions * 0.25)
        for i in range(start, min(end, n_sessions)):
            if i <= peak:
                sev[i] = _clip((i - start + 1) / max(1, peak - start + 1))
            else:
                sev[i] = _clip(1 - (i - peak) / max(1, end - peak))
    elif profile == "blip":
        sev[rng.randrange(2, max(3, n_sessions - 2))] = rng.uniform(0.8, 1.0)
    return sev


def simulate_session(user_id, session_index, severity, base, rng, start):
    """Return a list of raw events for one session."""
    session_id = f"S-{user_id}-{session_index:03d}"
    n_events = rng.randint(25, 45)
    # users lean on voice more and gesture less as they struggle
    w_voice = base["w_voice"] + 0.15 * severity
    w_gesture = max(0.05, base["w_gesture"] - 0.20 * severity)
    w_screen = max(0.05, 1 - w_voice - w_gesture)
    events, t = [], start
    for _ in range(n_events):
        t += timedelta(seconds=rng.uniform(2, 12))
        src = rng.choices(["voice", "gesture", "screen"], [w_voice, w_gesture, w_screen])[0]
        if src == "voice":
            p_fail = _clip(base["voice_fail"] + 0.30 * severity + rng.gauss(0, 0.03))
            ok = rng.random() > p_fail
            payload = {
                "latency_ms": round(max(100, base["latency"] * (1 + 0.8 * severity) + rng.gauss(0, 120)), 1),
                "asr_confidence": round(_clip(0.93 - 0.25 * severity + rng.gauss(0, 0.05)), 3),
                "language": rng.choice(LANGS),
                "intent": rng.choice(INTENTS),
                "was_corrected": rng.random() < _clip(base["corr"] + 0.30 * severity),
                "success": ok,
            }
        elif src == "gesture":
            p_fail = _clip(base["gesture_fail"] + 0.30 * severity + rng.gauss(0, 0.03))
            payload = {
                "gesture_name": rng.choice(GESTURES),
                "confidence": round(_clip(0.90 - 0.25 * severity + rng.gauss(0, 0.06)), 3),
                "stability_score": round(_clip(base["stability"] - 0.45 * severity + rng.gauss(0, 0.07)), 3),
                "was_retried": rng.random() < _clip(base["corr"] + 0.30 * severity),
                "success": rng.random() > p_fail,
            }
        else:
            p_fail = _clip(base["click_fail"] + 0.30 * severity + rng.gauss(0, 0.03))
            ok = rng.random() > p_fail
            payload = {
                "action": rng.choice(ACTIONS),
                "element_found": rng.random() > 0.03,
                "ocr_confidence": round(_clip(0.92 - 0.15 * severity + rng.gauss(0, 0.05)), 3),
                "click_success": ok,
                "retries": 0 if ok else rng.randint(1, 3),
            }
        events.append({
            "schema_version": SCHEMA_VERSION,
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z",
            "source": src,
            "payload": payload,
        })
    return events


def simulate_user(raw_id, n_sessions=20, profile="stable", seed=0, start=None):
    """Return (events, labels). labels[i] = 1 if session i is truly 'declined'."""
    rng = random.Random(seed)
    user_id = hash_user_id(raw_id)
    # each user has their own baseline so one global threshold cannot fit everyone
    w_voice = rng.uniform(0.25, 0.45)
    w_gesture = rng.uniform(0.2, 0.4)
    base = {
        "w_voice": w_voice, "w_gesture": w_gesture,
        "latency": rng.uniform(600, 1400),
        "stability": rng.uniform(0.6, 0.92),
        "voice_fail": rng.uniform(0.04, 0.15),
        "gesture_fail": rng.uniform(0.05, 0.18),
        "click_fail": rng.uniform(0.03, 0.12),
        "corr": rng.uniform(0.05, 0.15),
    }
    sev = severity_curve(profile, n_sessions, rng)
    start = start or datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    events, labels = [], []
    for i in range(n_sessions):
        session_start = start + timedelta(days=i, minutes=rng.randint(0, 120))
        evs = simulate_session(user_id, i, sev[i], base, rng, session_start)
        for e in evs:
            validate_event(e)
        events.extend(evs)
        labels.append(1 if sev[i] > 0.5 else 0)
    return events, labels


def simulate_population(n_users=60, n_sessions=20, seed=42, mix=(0.35, 0.3, 0.15, 0.2)):
    """Return (events, labels_by_user) for a mixed population."""
    rng = random.Random(seed)
    all_events, labels = [], {}
    for u in range(n_users):
        profile = rng.choices(PROFILES, mix)[0]
        ev, lab = simulate_user(f"user-{u:03d}", n_sessions, profile, seed=seed * 1000 + u)
        all_events.extend(ev)
        labels[hash_user_id(f"user-{u:03d}")] = {"profile": profile, "labels": lab}
    return all_events, labels
