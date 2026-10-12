import copy
import pytest

from app.schema import validate_event, hash_user_id
from app.simulator import simulate_user, simulate_population
from app.store import ContextStore
from app.aggregator import aggregate_session, aggregate_all, aggregate_user


def _one(events, source):
    return next(e for e in events if e["source"] == source)


def test_hash_is_stable_and_pseudonymous():
    assert hash_user_id("alice") == hash_user_id("alice")
    assert "alice" not in hash_user_id("alice")


def test_simulated_events_are_valid():
    events, labels = simulate_user("x", 5, "stable", seed=1)
    assert len(labels) == 5
    for e in events:
        validate_event(e)


@pytest.mark.parametrize("mutate", [
    lambda e: e.pop("session_id"),
    lambda e: e.update(source="lidar"),
    lambda e: e.update(timestamp="yesterday"),
    lambda e: e.update(schema_version="9.9"),
    lambda e: e["payload"].pop(next(iter(e["payload"]))),
])
def test_bad_events_rejected(mutate):
    events, _ = simulate_user("x", 2, "stable", seed=1)
    bad = copy.deepcopy(events[0])
    mutate(bad)
    with pytest.raises(ValueError):
        validate_event(bad)


def test_out_of_range_rejected():
    events, _ = simulate_user("x", 2, "stable", seed=1)
    bad = copy.deepcopy(_one(events, "gesture"))
    bad["payload"]["stability_score"] = 1.7
    with pytest.raises(ValueError):
        validate_event(bad)


def test_aggregate_known_session():
    def ev(src, **p):
        return {"source": src, "payload": p}
    events = [
        ev("voice", latency_ms=1000, success=True, was_corrected=False),
        ev("voice", latency_ms=2000, success=False, was_corrected=True),
        ev("gesture", stability_score=0.8, success=True, was_retried=False),
        ev("screen", click_success=False),
    ]
    f = aggregate_session(events)
    assert f["voice_command_latency_ms"] == 1500
    assert f["gesture_stability"] == 0.8
    assert f["click_error_rate"] == 1.0
    assert f["correction_frequency"] == pytest.approx(1 / 3)
    assert f["modality_share_voice"] == 0.5
    assert f["completion_accuracy"] == 0.5
    assert f["modality_share_voice"] + f["modality_share_gesture"] + f["modality_share_screen"] == pytest.approx(1)


def test_missing_modality_gives_no_nan():
    events = [{"source": "voice", "payload": {"latency_ms": 900, "success": True, "was_corrected": False}}]
    f = aggregate_session(events)
    assert f["gesture_stability"] == 0.0 and f["click_error_rate"] == 0.0
    assert f["modality_share_gesture"] == 0.0


def test_empty_session_raises():
    with pytest.raises(ValueError):
        aggregate_session([])


def test_store_roundtrip_and_ordering():
    events, labels = simulate_user("x", 6, "gradual", seed=3)
    store = ContextStore()
    assert store.add_events(events) == len(events)
    uid = events[0]["user_id"]
    aggregate_user(store, uid, labels)
    rows = store.get_user_sessions(uid)
    assert [r["session_index"] for r in rows] == list(range(6))
    assert [r["label_decline"] for r in rows] == labels
    assert rows[0]["start_ts"] < rows[-1]["start_ts"]


def test_store_rejects_invalid_event():
    store = ContextStore()
    with pytest.raises(ValueError):
        store.add_events([{"foo": 1}])
    assert store.count("events") == 0


def test_decline_visible_in_features():
    events, labels = simulate_user("x", 20, "gradual", seed=5)
    store = ContextStore()
    store.add_events(events)
    uid = events[0]["user_id"]
    aggregate_user(store, uid, labels)
    rows = store.get_user_sessions(uid)
    early = sum(r["completion_accuracy"] for r in rows[:4]) / 4
    late = sum(r["completion_accuracy"] for r in rows[-4:]) / 4
    assert late < early


def test_population_profiles_and_blip_labels():
    events, labels = simulate_population(n_users=30, n_sessions=15, seed=1)
    profiles = {v["profile"] for v in labels.values()}
    assert profiles == {"stable", "gradual", "recovering", "blip"}
    for v in labels.values():
        if v["profile"] == "blip":
            assert sum(v["labels"]) == 1
        if v["profile"] == "stable":
            assert sum(v["labels"]) == 0
    store = ContextStore()
    store.add_events(events)
    assert aggregate_all(store, labels) == 30 * 15
