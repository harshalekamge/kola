import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.simulator import simulate_user
from train_model import train_and_save


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    train_and_save(str(d), n_users=40, n_sessions=16, epochs=60)
    return str(d)


@pytest.fixture()
def client(model_dir, tmp_path):
    return TestClient(create_app(db_path=str(tmp_path / "t.db"), model_dir=model_dir))


def send_sessions(client, events, upto):
    sessions = sorted({e["session_id"] for e in events})[:upto]
    client.post("/events", json=[e for e in events if e["session_id"] in sessions])


def test_model_files_saved(model_dir):
    import os
    for f in ("drift_gru.pt", "normalizer.json", "metadata.json"):
        assert os.path.exists(os.path.join(model_dir, f))


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_invalid_event_rejected_with_message(client):
    r = client.post("/events", json={"foo": 1})
    assert r.status_code == 422 and "missing field" in r.json()["detail"]


def test_unknown_user_404(client):
    assert client.get("/signal/nobody").status_code == 404


def test_cold_start_returns_default(client):
    events, _ = simulate_user("a", 8, "stable", seed=1)
    send_sessions(client, events, 3)
    out = client.get(f"/signal/{events[0]['user_id']}").json()
    assert out["status"] == "insufficient_history"
    assert out["signal"]["ui_shortcut_mode"] == "standard"


def test_declining_user_gets_stronger_signal_than_stable_user(client):
    stable, _ = simulate_user("s", 16, "stable", seed=2)
    declin, _ = simulate_user("d", 16, "gradual", seed=3)
    send_sessions(client, stable, 16)
    send_sessions(client, declin, 16)
    s = client.get(f"/signal/{stable[0]['user_id']}").json()
    d = client.get(f"/signal/{declin[0]['user_id']}").json()
    assert s["status"] == d["status"] == "ok"
    assert d["signal"]["gesture_tolerance_multiplier"] >= s["signal"]["gesture_tolerance_multiplier"]
    assert d["raw_probability"] > s["raw_probability"]
