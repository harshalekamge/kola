import numpy as np

from app.policy import SignalPolicy
from app.trainer import split_users, train_with_validation

SHARES = {"voice": 0.4, "gesture": 0.3, "screen": 0.3}


def run(policy, probs, uid="u"):
    return [policy.update(uid, i, p, SHARES) for i, p in enumerate(probs)]


def test_single_spike_does_not_activate():
    out = run(SignalPolicy(), [0.1, 0.1, 0.95, 0.1, 0.1, 0.1])
    assert not any(changed for _, changed in out)
    assert all(sig.ui_shortcut_mode == "standard" for sig, _ in out)


def test_sustained_decline_activates_then_recovers():
    out = run(SignalPolicy(), [0.1, 0.1] + [0.9] * 5 + [0.05] * 8)
    modes = [sig.ui_shortcut_mode for sig, _ in out]
    assert "simplified" in modes
    assert modes[-1] == "standard"


def test_cooldown_blocks_rapid_flips():
    p = SignalPolicy(alpha=1.0, cooldown=3)
    out = run(p, [0.9, 0.0, 0.9, 0.0, 0.9, 0.0])
    changes = sum(c for _, c in out)
    assert changes <= 2


def test_users_are_independent():
    p = SignalPolicy(alpha=1.0, cooldown=1)
    run(p, [0.9, 0.9, 0.9], "a")
    assert p.is_active("a") and not p.is_active("b")


def test_split_users_disjoint_and_complete():
    users = [f"u{i}" for i in range(50)]
    tr, va, te = split_users(users, seed=1)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(users)


def test_training_restores_best_epoch():
    rng = np.random.RandomState(0)
    X = rng.randn(80, 5, 8).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(np.float32)
    model, hist = train_with_validation(X[:50], y[:50], X[50:], y[50:], epochs=60, patience=10)
    assert 0 <= hist["best_epoch"] < len(hist["val_loss"])
    assert min(hist["val_loss"]) == hist["val_loss"][hist["best_epoch"]]
