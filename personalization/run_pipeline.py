"""
PP1 pipeline, end to end (run:  python run_pipeline.py)

 1. Simulate raw events from "Components 1-3"        (app/simulator.py)
 2. Validate + store them in SQLite                   (app/store.py)
 3. Aggregate events into 8 features per session      (app/aggregator.py)
 4. Build 5-session windows, split users 70/30        (drift_predictor.py)
 5. Train the GRU on training users                   (drift_predictor.py)
 6. Evaluate GRU vs a static-threshold baseline on unseen users
 7. Turn predictions into personalization signals     (personalization_signal.py)
"""

import numpy as np

from app.simulator import simulate_population
from app.store import ContextStore
from app.aggregator import aggregate_all
from app.bridge import load_records_by_user
from drift_predictor import build_windows, normalize, train_model, predict_proba, FEATURES, WINDOW_SIZE
from personalization_signal import derive_signal


def metrics(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(1, len(y)); prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "false_adjust": fp / max(1, fp + tn)}


def static_threshold_baseline(X_raw):
    """The 'dumb' system: look only at the LAST session, one fixed rule for everyone."""
    last = X_raw[:, -1, :]
    acc = last[:, FEATURES.index("completion_accuracy")]
    err = last[:, FEATURES.index("click_error_rate")]
    return ((acc < 0.80) | (err > 0.25)).astype(int)


def show(name, m):
    print(f"   {name:<18} acc {m['accuracy']:.3f} | precision {m['precision']:.3f} | "
          f"recall {m['recall']:.3f} | F1 {m['f1']:.3f} | false-adjust {m['false_adjust']:.3f}")


def main(n_users=60, n_sessions=20, seed=42):
    print("1) Simulating raw events from Components 1-3...")
    events, labels = simulate_population(n_users, n_sessions, seed)
    print(f"   {len(events)} events for {n_users} users")

    print("2) Validating + storing in SQLite Context Store...")
    store = ContextStore("context_store.db")
    store.add_events(events)

    print("3) Aggregating events -> 8 features per session...")
    n = aggregate_all(store, labels)
    print(f"   {n} session rows")

    print(f"4) Building windows of {WINDOW_SIZE} sessions, splitting users 70/30...")
    by_user = load_records_by_user(store)
    X, y, meta = build_windows(by_user, WINDOW_SIZE)
    users = sorted(by_user); rng = np.random.RandomState(0); rng.shuffle(users)
    cut = int(len(users) * 0.7); train_u = set(users[:cut])
    tr = [i for i, m in enumerate(meta) if m[0] in train_u]
    te = [i for i, m in enumerate(meta) if m[0] not in train_u]
    Xtr, mean, std = normalize(X[tr]); Xte, _, _ = normalize(X[te], mean, std)
    print(f"   train windows {len(tr)} | test windows {len(te)} (unseen users: {len(users) - cut})")

    print("5) Training the GRU...")
    model, hist = train_model(Xtr, y[tr], epochs=150, lr=0.01)
    print(f"   loss {hist[0]:.3f} -> {hist[-1]:.3f}")

    print("6) Evaluating on unseen users:")
    probs = predict_proba(model, Xte)
    show("GRU", metrics((probs >= 0.5).astype(int), y[te]))
    show("Static threshold", metrics(static_threshold_baseline(X[te]), y[te]))

    print("7) Example personalization signals:")
    for k in (0, len(te) // 2, len(te) - 1):
        uid, sidx = meta[te[k]]
        last = X[te[k]][-1]
        shares = {"voice": last[4], "gesture": last[5], "screen": last[6]}
        s = derive_signal(uid, sidx, float(probs[k]), shares)
        print(f"   {uid} session {sidx}: P(decline)={s.decline_probability:.2f} -> "
              f"gesture x{s.gesture_tolerance_multiplier}, speech x{s.speech_timeout_multiplier}, "
              f"UI {s.ui_shortcut_mode}, modality {s.recommended_default_modality}")


if __name__ == "__main__":
    main()
