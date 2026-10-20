"""
Train the GRU once and SAVE it (run:  python train_model.py)

Creates in models/:
    drift_gru.pt     the trained weights
    normalizer.json  the mean/std used to scale features, plus window size and hidden size
    metadata.json    how it was trained and its test-set scores (simulated data)

The saved files are what the local API (app/api.py) loads, so Components 1-3
never need to retrain anything.
"""

import json
import os
import sys
from datetime import date

import numpy as np
import torch

from app.simulator import simulate_population
from app.store import ContextStore
from app.aggregator import aggregate_all
from app.bridge import load_records_by_user
from app.trainer import split_users, train_with_validation
from drift_predictor import build_windows, normalize, predict_proba, FEATURES, WINDOW_SIZE
from run_pipeline import metrics, static_threshold_baseline

HIDDEN = 16


def train_and_save(out_dir="models", n_users=100, n_sessions=20, data_seed=42,
                   split_seed=0, epochs=200):
    events, labels = simulate_population(n_users, n_sessions, data_seed)
    store = ContextStore(":memory:")
    store.add_events(events)
    aggregate_all(store, labels)
    by_user = load_records_by_user(store)

    X, y, meta = build_windows(by_user, WINDOW_SIZE)
    tr_u, va_u, te_u = split_users(list(by_user), split_seed)
    pick = lambda g: [i for i, m in enumerate(meta) if m[0] in g]
    tr, va, te = pick(tr_u), pick(va_u), pick(te_u)
    Xtr, mean, std = normalize(X[tr])
    Xva, _, _ = normalize(X[va], mean, std)
    Xte, _, _ = normalize(X[te], mean, std)

    model, hist = train_with_validation(Xtr, y[tr], Xva, y[va], epochs=epochs,
                                        hidden=HIDDEN, seed=split_seed)
    probs = predict_proba(model, Xte)
    gru = metrics((probs >= 0.5).astype(int), y[te])
    base = metrics(static_threshold_baseline(X[te]), y[te])

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "drift_gru.pt"))
    with open(os.path.join(out_dir, "normalizer.json"), "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "window": WINDOW_SIZE, "hidden": HIDDEN,
                   "mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}, f, indent=2)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"trained_on": str(date.today()), "data": "SIMULATED",
                   "n_users": n_users, "n_sessions": n_sessions,
                   "split_users": [len(tr_u), len(va_u), len(te_u)],
                   "best_epoch": hist["best_epoch"],
                   "test_gru": gru, "test_static_threshold": base}, f, indent=2)
    return {"gru": gru, "base": base, "best_epoch": hist["best_epoch"]}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "models"
    r = train_and_save(out)
    print(f"Saved model to {out}/  (best epoch {r['best_epoch']})")
    for name, m in (("GRU", r["gru"]), ("Static threshold", r["base"])):
        print(f"  {name:<17} acc {m['accuracy']:.3f} | precision {m['precision']:.3f} | "
              f"recall {m['recall']:.3f} | F1 {m['f1']:.3f} | false-adjust {m['false_adjust']:.3f}")
