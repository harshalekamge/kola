"""
Personalization service: saved GRU + signal policy, on top of the Context Store.

signal_for(user_id) does this:
  1. re-aggregate the user's raw events into session features,
  2. if fewer than `window` sessions exist -> cold start: default (standard) signal,
  3. otherwise predict P(decline) for every window of the user's history,
  4. replay those probabilities through SignalPolicy (smoothing + cooldown),
  5. return the latest signal.
Replaying the history each time keeps the service stateless and repeatable.
"""

import json
import os

import numpy as np
import torch

from app.aggregator import aggregate_user
from app.policy import SignalPolicy
from drift_predictor import DriftGRU
from personalization_signal import derive_signal

DEFAULT_SHARES = {"voice": 1 / 3, "gesture": 1 / 3, "screen": 1 / 3}


class PersonalizationService:
    def __init__(self, store, model_dir="models"):
        with open(os.path.join(model_dir, "normalizer.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.store = store
        self.features = cfg["features"]
        self.window = cfg["window"]
        n = len(self.features)
        self.mean = np.array(cfg["mean"], dtype=np.float32).reshape(1, 1, n)
        self.std = np.array(cfg["std"], dtype=np.float32).reshape(1, 1, n)
        self.model = DriftGRU(n_features=n, hidden_size=cfg["hidden"])
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "drift_gru.pt"),
                                              map_location="cpu", weights_only=True))
        self.model.eval()

    def _probabilities(self, rows):
        feats = np.array([[r[f] for f in self.features] for r in rows], dtype=np.float32)
        windows = np.stack([feats[i - self.window:i] for i in range(self.window, len(rows) + 1)])
        x = (windows - self.mean) / self.std
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.tensor(x))).numpy()

    def signal_for(self, user_id):
        aggregate_user(self.store, user_id)
        rows = self.store.get_user_sessions(user_id)
        if len(rows) < self.window:
            sig = derive_signal(user_id, max(0, len(rows) - 1), 0.0, DEFAULT_SHARES)
            return {"status": "insufficient_history", "sessions_seen": len(rows),
                    "sessions_needed": self.window, "policy_active": False, "signal": vars(sig)}
        probs = self._probabilities(rows)
        policy = SignalPolicy()
        sig = None
        for k, p in enumerate(probs):
            last = rows[self.window - 1 + k]
            shares = {"voice": last["modality_share_voice"], "gesture": last["modality_share_gesture"],
                      "screen": last["modality_share_screen"]}
            sig, _ = policy.update(user_id, last["session_index"], float(p), shares)
        return {"status": "ok", "sessions_seen": len(rows), "policy_active": policy.is_active(user_id),
                "raw_probability": round(float(probs[-1]), 3), "signal": vars(sig)}
