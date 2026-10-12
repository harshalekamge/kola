"""
Temporal Drift Predictor (Sequential Model)
---------------------------------------------
Component 4 (YOURS): "Personalized AI Learning and Temporal Interaction
Modeling" — per the approved Topic Assessment Form:

  "...train a sequential model (LSTM/GRU) on historical cross-modal
   interaction metrics across sessions to predict drift in user state
   (e.g., fatigue, declining accuracy, modality-preference shift) over
   time; expose predictions as a personalization signal..."

This module implements that sequential model as a small GRU classifier
in PyTorch. Given a window of the user's last K sessions (from
context_store.py), it predicts the probability that the user's state is
declining in the NEXT session — this probability is what
personalization_signal.py turns into concrete UI adjustments.
"""

import torch
import torch.nn as nn
import numpy as np

FEATURES = [
    "click_error_rate",
    "voice_command_latency_ms",
    "correction_frequency",
    "gesture_stability",
    "modality_share_voice",
    "modality_share_gesture",
    "modality_share_screen",
    "completion_accuracy",
]

WINDOW_SIZE = 5  # number of past sessions used to predict the next one


class DriftGRU(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 16):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        # x: (batch, window, n_features)
        _, h_n = self.gru(x)
        h_last = h_n[-1]           # (batch, hidden)
        logit = self.head(h_last)  # (batch, 1)
        return logit.squeeze(-1)


def build_windows(records_by_user: dict, window: int = WINDOW_SIZE):
    """Turn per-user chronological session records into (X, y, meta) training
    windows: X = features of sessions [i-window, i), y = decline label AT
    session i (i.e. predicting the *next* session's state)."""
    X, y, meta = [], [], []
    for uid, sessions in records_by_user.items():
        sessions = sorted(sessions, key=lambda r: r.session_index)
        for i in range(window, len(sessions)):
            past = sessions[i - window:i]
            target = sessions[i]
            feat_seq = [[getattr(s, f) for f in FEATURES] for s in past]
            X.append(feat_seq)
            y.append(target.state_decline_ground_truth)
            meta.append((uid, target.session_index))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), meta


def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    return (X - mean) / std, mean, std


def train_model(X_train, y_train, epochs: int = 60, lr: float = 0.01, seed: int = 0):
    torch.manual_seed(seed)
    model = DriftGRU(n_features=X_train.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train)

    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return model, history


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy()
    return probs
