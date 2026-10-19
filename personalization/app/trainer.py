"""
Training with a validation set and early stopping.

Why: the first pipeline trained for a fixed number of epochs and its training
loss fell to ~0.01, a sign the GRU can memorise its training users. Here we
watch a separate VALIDATION group of users and keep the model from the epoch
where validation loss was lowest. The TEST group is never touched until the end.
"""

import copy
import numpy as np
import torch
import torch.nn as nn

from drift_predictor import DriftGRU


def split_users(users, seed, fractions=(0.6, 0.2, 0.2)):
    """Split user ids into train / validation / test groups (by USER, not by window)."""
    users = sorted(users)
    rng = np.random.RandomState(seed)
    rng.shuffle(users)
    n_train = int(len(users) * fractions[0])
    n_val = int(len(users) * fractions[1])
    return set(users[:n_train]), set(users[n_train:n_train + n_val]), set(users[n_train + n_val:])


def train_with_validation(X_train, y_train, X_val, y_val, epochs=200, lr=0.01,
                          patience=25, hidden=16, seed=0):
    """Return (best_model, history). history has train_loss, val_loss, best_epoch."""
    torch.manual_seed(seed)
    model = DriftGRU(n_features=X_train.shape[-1], hidden_size=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(X_train), torch.tensor(y_train)
    Xv, yv = torch.tensor(X_val), torch.tensor(y_val)

    best_state, best_val, best_epoch, waited = None, float("inf"), 0, 0
    train_hist, val_hist = [], []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        train_hist.append(loss.item())
        val_hist.append(val_loss)
        if val_loss < best_val - 1e-4:
            best_val, best_epoch, waited = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"train_loss": train_hist, "val_loss": val_hist, "best_epoch": best_epoch}
