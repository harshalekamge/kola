"""
End-to-end Prototype Demo (v2 — aligned to approved Topic Assessment Form)
----------------------------------------------------------------------------
Component 4: Personalized AI Learning and Temporal Interaction Modeling

Pipeline:
    1. Generate synthetic multi-session cross-modal context data (dummy).
    2. Build sliding windows of past sessions -> train/test split BY USER.
    3. Train the GRU sequential drift predictor.
    4. Evaluate vs (synthetic) ground truth state-decline labels.
    5. Derive personalization signals for a few example predictions.
    6. Run one example through the AI Assistant Engine fusion mock.
    7. Save results CSV, metrics, and plots.

Run:  python3 demo.py
"""

import csv
from collections import defaultdict
import numpy as np

from context_store import generate_dataset, save_csv
from drift_predictor import build_windows, normalize, train_model, predict_proba, WINDOW_SIZE
from personalization_signal import derive_signal
from ai_assistant_engine import ModalityInput, fuse


def main():
    print("1) Generating synthetic multi-session context-store data (dummy)...")
    records = generate_dataset(n_users=50, n_sessions=20, decline_ratio=0.5, seed=7)
    save_csv(records, "context_store_sessions.csv")
    by_user = defaultdict(list)
    for r in records:
        by_user[r.user_id].append(r)
    print(f"   {len(records)} session records across {len(by_user)} simulated users "
          f"-> context_store_sessions.csv")

    print(f"2) Building sliding windows (window size = {WINDOW_SIZE} past sessions)...")
    X, y, meta = build_windows(by_user, window=WINDOW_SIZE)
    print(f"   {len(X)} training windows built")

    # split by user (70/30) so we evaluate generalization to unseen users
    user_ids = sorted(by_user.keys())
    rng = np.random.RandomState(0)
    rng.shuffle(user_ids)
    split = int(len(user_ids) * 0.7)
    train_users, test_users = set(user_ids[:split]), set(user_ids[split:])

    train_idx = [i for i, m in enumerate(meta) if m[0] in train_users]
    test_idx = [i for i, m in enumerate(meta) if m[0] in test_users]

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    X_train, mean, std = normalize(X_train_raw)
    X_test, _, _ = normalize(X_test_raw, mean, std)

    print("3) Training GRU sequential drift predictor...")
    model, loss_history = train_model(X_train, y_train, epochs=80, lr=0.01)
    print(f"   final training loss: {loss_history[-1]:.4f}")

    print("4) Evaluating on held-out users...")
    probs_test = predict_proba(model, X_test)
    preds_test = (probs_test >= 0.5).astype(int)

    tp = int(((preds_test == 1) & (y_test == 1)).sum())
    fp = int(((preds_test == 1) & (y_test == 0)).sum())
    tn = int(((preds_test == 0) & (y_test == 0)).sum())
    fn = int(((preds_test == 0) & (y_test == 1)).sum())

    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-6, (precision + recall))
    false_adjustment_rate = fp / max(1, (fp + tn))

    print(f"   Accuracy:              {accuracy:.3f}")
    print(f"   Precision:             {precision:.3f}")
    print(f"   Recall:                {recall:.3f}")
    print(f"   F1:                    {f1:.3f}")
    print(f"   False-adjustment rate: {false_adjustment_rate:.3f}")

    with open("summary_metrics.txt", "w") as f:
        f.write("Component 4 Prototype - Evaluation Summary (synthetic data)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Users (train/test split):   {len(train_users)}/{len(test_users)}\n")
        f.write(f"Sessions per user:          20\n")
        f.write(f"Window size:                {WINDOW_SIZE} past sessions\n")
        f.write(f"Test-set accuracy:          {accuracy:.3f}\n")
        f.write(f"Test-set precision:         {precision:.3f}\n")
        f.write(f"Test-set recall:            {recall:.3f}\n")
        f.write(f"Test-set F1:                {f1:.3f}\n")
        f.write(f"False-adjustment rate:      {false_adjustment_rate:.3f}\n")
        f.write(f"Final training loss (BCE):  {loss_history[-1]:.4f}\n")
    print("   Summary metrics saved -> summary_metrics.txt")

    print("5) Deriving personalization signals for held-out test predictions...")
    with open("personalization_signals.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "user_id", "session_index", "decline_probability", "ground_truth",
            "recommended_default_modality", "gesture_tolerance_multiplier",
            "speech_timeout_multiplier", "ui_shortcut_mode", "confidence",
        ])
        for i, idx in enumerate(test_idx):
            uid, session_idx = meta[idx]
            # last session in the window gives current modality shares
            last_session_feats = X_test_raw[i][-1]
            shares = {
                "voice": last_session_feats[4],
                "gesture": last_session_feats[5],
                "screen": last_session_feats[6],
            }
            sig = derive_signal(uid, session_idx, float(probs_test[i]), shares)
            writer.writerow([
                sig.user_id, sig.session_index, sig.decline_probability, int(y_test[i]),
                sig.recommended_default_modality, sig.gesture_tolerance_multiplier,
                sig.speech_timeout_multiplier, sig.ui_shortcut_mode, sig.confidence,
            ])
    print("   Personalization signals saved -> personalization_signals.csv")

    print("6) Running one example through the AI Assistant Engine fusion mock...")
    example_uid, example_session = meta[test_idx[0]]
    example_shares = {
        "voice": X_test_raw[0][-1][4], "gesture": X_test_raw[0][-1][5], "screen": X_test_raw[0][-1][6],
    }
    example_signal = derive_signal(example_uid, example_session, float(probs_test[0]), example_shares)
    mock_inputs = [
        ModalityInput(source="voice", action="open_settings", confidence=0.55),
        ModalityInput(source="gesture", action="scroll_down", confidence=0.50),
        ModalityInput(source="screen", action="click_search_icon", confidence=0.60),
    ]
    decision = fuse(example_uid, mock_inputs, example_signal)
    print(f"   {decision.explanation}")

    plot_results(loss_history, probs_test, y_test)


def plot_results(loss_history, probs_test, y_test):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   (matplotlib not available - skipping plots)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(loss_history, color="black")
    axes[0].set_title("GRU training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")

    order = np.argsort(probs_test)
    axes[1].scatter(range(len(probs_test)), probs_test[order], c="black", s=10,
                     label="predicted decline probability")
    axes[1].scatter(range(len(y_test)), y_test[order], c="gray", marker="x", s=15,
                     label="ground truth (0/1)")
    axes[1].set_title("Test-set predictions vs ground truth")
    axes[1].set_xlabel("Test sample (sorted by predicted probability)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("training_and_predictions.png", dpi=150)
    print("   Plot saved -> training_and_predictions.png")


if __name__ == "__main__":
    main()
