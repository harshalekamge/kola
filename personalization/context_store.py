"""
User Profile Context Store — Synthetic Data Generator
-------------------------------------------------------
Component 4 (YOURS, per approved Topic Assessment Form):
"Personalized AI Learning and Temporal Interaction Modeling"

Per the approved form, this store logs cross-modal performance metrics
PER SESSION, across MULTIPLE SESSIONS per user (not just within one
session), so that drift can be predicted across sessions:
    - click errors
    - timing drops (command cadence / latency)
    - modality switches
    - gesture stability
    - voice command latency
    - command correction frequency
    - completion accuracy

Upstream components this would normally come from (per the approved form):
    Component 1 -> Voice/NLU (Sinhala-English-Singlish command understanding)
    Component 2 -> Camera-based hand gesture control
    Component 3 -> Real-time screen understanding & UI automation

Since Components 1-3 are under separate development by teammates, this
generator produces realistic DUMMY multi-session logs standing in for
their combined telemetry, with a synthetic "state decline" (fatigue /
accuracy drop / modality-preference shift) injected partway through some
users' session history — this is the ground-truth signal the sequential
model (drift_predictor.py) is trained to predict.
"""

import csv
import random
import math
from dataclasses import dataclass, asdict


@dataclass
class SessionRecord:
    user_id: str
    session_index: int
    click_error_rate: float          # 0-1
    voice_command_latency_ms: float  # rises with decline
    correction_frequency: float      # corrections per command, rises with decline
    gesture_stability: float         # 0-1, falls with decline (tremor-like)
    modality_share_voice: float
    modality_share_gesture: float
    modality_share_screen: float
    completion_accuracy: float       # 0-1, falls with decline
    state_decline_ground_truth: int  # 0/1 - hidden label for evaluation only


def _gen_user_sessions(user_id: str, n_sessions: int, decline_onset_frac, seed: int):
    rng = random.Random(seed)
    onset_index = int(n_sessions * decline_onset_frac) if decline_onset_frac is not None else None

    # baseline modality preference varies per (simulated) user
    base_voice = rng.uniform(0.25, 0.45)
    base_gesture = rng.uniform(0.25, 0.45)
    base_screen = 1.0 - base_voice - base_gesture

    records = []
    for i in range(n_sessions):
        drift = 0.0
        if onset_index is not None and i >= onset_index:
            progress = (i - onset_index) / max(1, (n_sessions - onset_index))
            drift = 1 / (1 + math.exp(-6 * (progress - 0.5)))  # smooth ramp 0->1

        click_error = min(1.0, max(0.0, 0.05 + drift * 0.35 + rng.gauss(0, 0.02)))
        latency = 800 + drift * 700 + rng.gauss(0, 40)
        correction_freq = max(0.0, 0.1 + drift * 0.6 + rng.gauss(0, 0.03))
        gesture_stability = min(1.0, max(0.0, 0.9 - drift * 0.55 + rng.gauss(0, 0.03)))
        accuracy = min(1.0, max(0.0, 0.92 - drift * 0.35 + rng.gauss(0, 0.02)))

        # modality-preference shift: as gesture destabilizes, users lean on voice more
        voice_share = base_voice + drift * 0.20
        gesture_share = max(0.05, base_gesture - drift * 0.25)
        screen_share = max(0.05, 1.0 - voice_share - gesture_share)
        total = voice_share + gesture_share + screen_share
        voice_share, gesture_share, screen_share = (voice_share / total, gesture_share / total, screen_share / total)

        records.append(SessionRecord(
            user_id=user_id,
            session_index=i,
            click_error_rate=round(click_error, 3),
            voice_command_latency_ms=round(latency, 1),
            correction_frequency=round(correction_freq, 3),
            gesture_stability=round(gesture_stability, 3),
            modality_share_voice=round(voice_share, 3),
            modality_share_gesture=round(gesture_share, 3),
            modality_share_screen=round(screen_share, 3),
            completion_accuracy=round(accuracy, 3),
            state_decline_ground_truth=1 if drift > 0.5 else 0,
        ))
    return records


def generate_dataset(n_users: int = 50, n_sessions: int = 20, decline_ratio: float = 0.5, seed: int = 7):
    rng = random.Random(seed)
    all_records = []
    for u in range(n_users):
        user_id = f"U{u+1:03d}"
        will_decline = rng.random() < decline_ratio
        onset = rng.uniform(0.4, 0.65) if will_decline else None
        all_records.extend(_gen_user_sessions(user_id, n_sessions, onset, seed=seed + u))
    return all_records


def save_csv(records, path: str):
    fieldnames = list(asdict(records[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


if __name__ == "__main__":
    data = generate_dataset()
    save_csv(data, "context_store_sessions.csv")
    print(f"Generated {len(data)} session records for {50} simulated users -> context_store_sessions.csv")
