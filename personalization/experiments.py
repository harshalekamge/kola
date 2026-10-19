"""
PP1 experiments (run:  python experiments.py)

A. Window-size test: GRU with 3, 5 and 7 past sessions vs the static-threshold baseline.
   Repeated over 5 random user splits; we report mean +- std (not one lucky run).
B. Signal policy test: raw GRU output vs smoothing+hysteresis+cooldown policy,
   counting how often the interface settings flip and how many sessions are adjusted
   when the user is NOT declining.
C. Demo chart for slides: docs/pp1_demo.png  and  docs/results_window_sizes.png
Writes docs/RESULTS.md. Data is SIMULATED (see docs/INTERFACE_CONTRACT.md).
"""

import os
import numpy as np

from app.simulator import simulate_population
from app.store import ContextStore
from app.aggregator import aggregate_all
from app.bridge import load_records_by_user
from app.trainer import split_users, train_with_validation
from app.policy import SignalPolicy
from drift_predictor import build_windows, normalize, predict_proba
from run_pipeline import metrics, static_threshold_baseline

WINDOWS = (3, 5, 7)
SEEDS = (0, 1, 2, 3, 4)
N_USERS, N_SESSIONS, DATA_SEED = 100, 20, 42


def load_data():
    events, labels = simulate_population(N_USERS, N_SESSIONS, DATA_SEED)
    store = ContextStore(":memory:")
    store.add_events(events)
    aggregate_all(store, labels)
    return load_records_by_user(store), labels


def run_once(by_user, window, seed):
    X, y, meta = build_windows(by_user, window)
    tr_u, va_u, te_u = split_users(list(by_user), seed)
    idx = lambda group: [i for i, m in enumerate(meta) if m[0] in group]
    tr, va, te = idx(tr_u), idx(va_u), idx(te_u)
    Xtr, mean, std = normalize(X[tr])
    Xva, _, _ = normalize(X[va], mean, std)
    Xte, _, _ = normalize(X[te], mean, std)
    model, hist = train_with_validation(Xtr, y[tr], Xva, y[va], seed=seed)
    probs = predict_proba(model, Xte)
    return {
        "gru": metrics((probs >= 0.5).astype(int), y[te]),
        "base": metrics(static_threshold_baseline(X[te]), y[te]),
        "best_epoch": hist["best_epoch"],
        "probs": probs, "y": y[te], "meta": [meta[i] for i in te],
        "X_raw": X[te], "hist": hist,
    }


def summarize(runs, key):
    out = {}
    for m in ("accuracy", "precision", "recall", "f1", "false_adjust"):
        vals = np.array([r[key][m] for r in runs])
        out[m] = (vals.mean(), vals.std())
    return out


def policy_eval(run, labels_by_user):
    """Compare raw thresholding vs SignalPolicy on the test users of one run."""
    per_user = {}
    for p, y, (uid, sidx), Xr in zip(run["probs"], run["y"], run["meta"], run["X_raw"]):
        per_user.setdefault(uid, []).append((sidx, float(p), int(y), Xr[-1]))
    res = {"raw": dict(flips=0, false_sessions=0, missed=0, n=0),
           "policy": dict(flips=0, false_sessions=0, missed=0, n=0)}
    blip = {"raw": 0, "policy": 0, "users": 0}
    for uid, rows in per_user.items():
        rows.sort()
        pol = SignalPolicy()
        prev_raw, prev_pol = False, False
        profile = labels_by_user[uid]["profile"]
        blip_raw = blip_pol = 0
        for sidx, p, y, last in rows:
            shares = {"voice": last[4], "gesture": last[5], "screen": last[6]}
            sig, _ = pol.update(uid, sidx, p, shares)
            raw_on, pol_on = p >= 0.5, pol.is_active(uid)
            for name, on, prev in (("raw", raw_on, prev_raw), ("policy", pol_on, prev_pol)):
                res[name]["n"] += 1
                res[name]["flips"] += int(on != prev)
                res[name]["false_sessions"] += int(on and y == 0)
                res[name]["missed"] += int((not on) and y == 1)
            if profile == "blip":
                blip_raw += int(raw_on and y == 0)
                blip_pol += int(pol_on and y == 0)
            prev_raw, prev_pol = raw_on, pol_on
        if profile == "blip":
            blip["raw"] += blip_raw
            blip["policy"] += blip_pol
            blip["users"] += 1
    return res, blip, len(per_user)


def make_charts(all_runs, demo_run, labels_by_user, by_user):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("docs", exist_ok=True)
    # --- chart 1: window sizes ---
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    for k, (label, key) in enumerate((("GRU F1", "gru"), ("Static-threshold F1", "base"))):
        means = [summarize(all_runs[w], key)["f1"][0] for w in WINDOWS]
        stds = [summarize(all_runs[w], key)["f1"][1] for w in WINDOWS]
        ax.bar(np.arange(len(WINDOWS)) + k * width, means, width, yerr=stds, capsize=3,
               color=("#1f3a5f", "#9aa5b1")[k], label=label)
    ax.set_xticks(np.arange(len(WINDOWS)) + width / 2)
    ax.set_xticklabels([f"{w} sessions" for w in WINDOWS])
    ax.set_ylabel("F1 (mean ± std, 5 splits)")
    ax.set_title("History length vs detection quality (simulated data)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.savefig("docs/results_window_sizes.png", dpi=150)
    plt.close(fig)

    # --- chart 2: demo for one 'gradual' test user + one 'blip' user ---
    per_user = {}
    for p, y, (uid, sidx) in zip(demo_run["probs"], demo_run["y"], demo_run["meta"]):
        per_user.setdefault(uid, []).append((sidx, float(p), int(y)))
    picks = {}
    for uid in per_user:
        prof = labels_by_user[uid]["profile"]
        if prof in ("gradual", "blip") and prof not in picks:
            picks[prof] = uid
    fig, axes = plt.subplots(2, len(picks), figsize=(6 * len(picks), 6), sharex="col", squeeze=False)
    for col, (prof, uid) in enumerate(picks.items()):
        rows = sorted(per_user[uid])
        s = [r[0] for r in rows]
        probs = [r[1] for r in rows]
        truth = [r[2] for r in rows]
        pol = SignalPolicy()
        gest = []
        for sidx, p, _ in rows:
            sig, _ = pol.update(uid, sidx, p, {"voice": .4, "gesture": .3, "screen": .3})
            gest.append(sig.gesture_tolerance_multiplier)
        sessions = by_user[uid]
        acc = [r.completion_accuracy for r in sessions if r.session_index in s]
        ax0, ax1 = axes[0][col], axes[1][col]
        ax0.plot(s, acc, color="#1f3a5f", marker="o", ms=3, label="Completion accuracy (from events)")
        ax0.set_ylabel("Completion accuracy")
        ax0.set_title(f"{prof} user {uid}")
        for si, t in zip(s, truth):
            if t:
                ax0.axvspan(si - .5, si + .5, color="#d9534f", alpha=.15)
        ax1.plot(s, probs, color="#9aa5b1", label="GRU P(decline)")
        ax1.set_ylabel("P(decline)")
        ax1.set_ylim(0, 1)
        ax1b = ax1.twinx()
        ax1b.step(s, gest, where="mid", color="#d9534f", label="Gesture tolerance (after policy)")
        ax1b.set_ylim(0.9, 1.9)
        ax1b.set_ylabel("Gesture tolerance ×")
        ax1.set_xlabel("Session")
        if col == 0:
            ax0.legend(fontsize=8, loc="lower left")
            ax1.legend(fontsize=8, loc="upper left")
            ax1b.legend(fontsize=8, loc="upper right")
    fig.suptitle("Simulated users. Shaded = sessions that are truly in decline (hidden label, used only for scoring).", fontsize=10)
    plt.tight_layout()
    plt.savefig("docs/pp1_demo.png", dpi=150)
    plt.close(fig)


def main():
    by_user, labels = load_data()
    print(f"Data: {N_USERS} simulated users x {N_SESSIONS} sessions "
          f"(profiles: " + ", ".join(f"{p}={sum(v['profile']==p for v in labels.values())}"
                                     for p in ('stable', 'gradual', 'recovering', 'blip')) + ")")
    all_runs = {}
    lines = ["# PP1 results (simulated data)", "",
             f"{N_USERS} simulated users, {N_SESSIONS} sessions each; train/val/test = 60/20/20 users; "
             f"mean ± std over {len(SEEDS)} random user splits. Early stopping on validation loss.", "",
             "| Window | System | Accuracy | Precision | Recall | F1 | False-adjust |",
             "|---|---|---|---|---|---|---|"]
    for w in WINDOWS:
        all_runs[w] = [run_once(by_user, w, s) for s in SEEDS]
        for name, key in (("GRU", "gru"), ("Static threshold", "base")):
            sm = summarize(all_runs[w], key)
            row = " | ".join(f"{sm[m][0]:.3f} ± {sm[m][1]:.3f}" for m in
                             ("accuracy", "precision", "recall", "f1", "false_adjust"))
            lines.append(f"| {w} | {name} | {row} |")
            print(f"window {w} {name:<17} " + " | ".join(
                f"{m[:5]} {sm[m][0]:.3f}±{sm[m][1]:.3f}" for m in ("accuracy", "precision", "recall", "f1", "false_adjust")))
    epochs = [r["best_epoch"] for r in all_runs[5]]
    print(f"best epoch (window 5): {epochs}")

    # policy test pooled over seeds, window 5
    tot = {"raw": dict(flips=0, false_sessions=0, missed=0, n=0), "policy": dict(flips=0, false_sessions=0, missed=0, n=0)}
    blip_tot = {"raw": 0, "policy": 0, "users": 0}
    users_tot = 0
    for r in all_runs[5]:
        res, blip, nu = policy_eval(r, labels)
        users_tot += nu
        for k in tot:
            for f in tot[k]:
                tot[k][f] += res[k][f]
        for f in blip_tot:
            blip_tot[f] += blip[f]
    lines += ["", "## Signal policy (window 5)", "",
              "| Version | Setting flips per user | Adjusted sessions when NOT declining | Missed decline sessions |",
              "|---|---|---|---|"]
    for name, key in (("Raw GRU output (p ≥ 0.5)", "raw"), ("Smoothing + hysteresis + cooldown", "policy")):
        t = tot[key]
        lines.append(f"| {name} | {t['flips']/users_tot:.2f} | {t['false_sessions']/users_tot:.2f} | {t['missed']/users_tot:.2f} |")
        print(f"policy {name:<36} flips/user {t['flips']/users_tot:.2f} | false sessions/user {t['false_sessions']/users_tot:.2f} | missed/user {t['missed']/users_tot:.2f}")
    if blip_tot["users"]:
        lines += ["", f"One-off 'blip' users ({blip_tot['users']} across splits): sessions wrongly adjusted "
                  f"raw = {blip_tot['raw']}, policy = {blip_tot['policy']}."]
        print(f"blip users: wrongly adjusted sessions raw={blip_tot['raw']} policy={blip_tot['policy']} (n={blip_tot['users']})")
    lines += ["", "Limits: all data is simulated by app/simulator.py; real accuracy depends on Components 1-3 data."]
    os.makedirs("docs", exist_ok=True)
    with open("docs/RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    make_charts(all_runs, all_runs[5][0], labels, by_user)
    print("Saved docs/RESULTS.md, docs/results_window_sizes.png, docs/pp1_demo.png")


if __name__ == "__main__":
    main()