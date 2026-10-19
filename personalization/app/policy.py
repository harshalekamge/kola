"""
Signal policy: smoothing + hysteresis + cooldown.

Problem: if the settings follow the raw GRU probability, one noisy session can
flip the interface to 'simplified' and back, which confuses a real user.

Rules (per user):
  1. Smooth the probability with an exponential moving average (EMA).
  2. Turn the adaptation ON only when the smoothed value reaches `enter`.
  3. Turn it OFF only when the smoothed value falls to `exit_` (lower than enter).
  4. After any change, wait `cooldown` sessions before allowing another change.
"""

from dataclasses import dataclass

from personalization_signal import derive_signal


@dataclass
class _UserState:
    smoothed: float = 0.0
    active: bool = False
    since_change: int = 10**6
    seen: int = 0


class SignalPolicy:
    def __init__(self, alpha=0.5, enter=0.6, exit_=0.35, cooldown=2):
        self.alpha, self.enter, self.exit_, self.cooldown = alpha, enter, exit_, cooldown
        self._state = {}

    def update(self, user_id, session_index, probability, modality_shares):
        """Feed one new prediction; return (signal, changed)."""
        st = self._state.setdefault(user_id, _UserState())
        st.smoothed = probability if st.seen == 0 else (
            self.alpha * probability + (1 - self.alpha) * st.smoothed)
        st.seen += 1
        st.since_change += 1

        changed = False
        if st.since_change >= self.cooldown:
            if not st.active and st.smoothed >= self.enter:
                st.active, changed = True, True
            elif st.active and st.smoothed <= self.exit_:
                st.active, changed = False, True
            if changed:
                st.since_change = 0

        # when inactive, never go above the 'standard' range of derive_signal
        effective = st.smoothed if st.active else min(st.smoothed, 0.29)
        if st.active:
            effective = max(effective, 0.5)  # active means a real adjustment
        return derive_signal(user_id, session_index, effective, modality_shares), changed

    def is_active(self, user_id):
        return self._state.get(user_id, _UserState()).active
