"""
Personalization Signal
------------------------
Component 4 (YOURS), per the approved Topic Assessment Form:

  "...produces a personalization signal - a confidence-scored adjustment
   recommendation - that the voice, screen, and gesture modules can query
   and apply locally (for example, widening gesture tolerance or extending
   speech-parsing timeout)... adapts UI shortcuts, default modality, and
   interaction flow per user over time."

Given a predicted decline probability for a user's upcoming session, this
module produces the concrete, queryable signal that Components 1-3 (and
the AI Assistant Engine) would read.
"""

from dataclasses import dataclass


@dataclass
class PersonalizationSignal:
    user_id: str
    session_index: int
    decline_probability: float          # 0-1, from drift_predictor.py
    recommended_default_modality: str   # "voice" | "gesture" | "screen"
    gesture_tolerance_multiplier: float  # 1.0 default, up to 1.8
    speech_timeout_multiplier: float     # 1.0 default, up to 1.6
    ui_shortcut_mode: str                # "standard" | "simplified"
    confidence: float                    # how confident this recommendation is


def derive_signal(user_id: str, session_index: int, decline_probability: float,
                   last_session_modality_shares: dict) -> PersonalizationSignal:
    """last_session_modality_shares: {'voice':...,'gesture':...,'screen':...}
    used to recommend which modality to default to going forward."""

    recommended_modality = max(last_session_modality_shares, key=last_session_modality_shares.get)

    if decline_probability < 0.3:
        return PersonalizationSignal(
            user_id=user_id, session_index=session_index,
            decline_probability=round(decline_probability, 3),
            recommended_default_modality=recommended_modality,
            gesture_tolerance_multiplier=1.0,
            speech_timeout_multiplier=1.0,
            ui_shortcut_mode="standard",
            confidence=round(1 - decline_probability, 3),
        )

    # scale adjustments with confidence in the decline prediction
    gesture_mult = 1.0 + decline_probability * 0.8    # up to 1.8x tolerance
    speech_mult = 1.0 + decline_probability * 0.6      # up to 1.6x timeout
    shortcut_mode = "simplified" if decline_probability >= 0.6 else "standard"

    return PersonalizationSignal(
        user_id=user_id, session_index=session_index,
        decline_probability=round(decline_probability, 3),
        recommended_default_modality=recommended_modality,
        gesture_tolerance_multiplier=round(gesture_mult, 2),
        speech_timeout_multiplier=round(speech_mult, 2),
        ui_shortcut_mode=shortcut_mode,
        confidence=round(decline_probability, 3),
    )
