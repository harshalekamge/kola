"""
AI Assistant Engine (Decision-Level Multimodal Fusion) — MOCK
-----------------------------------------------------------------
Per the approved Topic Assessment Form's conceptual diagram, ALL 4
components communicate through a centralized "AI Assistant Engine" that
performs decision-level multimodal fusion, weighting/resolving inputs
from:
    Component 1 -> Voice/NLU intent
    Component 2 -> Gesture-recognized action
    Component 3 -> Screen-understanding detected UI element
    Component 4 -> Personalization signal (YOURS)

Since Components 1-3 are being built separately by teammates, this file
provides lightweight MOCK clients with the same message shape real
components would send, purely to demonstrate how your personalization
signal plugs into the fusion decision now. All processing is local
in-process (no network calls), consistent with the "no cloud, on-device
only" requirement in the approved form.
"""

from dataclasses import dataclass


@dataclass
class ModalityInput:
    source: str          # "voice" | "gesture" | "screen"
    action: str           # candidate action, e.g. "scroll_down"
    confidence: float      # 0-1 raw model confidence from that component


@dataclass
class FusionDecision:
    user_id: str
    chosen_action: str
    chosen_source: str
    final_confidence: float
    explanation: str


def fuse(user_id: str, inputs: list, personalization_signal) -> FusionDecision:
    """Very simple weighted-fusion mock: personalization signal re-weights
    each modality's confidence before picking the best candidate action.
    E.g. if decline probability is high and gesture tolerance was widened,
    gesture confidence gets a boost (we trust it more despite raw noise);
    the recommended_default_modality also gets a small confidence boost."""

    weights = {"voice": 1.0, "gesture": 1.0, "screen": 1.0}

    if personalization_signal.gesture_tolerance_multiplier > 1.0:
        weights["gesture"] *= personalization_signal.gesture_tolerance_multiplier
    if personalization_signal.speech_timeout_multiplier > 1.0:
        weights["voice"] *= personalization_signal.speech_timeout_multiplier
    weights[personalization_signal.recommended_default_modality] *= 1.1

    best = max(inputs, key=lambda inp: inp.confidence * weights.get(inp.source, 1.0))
    adjusted_conf = min(1.0, best.confidence * weights.get(best.source, 1.0))

    explanation = (
        f"Selected '{best.action}' from {best.source} "
        f"(raw_conf={best.confidence:.2f}, weight={weights[best.source]:.2f}, "
        f"decline_prob={personalization_signal.decline_probability:.2f})"
    )

    return FusionDecision(
        user_id=user_id,
        chosen_action=best.action,
        chosen_source=best.source,
        final_confidence=round(adjusted_conf, 3),
        explanation=explanation,
    )
