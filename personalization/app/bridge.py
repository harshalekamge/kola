"""Bridge: Context Store (SQLite) -> the SessionRecord objects the GRU code expects."""

from context_store import SessionRecord  # from the v2 prototype (root folder)


def load_records_by_user(store):
    by_user = {}
    for uid in store.list_users():
        recs = []
        for r in store.get_user_sessions(uid):
            recs.append(SessionRecord(
                user_id=uid, session_index=r["session_index"],
                click_error_rate=r["click_error_rate"],
                voice_command_latency_ms=r["voice_command_latency_ms"],
                correction_frequency=r["correction_frequency"],
                gesture_stability=r["gesture_stability"],
                modality_share_voice=r["modality_share_voice"],
                modality_share_gesture=r["modality_share_gesture"],
                modality_share_screen=r["modality_share_screen"],
                completion_accuracy=r["completion_accuracy"],
                state_decline_ground_truth=int(r["label_decline"]),
            ))
        by_user[uid] = recs
    return by_user
