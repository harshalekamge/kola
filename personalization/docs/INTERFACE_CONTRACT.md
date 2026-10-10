# Interface Contract: Components 1-3 -> Component 4 (Personalization)

Project J26-DS-326. Component 4 consumes ONE kind of message: an **event**.
Each component sends one JSON event per user action. Everything stays on the
user's device (localhost only, no cloud).

## 1. Envelope (same for all components)

| Field | Type | Meaning |
|---|---|---|
| schema_version | string | always `"1.0"` |
| user_id | string | pseudonymous hash (e.g. `u_3f9a1c2b`), never a real name |
| session_id | string | one usage period, e.g. `S-u_3f9a1c2b-004` |
| timestamp | string | ISO-8601 UTC, e.g. `2026-10-12T09:15:03.120Z` |
| source | string | `voice`, `gesture` or `screen` |
| payload | object | fields below, depend on `source` |

## 2. Payloads

**Component 1, voice** (`source: "voice"`)
`latency_ms` (number 0-60000), `asr_confidence` (0-1), `language` (`si`/`en`/`singlish`),
`intent` (string), `was_corrected` (bool), `success` (bool)

**Component 2, gesture** (`source: "gesture"`)
`gesture_name` (string), `confidence` (0-1), `stability_score` (0-1, 1 = steady hand),
`was_retried` (bool), `success` (bool)

**Component 3, screen** (`source: "screen"`)
`action` (string), `element_found` (bool), `ocr_confidence` (0-1),
`click_success` (bool), `retries` (int 0-20)

## 3. Example

```json
{"schema_version":"1.0","user_id":"u_3f9a1c2b","session_id":"S-u_3f9a1c2b-004",
 "timestamp":"2026-10-12T09:15:03.120Z","source":"voice",
 "payload":{"latency_ms":1180.0,"asr_confidence":0.91,"language":"singlish",
            "intent":"open_app","was_corrected":false,"success":true}}
```

## 4. How to send

Preferred: HTTP `POST http://127.0.0.1:8000/events` with the JSON body.
Fallback: append one event per line to a `.jsonl` file (see `data/sample_events.jsonl`).
Invalid events are rejected with a message naming the broken field.

## 5. What you get back

`GET /signal/{user_id}` returns the personalization signal: gesture tolerance
multiplier (1.0-1.8), speech timeout multiplier (1.0-1.6), recommended modality,
UI mode (`standard`/`simplified`), confidence.

## 6. Ground rules

- Send an event for every action, successful or not (failures are the signal).
- Keep `session_id` constant while the app is in one usage period.
- Do not include raw audio, images or screen content.
