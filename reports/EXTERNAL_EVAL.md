# Independent microphone evaluation protocol

Status: protocol only; no independent recordings or results are claimed.

Collect consented recordings outside ESC-50, including known classes, unseen classes, speech/music, overlapping sources and quiet/noisy rooms. Record device, sample rate, distance, room, source identity, annotator label and consent/license. Never publish private conversations.

Group all recordings of the same event/source/device session together. Reserve a final evaluation set before inspecting predictions. Use a separate development set to choose a margin threshold, then freeze it. The current 0.2 ranking-margin warning is an uncalibrated heuristic; it is not a probability or a validated out-of-distribution detector.

Report known-class accuracy and macro F1, accepted-prediction accuracy and coverage, unknown recall, and false rejection of known classes. Include counts and confusion examples; do not combine these numbers with ESC-50 results.

| Recording ID | True label / unknown | Device/session | Noise condition | Predicted label | Margin | Rejected | Consent/license |
|---|---|---|---|---|---|---|---|

Threshold selection set: pending. Frozen threshold: pending. Final evaluation: pending.
