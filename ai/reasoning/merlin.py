
from typing import Any


def evaluate_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """
    MERLIN v0: Evaluate one evidence record.

    This initial version checks the input structure and summarizes
    the available YOLO detection confidence values.

    The score is a rule-based technical indicator, NOT a probability
    of guilt or proof that evidence is authentic.
    """

    evidence_id = evidence.get("evidence_id", "UNKNOWN")
    detections = evidence.get("detections", [])

    if not isinstance(detections, list):
        raise ValueError("'detections' must be a list")

    valid_confidences = []

    for detection in detections:
        if not isinstance(detection, dict):
            continue

        confidence = detection.get("conf")

        if isinstance(confidence, (int, float)) and not isinstance(
            confidence, bool
        ):
            if 0.0 <= confidence <= 1.0:
                valid_confidences.append(float(confidence))

    if valid_confidences:
        score = round(
            sum(valid_confidences) / len(valid_confidences) * 100,
            2,
        )
        explanation = (
            f"{len(valid_confidences)} valid detection confidence value(s) "
            f"were found for evidence {evidence_id}. "
            "The score is the average YOLO detection confidence expressed "
            "as a percentage; it is not an overall forensic reliability score."
        )
    else:
        score = None
        explanation = (
            f"No valid detection confidence values were available "
            f"for evidence {evidence_id}; no score was calculated."
        )

    return {
        "score": score,
        "conflicts": [],
        "explanation": explanation,
    }

from datetime import datetime


def compare_evidence_records(evidence_records: list[dict]) -> dict:
    """
    MERLIN v0.2: Compare evidence records for metadata inconsistencies.

    Flags records that share an evidence_id but have different
    non-empty timestamps.

    This is a review flag, not proof of tampering.
    """

    conflicts = []
    records_by_id = {}

    for record in evidence_records:
        if not isinstance(record, dict):
            continue

        evidence_id = record.get("evidence_id")
        timestamp = record.get("timestamp")

        if not isinstance(evidence_id, str) or not evidence_id.strip():
            continue

        if not isinstance(timestamp, str) or not timestamp.strip():
            continue

        # Validate that the timestamp is parseable.
        try:
            datetime.fromisoformat(timestamp)
        except ValueError:
            continue

        records_by_id.setdefault(evidence_id, []).append(timestamp)

    for evidence_id, timestamps in records_by_id.items():
        unique_timestamps = sorted(set(timestamps))

        if len(unique_timestamps) > 1:
            conflicts.append({
                "type": "duplicate_id_timestamp_mismatch",
                "evidence_id": evidence_id,
                "timestamps": unique_timestamps,
                "explanation": (
                    f"Evidence ID {evidence_id} appears with multiple "
                    "timestamps. Review the source records to determine "
                    "whether this is expected."
                ),
            })

    return {
        "conflicts": conflicts,
        "explanation": (
            f"Compared {len(evidence_records)} evidence record(s); "
            f"found {len(conflicts)} metadata inconsistency flag(s)."
        ),
    }
