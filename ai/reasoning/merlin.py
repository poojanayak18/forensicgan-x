
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
