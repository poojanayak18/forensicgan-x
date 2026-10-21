
import pytest

from ai.reasoning.merlin import evaluate_evidence


def test_average_detection_confidence():
    evidence = {
        "evidence_id": "E001",
        "detections": [
            {"class": "person", "conf": 0.9},
            {"class": "car", "conf": 0.7},
        ],
    }

    result = evaluate_evidence(evidence)

    assert result["score"] == 80.0
    assert result["conflicts"] == []
    assert "average YOLO detection confidence" in result["explanation"]


def test_no_detections():
    evidence = {
        "evidence_id": "E002",
        "detections": [],
    }

    result = evaluate_evidence(evidence)

    assert result["score"] is None
    assert result["conflicts"] == []
    assert "No valid detection confidence values" in result["explanation"]


def test_invalid_confidence_is_ignored():
    evidence = {
        "evidence_id": "E003",
        "detections": [
            {"class": "person", "conf": 0.8},
            {"class": "car", "conf": 1.5},
            {"class": "dog", "conf": "unknown"},
        ],
    }

    result = evaluate_evidence(evidence)

    assert result["score"] == 80.0


def test_invalid_detections_structure():
    evidence = {
        "evidence_id": "E004",
        "detections": None,
    }

    with pytest.raises(ValueError):
        evaluate_evidence(evidence)
