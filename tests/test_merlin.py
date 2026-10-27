
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

from ai.reasoning.merlin import compare_evidence_records


def test_detects_timestamp_mismatch_for_same_evidence_id():
    records = [
        {"evidence_id": "E001", "timestamp": "2026-09-25T10:00:00"},
        {"evidence_id": "E001", "timestamp": "2026-09-25T11:00:00"},
    ]

    result = compare_evidence_records(records)

    assert len(result["conflicts"]) == 1
    assert result["conflicts"][0]["type"] == "duplicate_id_timestamp_mismatch"


def test_same_evidence_id_and_timestamp_has_no_conflict():
    records = [
        {"evidence_id": "E001", "timestamp": "2026-09-25T10:00:00"},
        {"evidence_id": "E001", "timestamp": "2026-09-25T10:00:00"},
    ]

    result = compare_evidence_records(records)

    assert result["conflicts"] == []

