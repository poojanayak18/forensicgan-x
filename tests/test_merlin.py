import pytest

from ai.reasoning.merlin import (
    evaluate_evidence,
    compare_evidence_records,
    generate_merlin_report,
)


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


def test_detects_timestamp_mismatch_for_same_evidence_id():
    records = [
        {"evidence_id": "E001", "timestamp": "2026-09-25T10:00:00"},
        {"evidence_id": "E001", "timestamp": "2026-09-25T11:00:00"},
    ]

    result = compare_evidence_records(records)

    assert len(result["conflicts"]) == 1
    assert (
        result["conflicts"][0]["type"]
        == "duplicate_id_timestamp_mismatch"
    )


def test_same_evidence_id_and_timestamp_has_no_conflict():
    records = [
        {"evidence_id": "E001", "timestamp": "2026-09-25T10:00:00"},
        {"evidence_id": "E001", "timestamp": "2026-09-25T10:00:00"},
    ]

    result = compare_evidence_records(records)

    assert result["conflicts"] == []


def test_generate_merlin_report():
    records = [
        {
            "evidence_id": "E001",
            "timestamp": "2026-09-25T10:00:00",
            "detections": [
                {"class": "person", "conf": 0.9}
            ],
        },
        {
            "evidence_id": "E001",
            "timestamp": "2026-09-25T11:00:00",
            "detections": [
                {"class": "car", "conf": 0.7}
            ],
        },
    ]

    report = generate_merlin_report(records)

    assert report["total_records"] == 2
    assert len(report["evidence_summaries"]) == 2
    assert len(report["conflicts"]) == 1
    assert report["evidence_summaries"][0]["score"] == 90.0
    assert report["evidence_summaries"][1]["score"] == 70.0