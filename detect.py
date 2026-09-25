import json
import sys
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_detection(image_path: str, evidence_id: str):
    """
    Run pretrained YOLOv8 detection on one evidence image.
    Returns the FORensicGAN X evidence JSON structure.
    """

    model = YOLO(MODEL_PATH)

    results = model(image_path, verbose=False)

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": class_name,
                "conf": round(confidence, 3),
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            })

    evidence = {
        "evidence_id": evidence_id,
        "type": "image",
        "detections": detections,
        "enhanced_path": None,
        "timestamp": datetime.now().isoformat(),
        "merlin_score": {
            "score": None,
            "conflicts": [],
            "explanation": None
        }
    }

    return evidence


def main():

    # ---------------------------------------------
    # Check command-line arguments
    # ---------------------------------------------

    if len(sys.argv) != 3:

        print("\nUsage:")
        print("python detect.py <image_path> <evidence_id>")

        print("\nExample:")
        print(
            "python detect.py "
            "test_data/test_image.jpg E001"
        )

        return

    image_path = sys.argv[1]
    evidence_id = sys.argv[2]

    # ---------------------------------------------
    # Validate image
    # ---------------------------------------------

    if not Path(image_path).exists():

        print(f"\nERROR: Image not found:")
        print(image_path)

        return

    # ---------------------------------------------
    # Run detection
    # ---------------------------------------------

    print("=" * 60)
    print("FORensicGAN X - DETECTION SERVICE")
    print("=" * 60)

    print(f"\nEvidence ID : {evidence_id}")
    print(f"Image       : {image_path}")

    evidence = run_detection(
        image_path=image_path,
        evidence_id=evidence_id
    )

    # ---------------------------------------------
    # Save JSON
    # ---------------------------------------------

    output_file = OUTPUT_DIR / f"{evidence_id}_detection.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(evidence, file, indent=4)

    # ---------------------------------------------
    # Display results
    # ---------------------------------------------

    print("\nDetection completed.")

    print(
        f"Objects detected: "
        f"{len(evidence['detections'])}"
    )

    for detection in evidence["detections"]:

        print(
            f"  {detection['class']:15s}"
            f" confidence={detection['conf']}"
        )

    print(f"\nJSON saved to: {output_file}")

    print("=" * 60)


if __name__ == "__main__":
    main()