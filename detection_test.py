from ultralytics import YOLO


def main():
    # Load pretrained YOLOv8
    model = YOLO("yolov8n.pt")

    # Run detection on our test image
    results = model("test_data/test_image.jpg")

    print("\n" + "=" * 60)
    print("FORensicGAN X - YOLOv8 DETECTION TEST")
    print("=" * 60)

    for result in results:
        print(f"\nImage: test_data/test_image.jpg")

        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected.")
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(
                f"Detected: {class_name:15s} "
                f"Confidence: {confidence:.2f} "
                f"BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()