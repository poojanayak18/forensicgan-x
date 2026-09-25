from pathlib import Path

import cv2


INPUT_PATH = Path("storage/preprocessed/E001.jpg")
OUTPUT_PATH = Path("test_data/realesrgan_input.jpg")


def create_test_crop():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input image not found: {INPUT_PATH}"
        )

    image = cv2.imread(str(INPUT_PATH))

    if image is None:
        raise ValueError(
            f"Unable to read image: {INPUT_PATH}"
        )

    height, width = image.shape[:2]

    # Take a central 512 x 512 crop.
    crop_size = 512

    center_x = width // 2
    center_y = height // 2

    x1 = max(center_x - crop_size // 2, 0)
    y1 = max(center_y - crop_size // 2, 0)

    x2 = x1 + crop_size
    y2 = y1 + crop_size

    crop = image[y1:y2, x1:x2]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(OUTPUT_PATH),
        crop
    )

    if not success:
        raise IOError(
            f"Failed to save crop: {OUTPUT_PATH}"
        )

    print("=" * 60)
    print("FORensicGAN X - Real-ESRGAN TEST CROP")
    print("=" * 60)
    print(f"Original : {width} x {height}")
    print(
        f"Crop     : "
        f"{crop.shape[1]} x {crop.shape[0]}"
    )
    print(f"Output   : {OUTPUT_PATH}")
    print("Status   : success")
    print("=" * 60)


if __name__ == "__main__":
    create_test_crop()