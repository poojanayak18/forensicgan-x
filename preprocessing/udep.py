from pathlib import Path

import cv2


# ============================================================
# UDEP
# Unified Digital Evidence Preprocessing
# ============================================================


def preprocess_image(input_path: str, output_path: str):
    """
    UDEP - Image preprocessing.

    Validates the input image and creates a working copy.

    IMPORTANT:
    The original evidence file is never modified.
    """

    input_file = Path(input_path)
    output_file = Path(output_path)

    # --------------------------------------------------------
    # 1. Check whether the evidence exists
    # --------------------------------------------------------

    if not input_file.exists():
        raise FileNotFoundError(
            f"Evidence image not found: {input_file}"
        )

    # --------------------------------------------------------
    # 2. Read the image
    # --------------------------------------------------------

    image = cv2.imread(str(input_file))

    if image is None:
        raise ValueError(
            f"Unable to decode image: {input_file}"
        )

    # --------------------------------------------------------
    # 3. Create output directory
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 4. Save working copy
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(output_file),
        image
    )

    if not success:
        raise IOError(
            f"Failed to save preprocessed image: {output_file}"
        )

    # --------------------------------------------------------
    # 5. Return metadata
    # --------------------------------------------------------

    return {
        "input_path": str(input_file),
        "output_path": str(output_file),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "channels": int(image.shape[2]),
        "status": "success"
    }


# ============================================================
# VIDEO PREPROCESSING
# ============================================================


def preprocess_video(input_path: str, output_dir: str):
    """
    UDEP - Video preprocessing.

    Validates the input video and extracts sequential
    frames as working copies.

    IMPORTANT:
    The original evidence video is never modified.
    """

    input_file = Path(input_path)
    output_directory = Path(output_dir)

    # --------------------------------------------------------
    # 1. Check whether the evidence exists
    # --------------------------------------------------------

    if not input_file.exists():
        raise FileNotFoundError(
            f"Evidence video not found: {input_file}"
        )

    # --------------------------------------------------------
    # 2. Open the video
    # --------------------------------------------------------

    video = cv2.VideoCapture(str(input_file))

    if not video.isOpened():
        raise ValueError(
            f"Unable to open video: {input_file}"
        )

    # --------------------------------------------------------
    # 3. Read video properties
    # --------------------------------------------------------

    fps = video.get(cv2.CAP_PROP_FPS)

    frame_count = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        video.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # --------------------------------------------------------
    # 4. Validate FPS
    # --------------------------------------------------------

    if fps <= 0:
        video.release()

        raise ValueError(
            "Invalid video FPS detected."
        )

    # --------------------------------------------------------
    # 5. Create output directory
    # --------------------------------------------------------

    frames_directory = (
        output_directory / "frames"
    )

    frames_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 6. Extract frames
    # --------------------------------------------------------

    extracted_frames = 0

    while True:

        success, frame = video.read()

        if not success:
            break

        frame_filename = (
            frames_directory
            / f"frame_{extracted_frames:06d}.jpg"
        )

        saved = cv2.imwrite(
            str(frame_filename),
            frame
        )

        if not saved:
            video.release()

            raise IOError(
                f"Failed to save frame: "
                f"{frame_filename}"
            )

        extracted_frames += 1

    # --------------------------------------------------------
    # 7. Release video
    # --------------------------------------------------------

    video.release()

    # --------------------------------------------------------
    # 8. Return metadata
    # --------------------------------------------------------

    return {
        "input_path": str(input_file),
        "output_directory": str(output_directory),
        "fps": round(fps, 2),
        "original_frame_count": frame_count,
        "extracted_frames": extracted_frames,
        "width": width,
        "height": height,
        "status": "success"
    }


# ============================================================
# TEST SECTION
# ============================================================


if __name__ == "__main__":

    # --------------------------------------------------------
    # IMAGE TEST
    # --------------------------------------------------------

    print("\n")

    image_result = preprocess_image(
        input_path="test_data/test_image.jpg",
        output_path="storage/preprocessed/E001.jpg"
    )

    print("=" * 60)
    print("FORensicGAN X - UDEP IMAGE TEST")
    print("=" * 60)

    print(
        f"Input      : "
        f"{image_result['input_path']}"
    )

    print(
        f"Output     : "
        f"{image_result['output_path']}"
    )

    print(
        f"Resolution : "
        f"{image_result['width']} "
        f"x "
        f"{image_result['height']}"
    )

    print(
        f"Channels   : "
        f"{image_result['channels']}"
    )

    print(
        f"Status     : "
        f"{image_result['status']}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # VIDEO TEST
    # --------------------------------------------------------

    print("\n")

    video_result = preprocess_video(
        input_path="test_data/test_video.mp4",
        output_dir="storage/preprocessed/video/E003"
    )

    print("=" * 60)
    print("FORensicGAN X - UDEP VIDEO TEST")
    print("=" * 60)

    print(
        f"Input Video      : "
        f"{video_result['input_path']}"
    )

    print(
        f"Resolution       : "
        f"{video_result['width']} "
        f"x "
        f"{video_result['height']}"
    )

    print(
        f"FPS              : "
        f"{video_result['fps']}"
    )

    print(
        f"Original Frames  : "
        f"{video_result['original_frame_count']}"
    )

    print(
        f"Extracted Frames : "
        f"{video_result['extracted_frames']}"
    )

    print(
        f"Output Directory : "
        f"{video_result['output_directory']}"
    )

    print(
        f"Status           : "
        f"{video_result['status']}"
    )

    print("=" * 60)