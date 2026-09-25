from pathlib import Path
import sys

import cv2
import torch

from preprocessing.torchvision_fix import apply_torchvision_fix

# Apply compatibility fix before importing Real-ESRGAN
apply_torchvision_fix()

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


# ============================================================
# FORensicGAN X
# Real-ESRGAN Enhancement Service
# ============================================================

MODEL_PATH = Path("models/RealESRGAN_x4plus.pth")

# Real-ESRGAN processing configuration
SCALE = 4
TILE_SIZE = 256
TILE_PAD = 10


# ============================================================
# MODEL CREATION
# ============================================================

def create_model():
    """
    Create the pretrained Real-ESRGAN x4plus model.

    No training or fine-tuning is performed.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Real-ESRGAN model not found: {MODEL_PATH}"
        )

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    upsampler = RealESRGANer(
        scale=SCALE,
        model_path=str(MODEL_PATH),
        model=model,
        tile=TILE_SIZE,
        tile_pad=TILE_PAD,
        pre_pad=0,
        half=False,
        device=device
    )

    return upsampler, device


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

def enhance_image(
    input_path: str,
    output_path: str,
    outscale: int = SCALE
):
    """
    Enhance an evidence image using pretrained Real-ESRGAN.

    IMPORTANT:
    The original evidence is never modified.

    Parameters
    ----------
    input_path:
        Path to the preprocessed evidence image.

    output_path:
        Path where the enhanced evidence will be saved.

    outscale:
        Output enlargement factor.
        Default = 4.
    """

    input_file = Path(input_path)
    output_file = Path(output_path)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input evidence not found: {input_file}"
        )

    image = cv2.imread(str(input_file))

    if image is None:
        raise ValueError(
            f"Unable to decode evidence image: {input_file}"
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    upsampler, device = create_model()

    input_height = int(image.shape[0])
    input_width = int(image.shape[1])

    expected_width = input_width * outscale
    expected_height = input_height * outscale

    # --------------------------------------------------------
    # Display processing information
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FORensicGAN X - REAL-ESRGAN ENHANCEMENT")
    print("=" * 60)

    print(f"Input       : {input_file}")

    print(
        f"Input Size  : "
        f"{input_width} x {input_height}"
    )

    print(f"Device      : {device}")

    print("Model       : RealESRGAN_x4plus")

    print(f"Scale       : {outscale}x")

    print(f"Tile Size   : {TILE_SIZE}")

    print(
        f"Expected    : "
        f"{expected_width} x {expected_height}"
    )

    print("Processing  : Started...")

    # --------------------------------------------------------
    # Real-ESRGAN inference
    # --------------------------------------------------------

    try:

        output, _ = upsampler.enhance(
            image,
            outscale=outscale
        )

    except RuntimeError as error:

        raise RuntimeError(
            "Real-ESRGAN inference failed. "
            "The image may be too large for available memory. "
            "Original evidence was not modified.\n"
            f"Original error: {error}"
        ) from error

    # --------------------------------------------------------
    # Save enhanced evidence
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_file),
        output
    )

    if not success:
        raise IOError(
            f"Failed to save enhanced image: {output_file}"
        )

    output_height = int(output.shape[0])
    output_width = int(output.shape[1])

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print(
        f"Output Size : "
        f"{output_width} x {output_height}"
    )

    print(f"Output      : {output_file}")

    print("Original    : PRESERVED")

    print("Status      : success")

    print("=" * 60)

    return {
        "input_path": str(input_file),
        "output_path": str(output_file),
        "device": str(device),
        "model": "RealESRGAN_x4plus",
        "scale": int(outscale),
        "input_width": input_width,
        "input_height": input_height,
        "output_width": output_width,
        "output_height": output_height,
        "status": "success"
    }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) not in (3, 4):

        print(
            "Usage:"
        )

        print(
            "python enhance.py "
            "<input_image> <output_image> [scale]"
        )

        print()
        print(
            "Example:"
        )

        print(
            "python enhance.py "
            "test_data/realesrgan_input.jpg "
            "storage/enhanced/E001_test.jpg"
        )

        sys.exit(1)

    input_image = sys.argv[1]

    output_image = sys.argv[2]

    if len(sys.argv) == 4:
        scale = int(sys.argv[3])
    else:
        scale = SCALE

    enhance_image(
        input_path=input_image,
        output_path=output_image,
        outscale=scale
    )