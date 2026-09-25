from pathlib import Path
import json
import shutil

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException

from metadata.extract_metadata import extract_metadata
from detect import run_detection


# ============================================================
# FORensicGAN X
# AI Service API
# ============================================================

app = FastAPI(
    title="FORensicGAN X - AI Service",
    description="AI perception service for digital evidence processing",
    version="1.0.0"
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIRECTORY = Path("storage/api_uploads")
RESULTS_DIRECTORY = Path("results")

UPLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "FORensicGAN X",
        "service": "AI Service",
        "status": "running",
        "python_environment": "working",
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "forensicgan-x-ai-service"
    }


# ============================================================
# METADATA EXTRACTION ENDPOINT
# ============================================================

@app.post("/metadata")
async def metadata_endpoint(
    evidence_id: str,
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Unsupported image format"
        )

    # --------------------------------------------------------
    # Save uploaded evidence
    # --------------------------------------------------------

    output_file = (
        UPLOAD_DIRECTORY
        / f"{evidence_id}{extension}"
    )

    try:

        with open(output_file, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ----------------------------------------------------
        # Extract metadata
        # ----------------------------------------------------

        metadata = extract_metadata(
            str(output_file),
            evidence_id
        )

        # ----------------------------------------------------
        # Save metadata JSON
        # ----------------------------------------------------

        metadata_file = (
            RESULTS_DIRECTORY
            / f"{evidence_id}_metadata.json"
        )

        with open(
            metadata_file,
            "w",
            encoding="utf-8"
        ) as json_file:

            json.dump(
                metadata,
                json_file,
                indent=2,
                default=str
            )

        return {
            "status": "success",
            "message": "Metadata extracted successfully",
            "metadata": metadata
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Metadata extraction failed: {error}"
        )


# ============================================================
# YOLOv8 DETECTION ENDPOINT
# ============================================================

@app.post("/detect")
async def detect_endpoint(
    evidence_id: str,
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Unsupported image format"
        )

    # --------------------------------------------------------
    # Save uploaded evidence
    # --------------------------------------------------------

    input_file = (
        UPLOAD_DIRECTORY
        / f"{evidence_id}{extension}"
    )

    try:

        with open(input_file, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ----------------------------------------------------
        # Run existing YOLOv8 detector
        # ----------------------------------------------------

        evidence = run_detection(
            image_path=str(input_file),
            evidence_id=evidence_id
        )

        # ----------------------------------------------------
        # Save detection JSON
        # ----------------------------------------------------

        output_file = (
            RESULTS_DIRECTORY
            / f"{evidence_id}_detection.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as json_file:

            json.dump(
                evidence,
                json_file,
                indent=2
            )

        return {
            "status": "success",
            "message": "YOLOv8 detection completed",
            "evidence": evidence
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {error}"
        )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("FORensicGAN X - AI SERVICE")
    print("=" * 50)

    print("Python environment: Working")
    print(f"OpenCV version    : {cv2.__version__}")
    print(f"NumPy version     : {np.__version__}")

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    print(f"Test image shape  : {image.shape}")

    print("\nAI service basic environment is working!")
    print("=" * 50)