from pathlib import Path
import hashlib
import json
import sys

from PIL import Image
from PIL.ExifTags import TAGS


# ============================================================
# FORensicGAN X
# Metadata Extraction Service
# ============================================================


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of the original evidence file.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def extract_exif(file_path: Path) -> dict:
    """
    Extract available EXIF metadata from an image.

    If EXIF data is unavailable, an empty dictionary is returned.
    """

    exif_data = {}

    try:

        with Image.open(file_path) as image:

            exif = image.getexif()

            if not exif:
                return exif_data

            for tag_id, value in exif.items():

                tag_name = TAGS.get(
                    tag_id,
                    str(tag_id)
                )

                # Convert values that cannot be serialized
                # directly into strings.
                try:
                    json.dumps(value)
                    exif_data[tag_name] = value

                except (TypeError, ValueError):
                    exif_data[tag_name] = str(value)

    except Exception as error:

        exif_data["extraction_error"] = str(error)

    return exif_data


def extract_metadata(
    evidence_path: str,
    evidence_id: str
) -> dict:

    """
    Extract forensic metadata from evidence.

    The original evidence file is never modified.
    """

    file_path = Path(evidence_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"Evidence file not found: {file_path}"
        )

    # --------------------------------------------------------
    # SHA-256
    # --------------------------------------------------------

    sha256_hash = calculate_sha256(file_path)

    # --------------------------------------------------------
    # Basic file information
    # --------------------------------------------------------

    file_size = file_path.stat().st_size

    # --------------------------------------------------------
    # EXIF
    # --------------------------------------------------------

    exif_data = extract_exif(file_path)

    # --------------------------------------------------------
    # Metadata result
    # --------------------------------------------------------

    metadata = {

        "evidence_id": evidence_id,

        "file": {
            "name": file_path.name,
            "path": str(file_path),
            "size_bytes": file_size
        },

        "integrity": {
            "algorithm": "SHA-256",
            "hash": sha256_hash
        },

        "exif": exif_data
    }

    return metadata


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 3:

        print()
        print("Usage:")
        print(
            "python metadata/extract_metadata.py "
            "<evidence_file> <evidence_id>"
        )

        print()
        print("Example:")
        print(
            "python metadata/extract_metadata.py "
            "test_data/test_image.jpg E001"
        )

        sys.exit(1)

    evidence_file = sys.argv[1]
    evidence_id = sys.argv[2]

    try:

        result = extract_metadata(
            evidence_file,
            evidence_id
        )

        print()
        print("=" * 60)
        print("FORensicGAN X - METADATA EXTRACTION")
        print("=" * 60)

        print(f"Evidence ID : {result['evidence_id']}")
        print(f"File        : {result['file']['name']}")
        print(
            f"Size        : "
            f"{result['file']['size_bytes']} bytes"
        )

        print()
        print("SHA-256:")
        print(result["integrity"]["hash"])

        print()
        print(
            f"EXIF fields : "
            f"{len(result['exif'])}"
        )

        print()
        print("Status      : success")

        print("=" * 60)

        # ----------------------------------------------------
        # Save metadata JSON
        # ----------------------------------------------------

        output_directory = Path("results")
        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_directory
            / f"{evidence_id}_metadata.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
                default=str
            )

        print()
        print(f"Metadata JSON: {output_file}")

    except Exception as error:

        print()
        print("Status      : FAILED")
        print(f"Error       : {error}")

        sys.exit(1)