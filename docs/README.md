# FORensicGAN X - AI Service

## Purpose

The AI Service is responsible for the perception and evidence-processing
components of FORensicGAN X.

## Current Components

- UDEP preprocessing
- Real-ESRGAN image enhancement
- YOLOv8 object detection
- EXIF metadata extraction
- SHA-256 evidence integrity hashing
- FastAPI service

## Evidence JSON Contract

The AI service produces structured evidence information using the following
fields:

### evidence_id
Unique identifier assigned to the evidence.

Example:
`E001`

### type
Type of evidence.

Current supported values:
- `image`
- `video`

### detections
Objects detected by YOLOv8.

Each detection contains:

- `class` - detected COCO object class
- `conf` - detection confidence
- `bbox` - bounding box coordinates `[x1, y1, x2, y2]`

### enhanced_path
Path of the derived enhanced evidence image.

### timestamp
Timestamp associated with the evidence record.

### merlin_score
Reserved structure for the MERLIN evidence-fusion stage.

Contains:

- `score`
- `conflicts`
- `explanation`

## Example

See `evidence_schema.json`.

## Integration Flow

Evidence
→ UDEP
→ Real-ESRGAN
→ YOLOv8
→ Structured Evidence JSON
→ Backend / Knowledge Graph

## Important

Original evidence must be preserved.

Enhanced images and AI-generated outputs are derived artifacts and should
not replace the original evidence.