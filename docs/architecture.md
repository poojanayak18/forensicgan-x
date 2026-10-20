
# ForensicGAN++ — System Architecture

## 1. Project Overview

ForensicGAN++ is an AI-assisted digital forensic investigation
framework designed to process digital evidence, detect and track
entities, analyse relationships between evidence items, and support
the reconstruction of event timelines.

The system follows a modular architecture that separates visual
perception from evidence reasoning and knowledge representation.

## 2. System Components

### 2.1 Evidence Perception

The perception component is responsible for evidence preprocessing,
image enhancement, object detection, and extraction of relevant
metadata.

Planned technologies:
- Python
- OpenCV
- Real-ESRGAN
- YOLOv8

### 2.2 Tracking and Evidence Reasoning

The reasoning component is responsible for associating tracked
entities with evidence, combining available evidence signals, and
identifying inconsistencies.

Planned technologies:
- ByteTrack
- Python
- MERLIN v0

MERLIN v0 is the proposed evidence-fusion and conflict-detection
component. Its inputs, scoring rules, thresholds, and output schema
must be documented and validated before implementation.

### 2.3 Knowledge Graph

Neo4j will be used to represent evidence items, detected entities,
video frames, and events through explicit relationships.

The final graph schema will be established after reviewing the
shared evidence JSON contract.

### 2.4 Timeline Reconstruction

The timeline component will organise supported events by timestamp
and preserve links to their associated evidence.

### 2.5 Backend and Integration

FastAPI will expose the relevant processing and reasoning
operations through API endpoints. JSON will be used to exchange
structured information between system components.

## 3. MERLIN v0 Design

MERLIN v0 will initially use weighted evidence fusion.

Provisional scoring model:

S = sum(w_i * s_i)

where:
- s_i represents an agreed evidence signal score.
- w_i represents the corresponding weight.
- The weights must be documented and sum to 1.

Conflict detection will identify significant disagreements between
comparable evidence signals. The conflict threshold and its effect
on the final score must be defined and tested.

This scoring model is a design proposal, not a validated forensic
probability or proof of guilt.

## 4. Shared JSON Contract

Status: Pending agreement with the perception-track teammate.

The contract must define:
- Evidence identifier and source
- Timestamp and frame information, where applicable
- Detection class, confidence, and bounding box
- Track identifier, where applicable
- Metadata and provenance
- MERLIN output and conflict information

Exact field names, data types, units, and required fields must be
confirmed against the actual perception and tracking outputs before
implementation.

## 5. Implementation Principles

- Keep components modular and independently testable.
- Use explicit schemas and type hints.
- Validate incoming JSON data.
- Preserve evidence identifiers and provenance.
- Record errors without silently discarding evidence.
- Document assumptions, limitations, and scoring rules.
- Do not treat model-generated scores as definitive forensic proof.

## 6. Day 1 Status

- [x] Python environment configured
- [x] Project folders created
- [x] Neo4j instance created and connection tested
- [ ] Shared JSON contract agreed
- [ ] Knowledge graph schema finalised
- [ ] MERLIN pseudocode reviewed
