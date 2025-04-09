# Model-Free Vertical Text OCR

> **Domain:** Logistics

## Overview

Many shipping labels, pallet stickers, and carton marks are printed vertically or in stacked blocks, making conventional OCR miss characters and slow downstream scans. The business needed a fast, training-free way to read such text across varied lighting and color schemes. Pain points included poor accuracy on rotated text, costly model retraining for each layout, manual rework to key in unread items. Without a reliable solution, operations faced delays at receiving and dispatch, higher exception handling costs, reduced traceability cascading into SLA breaches and penalties.

## Approach

- Audited large image inflow to map label styles, text orientations, color usage, ambient lighting variability across sites
- Normalized inputs by converting to monochrome; applied adaptive thresholding and morphological filters to stabilise edges and suppress noise
- Detected candidate glyph regions; used EMNIST character references for lightweight matching without custom model training
- Clustered and merged overlapping bounding boxes; grouped stacked lines into coherent blocks using spatial heuristics
- Re-ordered vertical characters into horizontal sequences; passed cleaned crops to Tesseract for final text extraction
- Built validation loop with spot-checks, confusion analysis, error tagging; iterated rules and thresholds until target quality was met

## Skills & Technologies

- OpenCV (Python)
- Image Preprocessing
- EMNIST Character Matching
- Tesseract OCR
- Bounding Box Clustering
- Morphological Operations
- Data Pipeline Design
- Error Analysis & QA
- Technical Documentation
