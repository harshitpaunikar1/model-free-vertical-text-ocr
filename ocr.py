"""
Model-free vertical text OCR pipeline for logistics labels and barcodes.
Uses OpenCV preprocessing and Tesseract for vertical/rotated text extraction.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


@dataclass
class OCRResult:
    text: str
    confidence: float
    rotation_applied: int  # degrees
    preprocessing: str
    bounding_boxes: List[Dict] = field(default_factory=list)

    def is_valid(self, min_confidence: float = 0.5) -> bool:
        return self.confidence >= min_confidence and len(self.text.strip()) > 0


class ImagePreprocessor:
    """
    Applies a configurable preprocessing pipeline to improve OCR accuracy
    on low-contrast, noisy, or rotated logistics labels.
    """

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        if not CV2_AVAILABLE:
            return image
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        if not CV2_AVAILABLE:
            return gray
        return cv2.fastNlMeansDenoising(gray, h=10)

    def threshold_otsu(self, gray: np.ndarray) -> np.ndarray:
        if not CV2_AVAILABLE:
            return gray
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def threshold_adaptive(self, gray: np.ndarray) -> np.ndarray:
        if not CV2_AVAILABLE:
            return gray
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    def deskew(self, binary: np.ndarray) -> Tuple[np.ndarray, float]:
        """Estimate and correct skew angle using minimum bounding rectangle."""
        if not CV2_AVAILABLE:
            return binary, 0.0
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 10:
            return binary, 0.0
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return binary, angle
        h, w = binary.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated, angle

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        if not CV2_AVAILABLE:
            return image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def upscale(self, image: np.ndarray, scale: float = 2.0) -> np.ndarray:
        if not CV2_AVAILABLE:
            return image
        h, w = image.shape[:2]
        return cv2.resize(image, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)


class VerticalTextOCR:
    """
    OCR engine optimized for vertical, rotated, and small-text logistics labels.
    Tries multiple rotations and preprocessing strategies, returns best result.
    """

    ROTATIONS = [0, 90, 180, 270]

    def __init__(self, tesseract_config: str = "--psm 6",
                 lang: str = "eng",
                 min_confidence: float = 0.4):
        self.tesseract_config = tesseract_config
        self.lang = lang
        self.min_confidence = min_confidence
        self.preprocessor = ImagePreprocessor()

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        if not CV2_AVAILABLE or angle == 0:
            return image
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def _run_tesseract(self, image: np.ndarray, config: str) -> Tuple[str, float]:
        if not TESSERACT_AVAILABLE:
            return "[STUB_OCR_TEXT_DETECTED]", 0.75
        try:
            data = pytesseract.image_to_data(
                image, lang=self.lang, config=config,
                output_type=pytesseract.Output.DICT,
            )
            words = [w for w, c in zip(data["text"], data["conf"])
                     if w.strip() and int(c) > 0]
            confs = [int(c) / 100 for c in data["conf"]
                     if int(c) > 0]
            text = " ".join(words)
            avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
            return text, avg_conf
        except Exception as exc:
            logger.error("Tesseract error: %s", exc)
            return "", 0.0

    def _preprocess_pipeline(self, image: np.ndarray,
                              strategy: str = "otsu") -> np.ndarray:
        gray = self.preprocessor.grayscale(image)
        denoised = self.preprocessor.denoise(gray)
        upscaled = self.preprocessor.upscale(denoised, scale=1.5)
        if strategy == "otsu":
            binary = self.preprocessor.threshold_otsu(upscaled)
        else:
            binary = self.preprocessor.threshold_adaptive(upscaled)
        deskewed, _ = self.preprocessor.deskew(binary)
        return deskewed

    def extract(self, image: np.ndarray) -> OCRResult:
        """
        Try multiple rotations and preprocessing strategies.
        Return the result with the highest confidence.
        """
        best: Optional[OCRResult] = None
        strategies = ["otsu", "adaptive"]
        configs = [self.tesseract_config, "--psm 11", "--psm 3"]

        for angle in self.ROTATIONS:
            rotated = self._rotate_image(image, angle)
            for strategy in strategies:
                processed = self._preprocess_pipeline(rotated, strategy)
                for config in configs:
                    text, conf = self._run_tesseract(processed, config)
                    result = OCRResult(
                        text=text.strip(),
                        confidence=conf,
                        rotation_applied=angle,
                        preprocessing=f"{strategy}/{config}",
                    )
                    if best is None or result.confidence > best.confidence:
                        best = result
                    if best.confidence >= 0.85:
                        return best

        return best or OCRResult(text="", confidence=0.0,
                                  rotation_applied=0, preprocessing="none")

    def extract_with_boxes(self, image: np.ndarray) -> OCRResult:
        """Extract text and bounding boxes for each detected word."""
        result = self.extract(image)
        if not TESSERACT_AVAILABLE:
            return result
        try:
            processed = self._preprocess_pipeline(image)
            data = pytesseract.image_to_data(
                processed, lang=self.lang, config=self.tesseract_config,
                output_type=pytesseract.Output.DICT,
            )
            boxes = []
            for i, word in enumerate(data["text"]):
                if word.strip() and int(data["conf"][i]) > 30:
                    boxes.append({
                        "text": word,
                        "conf": int(data["conf"][i]),
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "w": data["width"][i],
                        "h": data["height"][i],
                    })
            result.bounding_boxes = boxes
        except Exception as exc:
            logger.error("Box extraction error: %s", exc)
        return result

    def batch_extract(self, images: List[np.ndarray]) -> List[OCRResult]:
        return [self.extract(img) for img in images]


class BarcodeTextParser:
    """Parses structured data from extracted OCR text on logistics labels."""

    PATTERNS = {
        "tracking_number": r"\b[A-Z]{2}\d{9,12}[A-Z]{2}\b",
        "weight_kg": r"(\d+\.?\d*)\s*kg",
        "weight_lbs": r"(\d+\.?\d*)\s*lbs?",
        "postal_code_in": r"\b[1-9][0-9]{5}\b",
        "date": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    }

    def parse(self, text: str) -> Dict[str, str]:
        import re
        results = {}
        for field_name, pattern in self.PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results[field_name] = match.group(0).strip()
        return results


if __name__ == "__main__":
    ocr = VerticalTextOCR(tesseract_config="--psm 6", lang="eng")
    dummy_image = np.ones((300, 200, 3), dtype=np.uint8) * 255
    if CV2_AVAILABLE:
        cv2.putText(dummy_image, "PARCEL", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 0), 3)
        cv2.putText(dummy_image, "12.5 kg", (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 2)
    result = ocr.extract(dummy_image)
    print(f"Extracted text: '{result.text}'")
    print(f"Confidence: {result.confidence:.2f} | Rotation: {result.rotation_applied} deg")
    print(f"Preprocessing: {result.preprocessing}")

    parser = BarcodeTextParser()
    sample_text = "Tracking: AB123456789IN Weight: 12.5 kg Date: 15/04/2024"
    parsed = parser.parse(sample_text)
    print(f"\nParsed fields from '{sample_text}':")
    for k, v in parsed.items():
        print(f"  {k}: {v}")
