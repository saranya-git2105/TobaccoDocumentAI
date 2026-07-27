from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image


class ImageService:
    PDF_EXTENSIONS = {".pdf"}
    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".gif",
    }
    SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

    # Resize every upload to this width before OCR so token positions stay
    # consistent across JPG, PNG, PDF, and camera photos.
    CANONICAL_OCR_WIDTH = 1400
    CANONICAL_OCR_MAX_HEIGHT = 2200

    @classmethod
    def is_pdf(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.PDF_EXTENSIONS

    @classmethod
    def is_image(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.IMAGE_EXTENSIONS

    @classmethod
    def is_supported(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.SUPPORTED_EXTENSIONS

    @staticmethod
    def load_image_bgr(source_path: Path) -> Any:
        """Load an image as a BGR array, using Pillow for formats OpenCV cannot read."""
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)

        if image is not None:
            return image

        with Image.open(source_path) as pil_image:
            if getattr(pil_image, "n_frames", 1) > 1:
                pil_image.seek(0)

            rgb_array = np.array(pil_image.convert("RGB"))
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _render_pdf_page_to_bgr(page: pdfium.PdfPage, *, scale: float) -> Any:
        bitmap = page.render(scale=scale)
        image = bitmap.to_numpy()

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _render_pdf_first_page_bgr(
        source_path: Path,
        *,
        scale: float = 2.0,
    ) -> Any:
        doc = pdfium.PdfDocument(str(source_path))

        try:
            if len(doc) == 0:
                raise ValueError("The PDF file contains no pages.")

            return ImageService._render_pdf_page_to_bgr(
                doc[0],
                scale=scale,
            )
        finally:
            doc.close()

    @staticmethod
    def _rotate_bgr(image: Any, angle: float) -> Any:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (width / 2, height / 2),
            angle,
            1.0,
        )
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def _estimate_skew_angle(
        gray: Any,
        *,
        max_angle: float = 6.0,
        coarse_step: float = 0.5,
        fine_step: float = 0.1,
    ) -> float:
        """Find the rotation that makes printed text rows most horizontal.

        A photographed note is usually rotated by a fraction of a degree. That
        tilt makes the left-hand serial column drift vertically relative to the
        right-hand value columns, which breaks row assignment further down the
        page. Sharpening the horizontal projection profile recovers the angle.
        """
        scale = 800 / max(gray.shape[1], 1)

        if scale < 1.0:
            probe = cv2.resize(
                gray,
                (800, max(1, int(gray.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            probe = gray

        _, binary = cv2.threshold(
            probe,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        def sharpness(angle: float) -> float:
            if abs(angle) < 1e-6:
                rotated = binary
            else:
                height, width = binary.shape[:2]
                matrix = cv2.getRotationMatrix2D(
                    (width / 2, height / 2),
                    angle,
                    1.0,
                )
                rotated = cv2.warpAffine(
                    binary,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_NEAREST,
                    borderValue=0,
                )

            projection = rotated.sum(axis=1, dtype=np.float64)
            return float(np.var(np.diff(projection)))

        def search(center: float, radius: float, step: float) -> float:
            candidates = np.arange(
                center - radius,
                center + radius + (step / 2),
                step,
            )
            return max(candidates, key=sharpness)

        coarse = search(0.0, max_angle, coarse_step)
        fine = search(float(coarse), coarse_step, fine_step)
        return round(float(fine), 2)

    @staticmethod
    def _normalize_for_ocr(
        image: Any,
        *,
        canonical_width: int,
        maximum_height: int,
        deskew: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        """Apply in-memory OCR preprocessing. The source file is never modified."""
        original_height, original_width = image.shape[:2]
        scale = canonical_width / max(original_width, 1)
        resized_width = canonical_width
        resized_height = max(1, int(original_height * scale))

        if resized_height > maximum_height:
            scale = maximum_height / original_height
            resized_width = max(1, int(original_width * scale))
            resized_height = maximum_height

        resized = abs(scale - 1.0) > 0.01

        if resized:
            image = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=(
                    cv2.INTER_CUBIC
                    if scale > 1
                    else cv2.INTER_AREA
                ),
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        skew_angle = 0.0

        if deskew:
            skew_angle = ImageService._estimate_skew_angle(gray)

            if abs(skew_angle) >= 0.1:
                image = ImageService._rotate_bgr(image, skew_angle)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Mild contrast normalization helps PaddleOCR on several scan types.
        enhanced = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(gray)
        image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return image, {
            "originalWidth": original_width,
            "originalHeight": original_height,
            "processedWidth": resized_width,
            "processedHeight": resized_height,
            "scale": round(scale, 4),
            "canonicalWidth": canonical_width,
            "resized": resized,
            "claheApplied": True,
            "skewAngle": skew_angle,
        }

    @classmethod
    def prepare_document_array(
        cls,
        source_path: Path,
        *,
        canonical_width: int = CANONICAL_OCR_WIDTH,
        maximum_height: int = CANONICAL_OCR_MAX_HEIGHT,
        pdf_scale: float = 2.0,
    ) -> tuple[Any, dict[str, Any]]:
        """Load a document and return an in-memory OCR image.

        The uploaded file on disk is left unchanged. Only two transforms are
        applied to the in-memory copy:
        1. Resize to a fixed width so OCR geometry is stable across formats.
        2. CLAHE contrast normalization for reliable text detection.
        """
        if cls.is_pdf(source_path):
            image = cls._render_pdf_first_page_bgr(
                source_path,
                scale=pdf_scale,
            )
        else:
            image = cls.load_image_bgr(source_path)

        return cls._normalize_for_ocr(
            image,
            canonical_width=canonical_width,
            maximum_height=maximum_height,
        )
