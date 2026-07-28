from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
from paddleocr import PaddleOCR

from app.services.extraction_service import DeliveryNoteExtractor
from app.services.image_service import ImageService


class OcrService:
    _instance: PaddleOCR | None = None
    _initialization_lock = Lock()
    # PaddleOCR keeps mutable predictor state and is not safe to call from
    # multiple FastAPI worker threads at once. Serializing prediction prevents
    # overlapping uploads from dropping or combining table rows.
    _prediction_lock = Lock()
    _DET_LIMIT_SIDE_LEN = 1280
    _REFINEMENT_FIELDS = (
        "tbgr_number",
        "grower_name",
        "lot_number",
        "weight",
        "second_weight",
        "grade",
        "rate_per_kg",
        "bale_value",
    )
    _CRITICAL_ROW_FIELDS = (
        "tbgr_number",
        "grower_name",
        "date_of_purchase",
    )

    def __init__(self) -> None:
        self._ocr = self._get_ocr_instance()

    @classmethod
    def _get_ocr_instance(cls) -> PaddleOCR:
        if cls._instance is None:
            with cls._initialization_lock:
                if cls._instance is None:
                    print("Loading PaddleOCR models...")

                    start_time = time.perf_counter()

                    cls._instance = PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        enable_mkldnn=True,
                        cpu_threads=min(os.cpu_count() or 4, 8),
                        text_detection_model_name="PP-OCRv5_mobile_det",
                        text_recognition_model_name="PP-OCRv5_mobile_rec",
                        text_det_limit_side_len=cls._DET_LIMIT_SIDE_LEN,
                        text_det_limit_type="max",
                        text_recognition_batch_size=16,
                    )

                    elapsed = time.perf_counter() - start_time

                    print(
                        f"PaddleOCR models loaded in "
                        f"{elapsed:.2f} seconds."
                    )

        return cls._instance

    def _predict(self, input_data: Any) -> list[Any]:
        with self._prediction_lock:
            return list(
                self._ocr.predict(
                    input_data,
                    text_det_limit_side_len=self._DET_LIMIT_SIDE_LEN,
                )
            )

    def extract_text(self, image_path: Path) -> dict[str, Any]:
        if not image_path.exists():
            raise FileNotFoundError(
                f"File does not exist: {image_path}"
            )

        image, _ = ImageService.prepare_document_array(image_path)
        return self.extract_text_from_image(image)

    def extract_text_from_image(self, image: Any) -> dict[str, Any]:
        start_time = time.perf_counter()

        ocr_started = time.perf_counter()
        results = self._predict(image)
        items = self._results_to_items(results)
        ocr_elapsed = time.perf_counter() - ocr_started

        extract_started = time.perf_counter()
        document = DeliveryNoteExtractor.extract(items)
        extract_elapsed = time.perf_counter() - extract_started

        refine_elapsed = 0.0

        if self._document_needs_refinement(document):
            refine_started = time.perf_counter()
            refined_items = self._refine_table_rows(
                image,
                items,
                document,
            )

            if refined_items is not items:
                refined_document = DeliveryNoteExtractor.extract(
                    refined_items
                )
                document = self._merge_documents(
                    document,
                    refined_document,
                )

            refine_elapsed = time.perf_counter() - refine_started

        inference_time = time.perf_counter() - start_time
        print(
            "OCR completed in "
            f"{inference_time:.2f}s "
            f"(ocr={ocr_elapsed:.2f}s, "
            f"extract={extract_elapsed:.2f}s, "
            f"refine={refine_elapsed:.2f}s, "
            f"items={len(document.get('items', []))})"
        )
        return document

    def extract_text_from_pages(
        self,
        image_paths: list[Path],
    ) -> dict[str, Any]:
        if not image_paths:
            raise ValueError("At least one image path is required.")

        page_documents = [
            self.extract_text(image_path)
            for image_path in image_paths
        ]
        return self.merge_page_documents(page_documents)

    @staticmethod
    def merge_page_documents(
        pages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not pages:
            return {}

        if len(pages) == 1:
            return pages[0]

        merged = dict(pages[0])
        header_fields = (
            "delivery_note_number",
            "delivery_date",
            "buyer_name",
            "auction_platform_number",
            "auction_platform_name",
            "code_number",
            "printed_date",
        )

        for page in pages[1:]:
            for field in header_fields:
                if merged.get(field) in ("", None):
                    merged[field] = page.get(field, merged.get(field))

        merged_items: list[dict[str, Any]] = []

        for page in pages:
            merged_items.extend(page.get("items", []))

        merged["items"] = DeliveryNoteExtractor._postprocess_rows(
            merged_items
        )
        return merged

    @staticmethod
    def _results_to_items(
        results: Any,
        *,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        coordinate_scale: float = 1.0,
    ) -> list[dict[str, Any]]:
        extracted_items: list[dict[str, Any]] = []

        for result in results:
            result_json = result.json

            if callable(result_json):
                result_json = result_json()

            result_data = result_json.get("res", result_json)

            texts = result_data.get("rec_texts", [])
            scores = result_data.get("rec_scores", [])
            polygons = result_data.get("rec_polys", [])

            for index, text in enumerate(texts):
                cleaned_text = str(text).strip()

                if not cleaned_text:
                    continue

                confidence = (
                    float(scores[index])
                    if index < len(scores)
                    else 0.0
                )

                polygon = []

                if index < len(polygons):
                    current_polygon = polygons[index]

                    polygon = (
                        current_polygon.tolist()
                        if hasattr(current_polygon, "tolist")
                        else current_polygon
                    )

                adjusted_polygon = [
                    [
                        offset_x + (float(point[0]) / coordinate_scale),
                        offset_y + (float(point[1]) / coordinate_scale),
                    ]
                    for point in polygon
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ]

                extracted_items.append(
                    {
                        "text": cleaned_text,
                        "confidence": round(confidence, 4),
                        "boundingBox": adjusted_polygon,
                    }
                )

        return extracted_items

    @classmethod
    def _row_needs_refinement(cls, row: dict[str, Any]) -> bool:
        if any(
            row.get(field) in ("", None)
            for field in cls._CRITICAL_ROW_FIELDS
        ):
            return True

        if DeliveryNoteExtractor._grower_name_needs_refinement(
            str(row.get("grower_name", ""))
        ):
            return True

        missing_fields = sum(
            1
            for field in cls._REFINEMENT_FIELDS
            if row.get(field) in ("", None)
        )
        return missing_fields >= 2

    @classmethod
    def _document_needs_refinement(cls, document: dict[str, Any]) -> bool:
        rows = document.get("items", [])

        if len(rows) < 3:
            return not rows

        return any(cls._row_needs_refinement(row) for row in rows)

    def _refine_table_rows(
        self,
        image: Any,
        items: list[dict[str, Any]],
        document: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Re-OCR only incomplete table rows to recover missing cell values."""
        if image is None:
            return items

        incomplete_serials = {
            row["serial_number"]
            for row in document.get("items", [])
            if row.get("serial_number") is not None
            and self._row_needs_refinement(row)
        }

        if not incomplete_serials:
            return items

        height, width = image.shape[:2]
        header_bottom = self._find_table_header_bottom(items)
        serials = self._find_serial_anchors(
            items,
            width,
            header_bottom,
        )

        if len(serials) < 3:
            return items

        centers = self._complete_row_centers(serials)

        if len(centers) < 3:
            return items

        ordered_serials = [
            (serial, center)
            for serial, center in sorted(centers.items())
            if header_bottom < center < height
        ]

        if len(ordered_serials) < 3:
            return items

        row_gaps = [
            ordered_serials[index + 1][1] - ordered_serials[index][1]
            for index in range(len(ordered_serials) - 1)
            if ordered_serials[index + 1][1] > ordered_serials[index][1]
        ]
        typical_gap = (
            sorted(row_gaps)[len(row_gaps) // 2]
            if row_gaps
            else 30.0
        )
        minimum_row_height = max(int(typical_gap * 0.9), 20)
        row_bounds: list[tuple[int, int, int]] = []

        for index, (serial, center) in enumerate(ordered_serials):
            if serial not in incomplete_serials:
                continue

            previous_center = (
                ordered_serials[index - 1][1]
                if index > 0
                else max(
                    header_bottom,
                    center - (ordered_serials[1][1] - center),
                )
            )
            next_center = (
                ordered_serials[index + 1][1]
                if index + 1 < len(ordered_serials)
                else center + (center - ordered_serials[index - 1][1])
            )
            top = max(int((previous_center + center) / 2) + 1, 0)
            bottom = min(int((center + next_center) / 2) - 1, height)

            if bottom - top < minimum_row_height:
                # The page is deskewed before OCR, so the row sits centered on
                # its serial anchor rather than below it.
                half_height = minimum_row_height // 2
                row_center = int(center)
                top = max(row_center - half_height, int(header_bottom) + 1, 0)
                bottom = min(row_center + half_height, height)

            if bottom - top >= 8:
                row_bounds.append((serial, top, bottom))

        if not row_bounds:
            return items

        rows_by_serial = {
            row["serial_number"]: row
            for row in document.get("items", [])
            if row.get("serial_number") is not None
        }
        left_column_width = int(width * 0.58)
        prepared_crops: list[tuple[int, int, int, Any, float]] = []

        for serial, top, bottom in row_bounds:
            row_document = rows_by_serial.get(serial, {})
            missing_left_columns = any(
                row_document.get(field) in ("", None)
                for field in self._CRITICAL_ROW_FIELDS
            )
            needs_name_refinement = (
                DeliveryNoteExtractor._grower_name_needs_refinement(
                    str(row_document.get("grower_name", ""))
                )
            )
            missing_numeric_columns = any(
                row_document.get(field) in ("", None)
                for field in (
                    "weight",
                    "second_weight",
                    "grade",
                    "rate_per_kg",
                    "bale_value",
                )
            )
            use_full_row = missing_numeric_columns or not (
                missing_left_columns or needs_name_refinement
            )
            crop = (
                image[top:bottom, :]
                if use_full_row
                else image[top:bottom, :left_column_width]
            )
            # The TBGR digits are the smallest, densest glyphs on the row, so a
            # missing one warrants a larger crop than a missing numeric cell.
            row_scale = (
                3.0
                if missing_left_columns or needs_name_refinement
                else 2.0
                if missing_numeric_columns
                else (1.5 if width < 1600 else 1.0)
            )

            if row_scale > 1.0:
                crop = cv2.resize(
                    crop,
                    None,
                    fx=row_scale,
                    fy=row_scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            prepared_crops.append(
                (serial, top, bottom, crop, row_scale, use_full_row)
            )

        row_results = self._predict(
            [crop for _, _, _, crop, _, _ in prepared_crops]
        )

        if len(row_results) != len(prepared_crops):
            return items

        refined_items: list[dict[str, Any]] = []

        for (_, top, bottom, _, row_scale, use_full_row), row_result in zip(
            prepared_crops,
            row_results,
        ):
            row_items = self._results_to_items(
                [row_result],
                offset_y=float(top),
                coordinate_scale=row_scale,
            )
            # A left-column crop only ever holds the serial, the TBGR and the
            # name, so demanding as many tokens as a full-row crop would throw
            # away a good result.
            minimum_items = 3 if use_full_row else 2

            if len(row_items) >= minimum_items:
                refined_items.extend(row_items)
            else:
                refined_items.extend(
                    item
                    for item in items
                    if top <= self._item_center_y(item) < bottom
                )

        refined_bounds = [
            (top, bottom)
            for _, top, bottom, _, _, _ in prepared_crops
        ]
        outside_table_body = [
            item
            for item in items
            if not any(
                top <= self._item_center_y(item) < bottom
                for top, bottom in refined_bounds
            )
        ]
        return outside_table_body + refined_items

    @staticmethod
    def _merge_documents(
        original: dict[str, Any],
        refined: dict[str, Any],
    ) -> dict[str, Any]:
        refined_rows = {
            row.get("serial_number"): row
            for row in refined.get("items", [])
            if row.get("serial_number") is not None
        }
        merged_rows: list[dict[str, Any]] = []

        for row in original.get("items", []):
            serial_number = row.get("serial_number")
            fallback = refined_rows.get(serial_number, {})
            merged = dict(row)

            for field in (
                "tbgr_number",
                "grower_name",
                "date_of_purchase",
                "lot_number",
                "weight",
                "second_weight",
                "grade",
                "rate_per_kg",
                "bale_value",
            ):
                if merged.get(field) in ("", None):
                    merged[field] = fallback.get(field, merged.get(field))

            original_name = str(merged.get("grower_name", ""))
            refined_name = str(fallback.get("grower_name", ""))

            if DeliveryNoteExtractor._prefer_refined_grower_name(
                original_name,
                refined_name,
            ):
                merged["grower_name"] = refined_name

            merged_rows.append(merged)

        original["items"] = DeliveryNoteExtractor._postprocess_rows(
            merged_rows
        )
        # The quality figures were measured before refinement, so they have to
        # be recomputed or the response understates what was recovered.
        original["extraction_meta"] = (
            DeliveryNoteExtractor._build_extraction_meta(
                original["items"],
                layout_name=(
                    original.get("extraction_meta") or {}
                ).get("layout", "unknown"),
                totals=original.get("totals"),
            )
        )
        return original

    @staticmethod
    def _find_table_header_bottom(items: list[dict[str, Any]]) -> float:
        header_words = (
            "grower name",
            "purchase",
            "lot no",
            "weight",
            "grade",
            "bale value",
        )
        bottoms: list[float] = []

        for item in items:
            text = " ".join(str(item.get("text", "")).lower().split())

            if any(word in text for word in header_words):
                box = item.get("boundingBox", [])

                if box:
                    bottoms.append(max(float(point[1]) for point in box))

        return max(bottoms) if bottoms else 0.0

    @classmethod
    def _find_serial_anchors(
        cls,
        items: list[dict[str, Any]],
        page_width: int,
        header_bottom: float,
    ) -> list[tuple[int, float]]:
        anchors: list[tuple[int, float]] = []
        serial_right = page_width * 0.075

        for item in items:
            normalized = "".join(
                character
                for character in str(item.get("text", "")).lower()
                if character.isalnum()
            )
            box = item.get("boundingBox", [])

            if normalized.startswith("tbgr") and box:
                serial_right = min(
                    float(point[0])
                    for point in box
                )
                break

        for item in items:
            text = str(item.get("text", "")).strip()

            if not text.isdigit() or not 1 <= len(text) <= 3:
                continue

            box = item.get("boundingBox", [])

            if not box:
                continue

            center_x = sum(float(point[0]) for point in box) / len(box)
            center_y = sum(float(point[1]) for point in box) / len(box)
            number = int(text)

            if (
                center_x < serial_right
                and center_y > header_bottom
                and 1 <= number <= 999
            ):
                anchors.append((number, center_y))

        anchors.sort(key=lambda value: value[1])
        increasing: list[tuple[int, float]] = []

        for anchor in anchors:
            if not increasing or anchor[0] > increasing[-1][0]:
                increasing.append(anchor)

        return increasing

    @staticmethod
    def _complete_row_centers(
        serials: list[tuple[int, float]],
    ) -> dict[int, float]:
        centers = dict(serials)
        per_row_distances = [
            (right_y - left_y) / (right_number - left_number)
            for (left_number, left_y), (right_number, right_y)
            in zip(serials, serials[1:])
            if right_number > left_number and right_y > left_y
        ]

        if not per_row_distances:
            return centers

        per_row_distances.sort()
        typical_distance = per_row_distances[
            len(per_row_distances) // 2
        ]

        for (left_number, left_y), (right_number, right_y) in zip(
            serials,
            serials[1:],
        ):
            gap = right_number - left_number

            if gap <= 1:
                continue

            step = (right_y - left_y) / gap

            if not 0.5 * typical_distance <= step <= 1.5 * typical_distance:
                step = typical_distance

            for offset in range(1, gap):
                centers[left_number + offset] = left_y + (step * offset)

        return centers

    @staticmethod
    def _item_center_y(item: dict[str, Any]) -> float:
        box = item.get("boundingBox", [])

        if not box:
            return -1.0

        return sum(float(point[1]) for point in box) / len(box)