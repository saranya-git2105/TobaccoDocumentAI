from __future__ import annotations

import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
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

    def extract_text_from_image(
        self,
        image: Any,
        *,
        native_items: list[dict[str, Any]] | None = None,
        source: str = "image",
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        ocr_started = time.perf_counter()
        results = self._predict(image)
        items = self._results_to_items(results)
        ocr_elapsed = time.perf_counter() - ocr_started

        extract_started = time.perf_counter()
        document = DeliveryNoteExtractor.extract(items, source=source)
        protected_date_serials: set[int] = set()
        protected_grade_serials: set[int] = set()

        if native_items:
            native_document = DeliveryNoteExtractor.extract(
                native_items,
                source=source,
            )
            native_rows = native_document.get("items", [])
            ocr_rows = document.get("items", [])

            # A usable native text layer is more accurate than raster OCR, but
            # image-only scans and incomplete PDF overlays must still fall back
            # to PaddleOCR.
            if len(native_rows) >= max(3, int(len(ocr_rows) * 0.7)):
                document = self._merge_document_sources(
                    native_document,
                    document,
                )
                protected_date_serials = {
                    int(row["serial_number"])
                    for row in native_rows
                    if row.get("serial_number") is not None
                    and self._is_valid_purchase_date(
                        str(row.get("date_of_purchase", ""))
                    )
                }
                protected_grade_serials = {
                    int(row["serial_number"])
                    for row in native_rows
                    if row.get("serial_number") is not None
                    and self._is_valid_grade(str(row.get("grade", "")))
                }

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
                    refined_items,
                    source=source,
                )
                document = self._merge_documents(
                    document,
                    refined_document,
                )

            refine_elapsed = time.perf_counter() - refine_started

        date_refine_started = time.perf_counter()
        document = self._refine_purchase_date_cells(
            image,
            items,
            document,
            protected_serials=protected_date_serials,
        )
        refine_elapsed += time.perf_counter() - date_refine_started

        grade_refine_started = time.perf_counter()
        document = self._refine_grade_cells(
            image,
            items,
            document,
            protected_serials=protected_grade_serials,
        )
        refine_elapsed += time.perf_counter() - grade_refine_started

        tbgr_refine_started = time.perf_counter()
        document = self._refine_missing_tbgr_cells(
            image,
            items,
            document,
        )
        refine_elapsed += time.perf_counter() - tbgr_refine_started

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

        if not cls._is_valid_grade(str(row.get("grade", ""))):
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

    @staticmethod
    def _is_valid_grade(value: str) -> bool:
        normalized = DeliveryNoteExtractor._normalize_grade(value)
        return bool(
            re.fullmatch(
                r"(?:F\d{1,2}M?|[LXH]\d[LO]?|BNL)",
                normalized,
                re.IGNORECASE,
            )
        )

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
        centers = self._find_row_centers(
            items,
            document,
            width,
            header_bottom,
        )

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
        prepared_crops: list[
            tuple[int, int, int, Any, float, bool]
        ] = []

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
    def _is_valid_purchase_date(value: str) -> bool:
        try:
            datetime.strptime(value, "%d/%m/%y")
        except ValueError:
            return False

        return True

    @staticmethod
    def _find_purchase_date_column_bounds(
        items: list[dict[str, Any]],
        width: int,
        layout_name: str,
    ) -> tuple[int, int]:
        candidates: list[tuple[float, float]] = []
        recognized_date_boxes: list[tuple[float, float]] = []

        for item in items:
            normalized = " ".join(
                str(item.get("text", "")).lower().split()
            )
            box = item.get("boundingBox", [])

            if (
                box
                and re.search(
                    r"\d{1,2}/\d{1,2}/\d{2,4}",
                    normalized,
                )
            ):
                recognized_date_boxes.append(
                    (
                        min(float(point[0]) for point in box),
                        max(float(point[0]) for point in box),
                    )
                )

            if "purchase" not in normalized:
                continue

            if not box:
                continue

            left = min(float(point[0]) for point in box)
            right = max(float(point[0]) for point in box)
            candidates.append(((left + right) / 2, right - left))

        if recognized_date_boxes:
            left = median(box[0] for box in recognized_date_boxes)
            right = median(box[1] for box in recognized_date_boxes)
            margin = width * 0.008
            return (
                max(int(left - margin), 0),
                min(int(right + margin), width),
            )

        expected_center = width * (
            0.335 if layout_name == "wide" else 0.49
        )

        if candidates:
            center, header_width = min(
                candidates,
                key=lambda value: abs(value[0] - expected_center),
            )
            half_width = min(
                max(width * 0.042, header_width * 0.8),
                width * 0.06,
            )
        else:
            center = expected_center
            half_width = width * 0.047

        return (
            max(int(center - half_width), 0),
            min(int(center + half_width), width),
        )

    @classmethod
    def _select_purchase_date_candidate(
        cls,
        variant_results: list[Any],
        *,
        delivery_date: str,
    ) -> str:
        delivery: datetime | None = None

        try:
            delivery = datetime.strptime(delivery_date, "%d/%m/%Y")
        except ValueError:
            pass

        candidates: list[tuple[str, float]] = []

        for result in variant_results:
            result_items = cls._results_to_items([result])
            snippets = [
                (
                    str(item.get("text", "")),
                    float(item.get("confidence", 0.0)),
                )
                for item in result_items
            ]

            if result_items:
                ordered = sorted(
                    result_items,
                    key=lambda item: min(
                        float(point[0])
                        for point in item.get("boundingBox", [[0, 0]])
                    ),
                )
                snippets.append(
                    (
                        "".join(str(item.get("text", "")) for item in ordered),
                        sum(
                            float(item.get("confidence", 0.0))
                            for item in ordered
                        )
                        / len(ordered),
                    )
                )

            variant_candidates: list[tuple[str, float]] = []

            for text, confidence in snippets:
                normalized = (
                    text.upper()
                    .replace("O", "0")
                    .replace("Q", "0")
                    .replace("I", "1")
                    .replace("L", "1")
                    .replace(" ", "")
                )
                value, _ = DeliveryNoteExtractor._extract_purchase_date(
                    normalized
                )

                parsed: datetime | None = None

                if cls._is_valid_purchase_date(value):
                    parsed = datetime.strptime(value, "%d/%m/%y")

                # The delivery year is strong context for a damaged final date
                # digit. Keep a complete plausible prior-year reading, but
                # repair a future year or a one-digit year from the cell OCR.
                partial_match = re.search(
                    r"(\d{1,2})[/.](\d{1,2})[/.](\d{1,2})",
                    normalized,
                )

                if (
                    delivery is not None
                    and partial_match is not None
                    and (
                        parsed is None
                        or parsed > delivery
                    )
                ):
                    day_text, month_text, year_text = partial_match.groups()

                    if len(year_text) == 1 or parsed is not None:
                        try:
                            repaired = datetime(
                                delivery.year,
                                int(month_text),
                                int(day_text),
                            )
                        except ValueError:
                            repaired = None

                        if (
                            repaired is not None
                            and delivery - timedelta(days=366)
                            <= repaired
                            <= delivery
                        ):
                            parsed = repaired
                            value = repaired.strftime("%d/%m/%y")

                if parsed is None:
                    continue

                if (
                    delivery is not None
                    and not delivery - timedelta(days=366)
                    <= parsed
                    <= delivery
                ):
                    continue

                variant_candidates.append((value, confidence))

            if variant_candidates:
                candidates.append(
                    max(variant_candidates, key=lambda value: value[1])
                )

        if not candidates:
            return ""

        votes = Counter(value for value, _ in candidates)
        value, count = votes.most_common(1)[0]

        if count >= 2:
            return value

        ranked = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        best_value, best_confidence = ranked[0]
        next_confidence = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_confidence >= 0.9 and best_confidence - next_confidence >= 0.12:
            return best_value

        return ""

    def _refine_purchase_date_cells(
        self,
        image: Any,
        items: list[dict[str, Any]],
        document: dict[str, Any],
        *,
        protected_serials: set[int],
    ) -> dict[str, Any]:
        """Read every raster-only purchase date from its exact table cell."""
        if image is None:
            return document

        height, width = image.shape[:2]
        header_bottom = self._find_table_header_bottom(items)
        centers = self._find_row_centers(
            items,
            document,
            width,
            header_bottom,
        )

        if len(centers) < 2:
            return document

        ordered_centers = sorted(centers.items(), key=lambda value: value[1])
        layout_name = str(
            (document.get("extraction_meta") or {}).get("layout", "")
        )
        left, right = self._find_purchase_date_column_bounds(
            items,
            width,
            layout_name,
        )
        row_serials = {
            int(row["serial_number"])
            for row in document.get("items", [])
            if row.get("serial_number") is not None
        }
        prepared: list[tuple[int, list[Any]]] = []

        for index, (serial, center) in enumerate(ordered_centers):
            if serial not in row_serials or serial in protected_serials:
                continue

            previous_center = (
                ordered_centers[index - 1][1]
                if index > 0
                else center - (ordered_centers[index + 1][1] - center)
            )
            next_center = (
                ordered_centers[index + 1][1]
                if index + 1 < len(ordered_centers)
                else center + (center - ordered_centers[index - 1][1])
            )
            top = max(int((previous_center + center) / 2) + 2, 0)
            bottom = min(int((center + next_center) / 2) - 2, height)

            if bottom - top < 6 or right - left < 8:
                continue

            crop = image[top:bottom, left:right]
            enlarged = cv2.resize(
                crop,
                None,
                fx=3.0,
                fy=3.0,
                interpolation=cv2.INTER_CUBIC,
            )
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            prepared.append(
                (
                    serial,
                    [
                        enlarged,
                        cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
                    ],
                )
            )

        if not prepared:
            return document

        flat_crops = [
            crop
            for _, variants in prepared
            for crop in variants
        ]
        predictions = self._predict(flat_crops)

        if len(predictions) != len(flat_crops):
            return document

        recovered: dict[int, str] = {}
        prediction_index = 0

        for serial, variants in prepared:
            variant_results = predictions[
                prediction_index : prediction_index + len(variants)
            ]
            prediction_index += len(variants)
            candidate = self._select_purchase_date_candidate(
                variant_results,
                delivery_date=str(document.get("delivery_date", "")),
            )

            if candidate:
                recovered[serial] = candidate

        changed = False
        delivery: datetime | None = None

        try:
            delivery = datetime.strptime(
                str(document.get("delivery_date", "")),
                "%d/%m/%Y",
            )
        except ValueError:
            pass

        if delivery is not None:
            for row in document.get("items", []):
                serial_number = row.get("serial_number")

                if serial_number in protected_serials:
                    continue

                value = str(row.get("date_of_purchase", ""))

                try:
                    parsed = datetime.strptime(value, "%d/%m/%y")
                except ValueError:
                    continue

                if not delivery - timedelta(days=366) <= parsed <= delivery:
                    row["date_of_purchase"] = ""
                    changed = True

        for row in document.get("items", []):
            serial_number = row.get("serial_number")

            if serial_number in recovered:
                row["date_of_purchase"] = recovered[int(serial_number)]
                changed = True

        changed = (
            self._fill_missing_dates_from_identity(
                document.get("items", [])
            )
            or changed
        )

        if not changed:
            return document

        document["items"] = DeliveryNoteExtractor._postprocess_rows(
            document.get("items", [])
        )
        document["extraction_meta"] = (
            DeliveryNoteExtractor._build_extraction_meta(
                document["items"],
                layout_name=layout_name or "unknown",
                totals=document.get("totals"),
                source=str(
                    (document.get("extraction_meta") or {}).get("source")
                    or "image"
                ),
            )
        )
        return document

    @classmethod
    def _fill_missing_dates_from_identity(
        cls,
        rows: list[dict[str, Any]],
    ) -> bool:
        """Fill only blanks backed by repeated TBGR or grower identity."""
        dates_by_tbgr: dict[str, list[str]] = {}
        dates_by_grower: dict[str, list[str]] = {}

        for row in rows:
            value = str(row.get("date_of_purchase", "")).strip()

            if not cls._is_valid_purchase_date(value):
                continue

            tbgr_number = str(row.get("tbgr_number", "")).strip()
            grower_key = re.sub(
                r"[^A-Za-z]",
                "",
                str(row.get("grower_name", "")),
            ).upper()

            if re.fullmatch(r"\d{8}", tbgr_number):
                dates_by_tbgr.setdefault(tbgr_number, []).append(value)

            if len(grower_key) >= 8:
                dates_by_grower.setdefault(grower_key, []).append(value)

        def unanimous(values: list[str]) -> str:
            return (
                values[0]
                if len(values) >= 2 and len(set(values)) == 1
                else ""
            )

        changed = False

        for row in rows:
            if str(row.get("date_of_purchase", "")).strip():
                continue

            tbgr_number = str(row.get("tbgr_number", "")).strip()
            grower_key = re.sub(
                r"[^A-Za-z]",
                "",
                str(row.get("grower_name", "")),
            ).upper()
            replacement = unanimous(
                dates_by_tbgr.get(tbgr_number, [])
            ) or unanimous(dates_by_grower.get(grower_key, []))

            if replacement:
                row["date_of_purchase"] = replacement
                changed = True

        return changed

    @classmethod
    def _find_grade_column_bounds(
        cls,
        items: list[dict[str, Any]],
        width: int,
        layout_name: str,
    ) -> tuple[int, int]:
        recognized: list[tuple[float, float]] = []
        header_center: float | None = None
        expected_center = width * (
            0.535 if layout_name == "wide" else 0.777
        )

        for item in items:
            text = str(item.get("text", "")).strip()
            box = item.get("boundingBox", [])

            if not box:
                continue

            left = min(float(point[0]) for point in box)
            right = max(float(point[0]) for point in box)
            center = (left + right) / 2

            if (
                cls._is_valid_grade(text)
                and abs(center - expected_center) <= width * 0.08
            ):
                recognized.append((left, right))
            elif "grade" in text.lower():
                header_center = center

        if recognized:
            left = median(box[0] for box in recognized)
            right = median(box[1] for box in recognized)
            margin = width * 0.012
            return (
                max(int(left - margin), 0),
                min(int(right + margin), width),
            )

        center = header_center or expected_center
        half_width = width * 0.035
        return (
            max(int(center - half_width), 0),
            min(int(center + half_width), width),
        )

    @classmethod
    def _grade_candidate_from_text(
        cls,
        text: str,
        *,
        dominant_family: str,
    ) -> str:
        normalized = DeliveryNoteExtractor._normalize_grade(text)

        if cls._is_valid_grade(normalized):
            return normalized

        compact = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        compact = (
            compact.replace("Q", "0")
            .replace("D", "0")
            .replace("Z", "2")
        )

        if dominant_family == "L":
            match = re.fullmatch(r"([123])([0OL])", compact)

            if match:
                suffix = "L" if match.group(2) == "L" else "O"
                return f"L{match.group(1)}{suffix}"

            damaged_l_suffix = re.fullmatch(r"([123])A", compact)

            if damaged_l_suffix:
                return f"L{damaged_l_suffix.group(1)}L"

        return ""

    @classmethod
    def _select_grade_candidate(
        cls,
        variant_results: list[Any],
        *,
        dominant_family: str,
    ) -> str:
        candidates: list[tuple[str, float]] = []

        for result in variant_results:
            result_items = cls._results_to_items([result])
            snippets = [
                (
                    str(item.get("text", "")),
                    float(item.get("confidence", 0.0)),
                )
                for item in result_items
            ]

            if result_items:
                ordered = sorted(
                    result_items,
                    key=lambda item: min(
                        float(point[0])
                        for point in item.get("boundingBox", [[0, 0]])
                    ),
                )
                snippets.append(
                    (
                        "".join(str(item.get("text", "")) for item in ordered),
                        sum(
                            float(item.get("confidence", 0.0))
                            for item in ordered
                        )
                        / len(ordered),
                    )
                )

            variant_candidates = [
                (candidate, confidence)
                for text, confidence in snippets
                if (
                    candidate := cls._grade_candidate_from_text(
                        text,
                        dominant_family=dominant_family,
                    )
                )
            ]

            if variant_candidates:
                candidates.append(
                    max(variant_candidates, key=lambda value: value[1])
                )

        if not candidates:
            return ""

        votes = Counter(value for value, _ in candidates)
        value, count = votes.most_common(1)[0]

        if count >= 2:
            return value

        value, confidence = max(candidates, key=lambda candidate: candidate[1])
        return value if confidence >= 0.88 else ""

    def _refine_grade_cells(
        self,
        image: Any,
        items: list[dict[str, Any]],
        document: dict[str, Any],
        *,
        protected_serials: set[int],
    ) -> dict[str, Any]:
        """Re-read grades from narrow cells with table-line suppression."""
        if image is None:
            return document

        height, width = image.shape[:2]
        header_bottom = self._find_table_header_bottom(items)
        centers = self._find_row_centers(
            items,
            document,
            width,
            header_bottom,
        )

        if len(centers) < 2:
            return document

        ordered_centers = sorted(centers.items(), key=lambda value: value[1])
        layout_name = str(
            (document.get("extraction_meta") or {}).get("layout", "")
        )
        left, right = self._find_grade_column_bounds(
            items,
            width,
            layout_name,
        )
        rows = document.get("items", [])
        row_serials = {
            int(row["serial_number"])
            for row in rows
            if row.get("serial_number") is not None
        }
        families = Counter(
            DeliveryNoteExtractor._normalize_grade(
                str(row.get("grade", ""))
            )[:1]
            for row in rows
            if self._is_valid_grade(str(row.get("grade", "")))
        )
        dominant_family = families.most_common(1)[0][0] if families else ""
        prepared: list[tuple[int, list[Any]]] = []

        for index, (serial, center) in enumerate(ordered_centers):
            if serial not in row_serials or serial in protected_serials:
                continue

            previous_center = (
                ordered_centers[index - 1][1]
                if index > 0
                else center - (ordered_centers[index + 1][1] - center)
            )
            next_center = (
                ordered_centers[index + 1][1]
                if index + 1 < len(ordered_centers)
                else center + (center - ordered_centers[index - 1][1])
            )
            top = max(int((previous_center + center) / 2) + 2, 0)
            bottom = min(int((center + next_center) / 2) - 2, height)

            if bottom - top < 6 or right - left < 8:
                continue

            crop = image[top:bottom, left:right]
            enlarged = cv2.resize(
                crop,
                None,
                fx=4.0,
                fy=4.0,
                interpolation=cv2.INTER_CUBIC,
            )
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            inverted = cv2.bitwise_not(binary)
            horizontal = cv2.morphologyEx(
                inverted,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (max(int(inverted.shape[1] * 0.7), 3), 1),
                ),
            )
            vertical = cv2.morphologyEx(
                inverted,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (1, max(int(inverted.shape[0] * 0.9), 3)),
                ),
            )
            cleaned = cv2.bitwise_not(
                cv2.subtract(
                    cv2.subtract(inverted, horizontal),
                    vertical,
                )
            )
            prepared.append(
                (
                    serial,
                    [
                        enlarged,
                        cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR),
                    ],
                )
            )

        if not prepared:
            return document

        flat_crops = [
            crop
            for _, variants in prepared
            for crop in variants
        ]
        predictions = self._predict(flat_crops)

        if len(predictions) != len(flat_crops):
            return document

        recovered: dict[int, str] = {}
        prediction_index = 0

        for serial, variants in prepared:
            variant_results = predictions[
                prediction_index : prediction_index + len(variants)
            ]
            prediction_index += len(variants)
            candidate = self._select_grade_candidate(
                variant_results,
                dominant_family=dominant_family,
            )

            if candidate:
                recovered[serial] = candidate

        changed = False
        for row in rows:
            serial_number = row.get("serial_number")

            if serial_number in recovered:
                row["grade"] = recovered[int(serial_number)]
                changed = True
                continue

            if not self._is_valid_grade(str(row.get("grade", ""))):
                repaired = self._grade_candidate_from_text(
                    str(row.get("grade", "")),
                    dominant_family=dominant_family,
                )

                if repaired:
                    row["grade"] = repaired
                    changed = True

        changed = self._fill_invalid_grades_from_tbgr(rows) or changed

        if not changed:
            return document

        document["items"] = DeliveryNoteExtractor._postprocess_rows(rows)
        document["extraction_meta"] = (
            DeliveryNoteExtractor._build_extraction_meta(
                document["items"],
                layout_name=layout_name or "unknown",
                totals=document.get("totals"),
                source=str(
                    (document.get("extraction_meta") or {}).get("source")
                    or "image"
                ),
            )
        )
        return document

    @classmethod
    def _fill_invalid_grades_from_tbgr(
        cls,
        rows: list[dict[str, Any]],
    ) -> bool:
        grades_by_tbgr: dict[str, list[str]] = {}

        for row in rows:
            tbgr_number = str(row.get("tbgr_number", "")).strip()
            grade = DeliveryNoteExtractor._normalize_grade(
                str(row.get("grade", ""))
            )

            if (
                re.fullmatch(r"\d{8}", tbgr_number)
                and cls._is_valid_grade(grade)
            ):
                grades_by_tbgr.setdefault(tbgr_number, []).append(grade)

        agreed = {
            tbgr_number: grades[0]
            for tbgr_number, grades in grades_by_tbgr.items()
            if len(grades) >= 2 and len(set(grades)) == 1
        }
        changed = False

        for row in rows:
            if cls._is_valid_grade(str(row.get("grade", ""))):
                continue

            replacement = agreed.get(str(row.get("tbgr_number", "")).strip())

            if replacement:
                row["grade"] = replacement
                changed = True

        return changed

    def _refine_missing_tbgr_cells(
        self,
        image: Any,
        items: list[dict[str, Any]],
        document: dict[str, Any],
    ) -> dict[str, Any]:
        """Re-OCR missing TBGR cells and clear prefix outliers."""
        rows = document.get("items", [])
        prefix_counts = Counter(
            str(row.get("tbgr_number", "")).strip()[:3]
            for row in rows
            if re.fullmatch(
                r"\d{8}",
                str(row.get("tbgr_number", "")).strip(),
            )
        )
        dominant_count = max(prefix_counts.values(), default=0)
        reliable_prefixes = [
            prefix
            for prefix, count in prefix_counts.most_common()
            if count >= 2 and count * 3 >= dominant_count
        ]
        target_serials = {
            int(row["serial_number"])
            for row in rows
            if row.get("serial_number") is not None
            and (
                not re.fullmatch(
                    r"\d{8}",
                    str(row.get("tbgr_number", "")).strip(),
                )
                or (
                    reliable_prefixes
                    and not str(row.get("tbgr_number", "")).startswith(
                        tuple(reliable_prefixes)
                    )
                )
            )
        }

        if image is None or not target_serials:
            return document

        height, width = image.shape[:2]
        header_bottom = self._find_table_header_bottom(items)
        centers = self._find_row_centers(
            items,
            document,
            width,
            header_bottom,
        )

        if len(centers) < 2:
            return document

        left, right = self._find_tbgr_column_bounds(
            items,
            width,
            str(
                (document.get("extraction_meta") or {}).get(
                    "layout",
                    "",
                )
            ),
        )
        ordered_centers = sorted(centers.items())
        prepared: list[tuple[int, list[Any]]] = []

        for index, (serial, center) in enumerate(ordered_centers):
            if serial not in target_serials:
                continue

            previous_center = (
                ordered_centers[index - 1][1]
                if index > 0
                else center - (
                    ordered_centers[index + 1][1] - center
                )
            )
            next_center = (
                ordered_centers[index + 1][1]
                if index + 1 < len(ordered_centers)
                else center + (
                    center - ordered_centers[index - 1][1]
                )
            )
            top = max(int((previous_center + center) / 2), 0)
            bottom = min(int((center + next_center) / 2), height)

            if bottom - top < 6 or right - left < 8:
                continue

            crop = image[top:bottom, left:right]
            enlarged = cv2.resize(
                crop,
                None,
                fx=4.0,
                fy=4.0,
                interpolation=cv2.INTER_CUBIC,
            )
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            prepared.append((serial, [enlarged, binary_bgr]))

        if not prepared:
            return document

        flat_crops = [
            crop
            for _, variants in prepared
            for crop in variants
        ]
        predictions = self._predict(flat_crops)

        if len(predictions) != len(flat_crops):
            return document

        prefixes = reliable_prefixes or DeliveryNoteExtractor._collect_tbgr_prefixes(
            rows
        )
        recovered: dict[int, str] = {}
        prediction_index = 0

        for serial, variants in prepared:
            variant_results = predictions[
                prediction_index : prediction_index + len(variants)
            ]
            prediction_index += len(variants)
            candidate = self._select_tbgr_candidate(
                variant_results,
                prefixes,
            )

            if candidate:
                recovered[serial] = candidate

        if not recovered:
            return document

        for row in document.get("items", []):
            serial_number = row.get("serial_number")

            if serial_number in recovered:
                row["tbgr_number"] = recovered[int(serial_number)]

        document["items"] = DeliveryNoteExtractor._postprocess_rows(
            document.get("items", [])
        )
        self._agree_tbgr_numbers_by_grower(document["items"])
        document["extraction_meta"] = (
            DeliveryNoteExtractor._build_extraction_meta(
                document["items"],
                layout_name=(
                    document.get("extraction_meta") or {}
                ).get("layout", "unknown"),
                totals=document.get("totals"),
                source=str(
                    (document.get("extraction_meta") or {}).get("source")
                    or "image"
                ),
            )
        )
        return document

    @staticmethod
    def _agree_tbgr_numbers_by_grower(
        rows: list[dict[str, Any]],
    ) -> None:
        """Correct a TBGR outlier when one grower has a strict local majority."""
        numbers_by_grower: dict[str, list[str]] = {}

        for row in rows:
            grower_key = re.sub(
                r"[^A-Za-z]",
                "",
                str(row.get("grower_name", "")),
            ).upper()
            tbgr_number = str(row.get("tbgr_number", "")).strip()

            if (
                len(grower_key) >= 8
                and re.fullmatch(r"\d{8}", tbgr_number)
            ):
                numbers_by_grower.setdefault(grower_key, []).append(
                    tbgr_number
                )

        agreed: dict[str, str] = {}

        for grower_key, numbers in numbers_by_grower.items():
            number, votes = Counter(numbers).most_common(1)[0]

            if votes >= 2 and votes * 2 > len(numbers):
                agreed[grower_key] = number

        for row in rows:
            grower_key = re.sub(
                r"[^A-Za-z]",
                "",
                str(row.get("grower_name", "")),
            ).upper()

            if grower_key in agreed:
                row["tbgr_number"] = agreed[grower_key]

    @staticmethod
    def _find_tbgr_column_bounds(
        items: list[dict[str, Any]],
        width: int,
        layout_name: str,
    ) -> tuple[int, int]:
        tbgr_box: list[Any] | None = None
        grower_box: list[Any] | None = None

        for item in items:
            text = "".join(
                character
                for character in str(item.get("text", "")).lower()
                if character.isalnum()
            )
            box = item.get("boundingBox", [])

            if not box:
                continue

            if tbgr_box is None and "tbgr" in text:
                tbgr_box = box
            if grower_box is None and (
                "growername" in text
                or "nameofthegrower" in text
            ):
                grower_box = box

        if tbgr_box and grower_box and tbgr_box is not grower_box:
            tbgr_left = min(float(point[0]) for point in tbgr_box)
            grower_left = min(float(point[0]) for point in grower_box)
            left = max(int(tbgr_left - (width * 0.015)), 0)
            right = min(int(grower_left - (width * 0.005)), width)

            if right - left >= width * 0.035:
                return left, right

        if layout_name == "wide":
            return int(width * 0.045), int(width * 0.17)

        return int(width * 0.08), int(width * 0.23)

    @classmethod
    def _select_tbgr_candidate(
        cls,
        results: list[Any],
        prefixes: list[str],
    ) -> str:
        candidates: list[tuple[float, str]] = []

        for item in cls._results_to_items(results):
            text = str(item.get("text", ""))
            confidence = float(item.get("confidence", 0.0))

            for digit_group in re.findall(r"\d+", text):
                if len(digit_group) == 8:
                    candidates.append((confidence, digit_group))
                elif 8 < len(digit_group) <= 10:
                    # A tight crop can join the one- or two-digit serial to
                    # the TBGR number; the registration number is the last
                    # eight digits in that reading.
                    candidates.append((confidence - 0.02, digit_group[-8:]))

            recovered = DeliveryNoteExtractor._recover_tbgr_from_text(
                text,
                prefixes,
            )

            if recovered:
                candidates.append((confidence - 0.05, recovered))

        if not candidates:
            return ""

        valid_prefixes = set(prefixes)
        prefixed = [
            candidate
            for candidate in candidates
            if not valid_prefixes or candidate[1][:3] in valid_prefixes
        ]

        if valid_prefixes and not prefixed:
            return ""

        return max(prefixed or candidates, key=lambda value: value[0])[1]

    @staticmethod
    def _merge_document_sources(
        primary: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge a high-confidence source with OCR as its field fallback."""
        merged = dict(primary)
        header_fields = (
            "delivery_note_number",
            "delivery_date",
            "buyer_name",
            "auction_platform_number",
            "auction_platform_name",
            "code_number",
            "printed_date",
        )

        for field in header_fields:
            if merged.get(field) in ("", None):
                merged[field] = fallback.get(field, merged.get(field))

        primary_rows = {
            row.get("serial_number"): row
            for row in primary.get("items", [])
            if row.get("serial_number") is not None
        }
        fallback_rows = {
            row.get("serial_number"): row
            for row in fallback.get("items", [])
            if row.get("serial_number") is not None
        }
        row_fields = (
            "tbgr_number",
            "grower_name",
            "grower_name_ocr",
            "date_of_purchase",
            "lot_number",
            "weight",
            "second_weight",
            "grade",
            "rate_per_kg",
            "bale_value",
        )
        merged_rows: list[dict[str, Any]] = []

        for serial_number in sorted(set(primary_rows) | set(fallback_rows)):
            row = dict(primary_rows.get(serial_number, {}))
            fallback_row = fallback_rows.get(serial_number, {})

            if not row:
                row = dict(fallback_row)
            else:
                for field in row_fields:
                    if row.get(field) in ("", None):
                        row[field] = fallback_row.get(field, row.get(field))

            merged_rows.append(row)

        merged["items"] = DeliveryNoteExtractor._postprocess_rows(merged_rows)

        if merged.get("totals") in ({}, None):
            merged["totals"] = fallback.get("totals", merged.get("totals"))

        merged["extraction_meta"] = DeliveryNoteExtractor._build_extraction_meta(
            merged["items"],
            layout_name=(
                merged.get("extraction_meta") or {}
            ).get(
                "layout",
                (fallback.get("extraction_meta") or {}).get(
                    "layout",
                    "unknown",
                ),
            ),
            totals=merged.get("totals"),
            source=str(
                (merged.get("extraction_meta") or {}).get("source")
                or (fallback.get("extraction_meta") or {}).get("source")
                or "image"
            ),
        )
        return merged

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
                "grower_name_ocr",
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
                refined_ocr = str(fallback.get("grower_name_ocr", "")).strip()
                if refined_ocr:
                    merged["grower_name_ocr"] = refined_ocr

            refined_grade = str(fallback.get("grade", ""))

            if (
                not OcrService._is_valid_grade(
                    str(merged.get("grade", ""))
                )
                and OcrService._is_valid_grade(refined_grade)
            ):
                merged["grade"] = DeliveryNoteExtractor._normalize_grade(
                    refined_grade
                )

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
                source=str(
                    (original.get("extraction_meta") or {}).get("source")
                    or "image"
                ),
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

            if (
                "total" not in text
                and any(word in text for word in header_words)
            ):
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

        def collect(right_boundary: float) -> list[tuple[int, float]]:
            found: list[tuple[int, float]] = []

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
                    center_x < right_boundary
                    and center_y > header_bottom
                    and 1 <= number <= 60
                ):
                    found.append((number, center_y))

            found.sort(key=lambda value: value[1])

            if not found:
                return []

            # Use the longest increasing sequence instead of a greedy scan.
            # One misread serial such as 30 in the middle of rows 1..18 must
            # not prevent every valid anchor below it from being considered.
            paths: list[list[tuple[int, float]]] = []

            for index, anchor in enumerate(found):
                best_prefix: list[tuple[int, float]] = []

                for previous_index in range(index):
                    if (
                        found[previous_index][0] < anchor[0]
                        and len(paths[previous_index]) > len(best_prefix)
                    ):
                        best_prefix = paths[previous_index]

                paths.append([*best_prefix, anchor])

            return max(paths, key=len)

        anchors = collect(serial_right)

        # Some compact PDFs place the whole table farther from the page's left
        # edge and PaddleOCR misses the TBGR header used to measure the serial
        # boundary. A wider retry still excludes every four/eight-digit table
        # value while recovering the one/two-digit serial column.
        if len(anchors) < 2:
            anchors = collect(page_width * 0.20)

        return anchors

    @classmethod
    def _find_row_centers(
        cls,
        items: list[dict[str, Any]],
        document: dict[str, Any],
        page_width: int,
        header_bottom: float,
    ) -> dict[int, float]:
        """Combine serial anchors with the more reliable lot-number column."""
        serials = cls._find_serial_anchors(
            items,
            page_width,
            header_bottom,
        )
        centers = cls._complete_row_centers(serials)
        serial_by_lot = {
            str(row.get("lot_number", "")).strip(): int(
                row["serial_number"]
            )
            for row in document.get("items", [])
            if row.get("serial_number") is not None
            and re.fullmatch(
                r"\d{4,6}",
                str(row.get("lot_number", "")).strip(),
            )
        }

        for item in items:
            text = str(item.get("text", "")).strip()
            serial_number = serial_by_lot.get(text)

            if serial_number is None:
                continue

            center_y = cls._item_center_y(item)

            if center_y > header_bottom:
                centers[serial_number] = center_y

        return centers

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