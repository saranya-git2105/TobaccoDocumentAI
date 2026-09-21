from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from statistics import median
from typing import Any

from app.services.image_service import ImageService


@dataclass(frozen=True)
class _Token:
    text: str
    confidence: float
    left: float
    top: float
    right: float
    bottom: float

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 1.0)

    @property
    def width(self) -> float:
        return max(self.right - self.left, 1.0)


class DeliveryNoteExtractor:
    """Maps positioned PaddleOCR text to the DeliveryNote JSON contract."""

    _DATE_PREFIX_PATTERN = re.compile(
        r"^(?:MP2?|AP2?|AF2?|APV|R2?|L\d?[A-Z]{0,3}|LIST)\s*",
        re.IGNORECASE,
    )
    _HANDWRITTEN_NAME_MARKERS = re.compile(
        r"\b(?:AP2?|MP2?|AF2?|R2?|LIST|L1ST|L2B|L3A|L1A|L2A|L1B)\/?\d*\b",
        re.IGNORECASE,
    )
    # The handwritten row markers are sometimes joined to the final printed
    # name by OCR. These endings are evidence that the name needs a focused
    # second pass; they are not removed blindly because a short ending such as
    # "AP" can also be part of a genuine name (for example, PRATAP).
    _ATTACHED_HANDWRITING_SUFFIX = re.compile(
        r"(?:APV?|MPV?|AFV?|PPLB|PLB|PL|R[1-3]|L[1-4][A-Z]{0,2})$",
        re.IGNORECASE,
    )
    _DELIVERY_NOTE_PATTERN = re.compile(
        r"\d{1,3}/\d{12,}[A-Z]?",
    )
    # A TBGR number is eight digits. The leading digits vary by auction
    # platform, so no specific prefix is assumed; when a number needs repairing
    # the prefix is learned from the other rows on the same note.
    _TBGR_PATTERN = re.compile(r"\d{8}")

    # Digits a letter may stand for when OCR misreads a TBGR number.
    _TBGR_LETTER_DIGITS = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "U": "0",
        "I": "1",
        "L": "1",
        "J": "1",
        "Z": "2",
        "S": "35",
        "A": "4",
        "B": "83",
        "G": "6",
        "T": "7",
        "E": "8",
    }

    # Covers both printed grade families: F02 / F02M / F03 and L2L / L2O / X2L.
    _GRADE_PATTERN = re.compile(r"[FLX]\d{1,2}[A-Z]?", re.IGNORECASE)
    # Letters OCR commonly returns in place of digits inside a grade cell.
    _DIGIT_CONFUSIONS = str.maketrans(
        {
            "O": "0",
            "C": "0",
            "Q": "0",
            "D": "0",
            "U": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "S": "5",
            "G": "6",
            "B": "8",
        }
    )
    _GRADE_DIRECT_MAP = {
        "2L": "L2L",
        "20": "L2O",
        "2O": "L2O",
        "L20": "L2O",
        "L2O": "L2O",
        "L2L": "L2L",
        "X2L": "X2L",
        "120": "L2O",
    }

    # A single bale ranges well below the old 100 kg floor, so the floor is
    # low enough to keep genuine light bales and bale/rate arithmetic is used
    # as the real consistency check.
    _MIN_WEIGHT = 50.0
    _MAX_WEIGHT = 300.0
    _MIN_RATE = 100.0
    _MAX_RATE = 500.0

    # Fixed column maps for the two Tobacco Board print layouts.
    # Extraction uses these instead of re-measuring headers on every upload,
    # so JPG / PNG / PDF produce the same column assignment logic.
    _LAYOUT_TEMPLATES: dict[str, dict[str, tuple[float, float]]] = {
        "wide": {
            "left": (0.0, 0.37),
            "grower": (0.0, 0.30),
            "lot": (0.37, 0.43),
            "weight": (0.43, 0.47),
            "second_weight": (0.47, 0.51),
            "grade": (0.51, 0.56),
            "rate": (0.56, 0.62),
            "bale": (0.62, 1.01),
        },
        "compact": {
            "left": (0.0, 0.53),
            "grower": (0.0, 0.48),
            "lot": (0.53, 0.62),
            "weight": (0.62, 0.687),
            "second_weight": (0.687, 0.75),
            "grade": (0.75, 0.805),
            "rate": (0.805, 0.875),
            "bale": (0.875, 1.01),
        },
    }

    _FIELD_LABELS = {
        "delivery_note_number": (
            r"delivery\s*(?:note|challan)\s*(?:no|number|#)",
            r"(?:note|challan)\s*(?:no|number|#)",
        ),
        "delivery_date": (
            r"delivery\s*date",
            r"date\s*of\s*delivery",
        ),
        "buyer_name": (
            r"buyer\s*(?:name)?",
            r"name\s*of\s*buyer",
        ),
        "auction_platform_number": (
            r"auction\s*(?:platform|floor)\s*(?:no|number|#)",
            r"platform\s*(?:no|number|#)",
        ),
        "auction_platform_name": (
            r"auction\s*(?:platform|floor)\s*name",
            r"platform\s*name",
        ),
        "code_number": (
            r"code\s*(?:no|number|#)",
            r"buyer\s*code",
        ),
        "printed_date": (
            r"print(?:ed)?\s*date",
            r"print(?:ed)?\s*on",
            r"date\s*printed",
        ),
    }

    _COLUMN_LABELS = {
        "serial_number": (
            r"s\.?\s*no",
            r"sl\.?\s*no",
            r"serial",
        ),
        "tbgr_number": (
            r"tbgr\s*\.?\s*(?:no|number)?\.?",
            r"tbgrno\.?",
            r"grower\s*(?:no|number|id)",
        ),
        "grower_name": (
            r"grower\s*name",
            r"name\s*of\s*(?:the\s*)?grower",
        ),
        "date_of_purchase": (
            r"(?:date\s*of\s*)?purchase",
            r"purchase\s*date",
        ),
        "lot_number": (
            r"lot\s*(?:no|number|#)?\.?",
            r"lol\s*(?:no|number)?\.?",
        ),
        "weight": (
            r"(?:first\s*)?weight",
            r"qty|quantity",
        ),
        "second_weight": (
            r"second\s*weight",
            r"weight\s*2",
        ),
        "grade": (
            r"grade",
        ),
        "rate_per_kg": (
            r"rate(?:\s*per\s*kg\.?)?",
            r"price(?:\s*per\s*kg)?",
        ),
        "bale_value": (
            r"bale\s*value(?:\s*\(?(?:rs|inr)\.?\)?)?",
            r"(?:total\s*)?value|amount",
        ),
    }

    _FOOTER_WORDS = (
        "total",
        "signature",
        "authorised",
        "authorized",
        "remarks",
    )

    @classmethod
    def extract(
        cls,
        items: list[dict[str, Any]],
        *,
        source: str = "image",
    ) -> dict[str, Any]:
        items = cls._expand_items(items)
        tokens = cls._to_tokens(items)
        fields = {
            field_name: cls._extract_field(tokens, labels)
            for field_name, labels in cls._FIELD_LABELS.items()
        }

        normalized_delivery_note = cls._normalize_delivery_note_number(
            fields["delivery_note_number"]
        )

        if not normalized_delivery_note:
            normalized_delivery_note = cls._normalize_delivery_note_number(
                cls._extract_delivery_note_fallback(tokens)
            )

        fields["delivery_note_number"] = normalized_delivery_note

        fields["auction_platform_number"] = cls._digits_only(
            fields["auction_platform_number"]
        )
        fields["code_number"] = cls._clean_value(fields["code_number"])
        fields["printed_date"] = cls._normalize_printed_date(
            cls._extract_printed_on(tokens)
            or fields["printed_date"]
        )
        fields["delivery_date"] = (
            cls._extract_delivery_date(tokens)
            or fields["delivery_date"]
            or cls._extract_delivery_date_from_note(
                fields["delivery_note_number"]
            )
            or cls._extract_date_from_printed(fields["printed_date"])
        )
        fields["buyer_name"] = cls._normalize_buyer_name(
            fields["buyer_name"]
            or cls._extract_buyer_name(tokens)
        )
        fields["auction_platform_name"] = (
            fields["auction_platform_name"]
            or cls._extract_platform_name(tokens)
        )

        headers = cls._find_table_headers(tokens)
        page_width = max(token.right for token in tokens) if tokens else 1.0
        layout_name = cls._resolve_layout_name(
            headers,
            page_width,
            normalized_delivery_note,
        )
        table_items = cls._postprocess_rows(
            cls._extract_table(
                tokens,
                delivery_note_number=normalized_delivery_note,
                layout_name=layout_name,
            )
        )
        totals = cls._extract_totals(tokens, table_items)
        extraction_meta = cls._build_extraction_meta(
            table_items,
            layout_name=layout_name,
            totals=totals,
            source=source,
        )

        return {
            "document_type": "Tobacco Board Delivery Note",
            **fields,
            "items": table_items,
            "totals": totals,
            "extraction_meta": extraction_meta,
        }

    @staticmethod
    def _to_tokens(items: list[dict[str, Any]]) -> list[_Token]:
        tokens: list[_Token] = []

        for item in items:
            text = str(item.get("text", "")).strip()
            polygon = item.get("boundingBox", [])

            if not text or not isinstance(polygon, list) or not polygon:
                continue

            points = [
                point
                for point in polygon
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]

            if not points:
                continue

            x_values = [float(point[0]) for point in points]
            y_values = [float(point[1]) for point in points]
            tokens.append(
                _Token(
                    text=text,
                    confidence=float(item.get("confidence", 0.0)),
                    left=min(x_values),
                    top=min(y_values),
                    right=max(x_values),
                    bottom=max(y_values),
                )
            )

        return sorted(tokens, key=lambda token: (token.center_y, token.left))

    @classmethod
    def _expand_items(
        cls,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []

        for item in items:
            text = str(item.get("text", "")).strip()

            if not text:
                continue

            parts = cls._split_compound_text(text)

            if len(parts) <= 1 or not cls._should_split_compound_text(text):
                expanded.append(item)
                continue

            for part in parts:
                expanded.append({**item, "text": part})

        return expanded

    @staticmethod
    def _split_compound_text(text: str) -> list[str]:
        remaining = text.strip()
        parts: list[str] = []

        if not remaining:
            return parts

        pattern_order = (
            r"\d{8}",
            r"\d{1,2}/\d{1,2}/\d{2,4}",
            r"[LX]\d[LO0]",
            r"\d+\.\d{2}",
            r"\b\d{4}\b",
        )

        while remaining:
            match: re.Match[str] | None = None

            for pattern in pattern_order:
                current_match = re.search(pattern, remaining, re.IGNORECASE)

                if current_match and (
                    match is None
                    or current_match.start() < match.start()
                ):
                    match = current_match

            if match is None:
                chunk = re.sub(r"[^A-Za-z0-9' ]", " ", remaining)
                chunk = re.sub(r"\s+", " ", chunk).strip()

                if chunk and chunk.lower() not in {"total", "no"}:
                    parts.append(chunk)

                break

            prefix = remaining[: match.start()].strip()
            prefix = re.sub(r"[^A-Za-z0-9' ]", " ", prefix)
            prefix = re.sub(r"\s+", " ", prefix).strip()

            if prefix and prefix.lower() not in {"total", "no"}:
                parts.append(prefix)

            parts.append(match.group(0))
            remaining = remaining[match.end() :].strip()

        return [part for part in parts if part]

    @staticmethod
    def _should_split_compound_text(text: str) -> bool:
        signals = sum(
            bool(re.search(pattern, text, re.IGNORECASE))
            for pattern in (
                r"\d{8}",
                r"\d{1,2}/\d{1,2}/\d{2,4}",
                r"\d+\.\d{2}",
                r"[LX]\d[LO0]",
            )
        )
        return signals >= 2 and len(text) >= 12

    @classmethod
    def _extract_delivery_note_fallback(
        cls,
        tokens: list[_Token],
    ) -> str:
        for token in tokens:
            compact = token.text.replace(" ", "")
            match = cls._DELIVERY_NOTE_PATTERN.search(compact)

            if match:
                validated = cls._normalize_delivery_note_number(
                    match.group(0)
                )

                if validated:
                    return validated

        header_tokens = [
            token
            for token in tokens
            if token.top < 320
        ]
        grouped: dict[int, list[_Token]] = {}

        for token in header_tokens:
            bucket = int(round(token.center_y / 8) * 8)
            grouped.setdefault(bucket, []).append(token)

        for row_tokens in grouped.values():
            if not any(
                "note" in token.text.lower()
                or "deliver" in token.text.lower()
                or "deliveny" in token.text.lower()
                for token in row_tokens
            ):
                continue

            digit_parts: list[str] = []

            for token in sorted(row_tokens, key=lambda value: value.left):
                digit_parts.extend(
                    re.findall(r"\d+[A-Z]?", token.text, re.IGNORECASE)
                )

            if (
                len(digit_parts) >= 2
                and digit_parts[0] in {"25", "30"}
                and len("".join(digit_parts[1:])) >= 8
            ):
                validated = cls._normalize_delivery_note_number(
                    f"{digit_parts[0]}/{''.join(digit_parts[1:])}"
                )

                if validated:
                    return validated

        for row_tokens in grouped.values():
            combined = "".join(
                token.text.replace(" ", "")
                for token in sorted(row_tokens, key=lambda value: value.left)
            )
            match = cls._DELIVERY_NOTE_PATTERN.search(combined)

            if match:
                validated = cls._normalize_delivery_note_number(
                    match.group(0)
                )

                if validated:
                    return validated

        return ""

    @classmethod
    def _normalize_delivery_note_number(cls, value: str) -> str:
        cleaned = value.replace(" ", "")
        match = re.fullmatch(
            r"(\d{1,3})/(\d{12,})([A-Z]?)",
            cleaned,
        )

        if not match:
            return ""

        body = match.group(2)
        day = int(body[0:2])
        month = int(body[2:4])

        if not (1 <= day <= 31 and 1 <= month <= 12):
            return ""

        return f"{match.group(1)}/{body}{match.group(3)}"

    @classmethod
    def _extract_field(
        cls,
        tokens: list[_Token],
        labels: tuple[str, ...],
    ) -> str:
        label_expression = "|".join(f"(?:{label})" for label in labels)
        inline_pattern = re.compile(
            rf"^\s*(?:{label_expression})\s*[:#=\-]?\s*(.+?)\s*$",
            re.IGNORECASE,
        )
        label_pattern = re.compile(
            rf"^\s*(?:{label_expression})\s*[:#=\-]?\s*$",
            re.IGNORECASE,
        )

        inline_matches: list[tuple[float, str]] = []
        label_tokens: list[_Token] = []

        for token in tokens:
            if label_pattern.match(token.text):
                label_tokens.append(token)
                continue

            inline_match = inline_pattern.match(token.text)
            if inline_match:
                value = cls._clean_value(inline_match.group(1))
                if value:
                    inline_matches.append((token.confidence, value))

        if inline_matches:
            return max(inline_matches, key=lambda match: match[0])[1]

        candidates: list[tuple[float, float, str]] = []

        for label_token in label_tokens:
            for token in tokens:
                if token is label_token or cls._is_known_label(token.text):
                    continue

                same_line_distance = abs(token.center_y - label_token.center_y)
                same_line_limit = max(label_token.height, token.height) * 0.8

                if (
                    token.left >= label_token.right - 2
                    and same_line_distance <= same_line_limit
                ):
                    distance = max(token.left - label_token.right, 0)
                    candidates.append(
                        (distance, -token.confidence, token.text)
                    )

            if not candidates:
                below = [
                    token
                    for token in tokens
                    if token.top >= label_token.bottom
                    and token.center_y
                    <= label_token.bottom + (label_token.height * 2.5)
                    and not cls._is_known_label(token.text)
                ]

                if below:
                    nearest = min(
                        below,
                        key=lambda token: (
                            token.top - label_token.bottom,
                            abs(token.left - label_token.left),
                        ),
                    )
                    candidates.append(
                        (
                            nearest.top - label_token.bottom,
                            -nearest.confidence,
                            nearest.text,
                        )
                    )

        if not candidates:
            return ""

        return cls._clean_value(min(candidates)[2])

    @classmethod
    def _extract_buyer_name(cls, tokens: list[_Token]) -> str:
        to_token = next(
            (
                token
                for token in tokens
                if cls._normalize(token.text).rstrip(":") == "to"
            ),
            None,
        )

        if to_token is None:
            return ""

        candidates = [
            token
            for token in tokens
            if token.top >= to_token.bottom
            and token.top <= to_token.bottom + (to_token.height * 2)
            and abs(token.left - to_token.left) <= to_token.height * 2
        ]

        if not candidates:
            return ""

        first = min(candidates, key=lambda token: token.top)
        page_width = max(token.right for token in tokens)

        # OCR may split the buyer line into several tokens. Join what is printed
        # on that line, stopping before the labels sharing the same line.
        line = sorted(
            (
                token
                for token in tokens
                if abs(token.center_y - first.center_y)
                <= max(first.height, token.height) * 0.6
                and token.right > first.left
                and token.center_x <= page_width * 0.55
                and not cls._is_known_label(token.text)
                and not re.search(r"code\s*no", token.text, re.IGNORECASE)
            ),
            key=lambda token: token.left,
        )

        if not line:
            return cls._clean_value(first.text)

        return cls._clean_value(" ".join(token.text for token in line))

    @classmethod
    def _extract_platform_name(cls, tokens: list[_Token]) -> str:
        label_pattern = re.compile(
            r"auction\s*(?:platform|floor)\s*(?:no|number|#)",
            re.IGNORECASE,
        )
        label_token = next(
            (
                token
                for token in tokens
                if label_pattern.search(token.text)
            ),
            None,
        )

        if label_token is None:
            return ""

        candidates = [
            token
            for token in tokens
            if token is not label_token
            and token.center_y >= label_token.bottom
            and token.top <= label_token.bottom + (label_token.height * 1.5)
            and abs(token.center_x - label_token.center_x)
            <= max(label_token.width, label_token.height * 5)
            and re.search(r"[A-Za-z]", token.text)
        ]

        if not candidates:
            return ""

        return cls._clean_value(
            min(candidates, key=lambda token: token.top).text
        )

    @classmethod
    def _extract_table(
        cls,
        tokens: list[_Token],
        *,
        delivery_note_number: str = "",
        layout_name: str | None = None,
    ) -> list[dict[str, Any]]:
        headers = cls._find_table_headers(tokens)

        if len(headers) < 2:
            return []

        page_width = max(token.right for token in tokens)
        resolved_layout = layout_name or cls._resolve_layout_name(
            headers,
            page_width,
            delivery_note_number,
        )
        header_bottom = cls._table_body_start(headers)

        # The footer total line must not leak into the last data row.
        footer_top = min(
            (
                token.top
                for token in tokens
                if token.top > header_bottom
                and cls._normalize(token.text) == "total"
            ),
            default=None,
        )
        body_tokens = [
            token
            for token in tokens
            if token.top > header_bottom
            and not cls._is_known_label(token.text)
            and cls._normalize(token.text) != "total"
            and (footer_top is None or token.center_y < footer_top)
        ]

        if not body_tokens:
            return []

        page_width = max(token.right for token in tokens)
        tbgr_header = next(
            (
                token
                for name, token in headers
                if name == "tbgr_number"
            ),
            None,
        )
        serial_right = (
            tbgr_header.left
            if tbgr_header is not None
            else page_width * 0.12
        )
        serial_tokens = [
            token
            for token in body_tokens
            if token.center_x < serial_right
            and re.fullmatch(r"\d{1,3}", token.text.strip())
            and 1 <= int(token.text.strip()) <= 60
        ]

        column_ranges = cls._build_column_ranges(
            headers,
            page_width,
            layout_name=resolved_layout,
        )
        serial_tokens = cls._dedupe_serial_tokens(serial_tokens)
        serial_tokens = cls._supplement_serials_from_lot_rows(
            serial_tokens,
            body_tokens,
            page_width,
            column_ranges,
        )

        if not serial_tokens:
            return []

        serial_tokens.sort(key=lambda token: token.center_y)
        serial_tokens = cls._complete_serial_tokens(serial_tokens)
        row_assignments = cls._assign_tokens_to_serial_rows(
            serial_tokens,
            body_tokens,
            page_width,
            serial_right,
        )
        extracted_rows: list[dict[str, Any]] = []

        for serial_token in sorted(
            serial_tokens,
            key=lambda token: int(token.text),
        ):
            serial_number = int(serial_token.text)
            row = row_assignments.get(serial_number, [])

            extracted_rows.append(
                cls._extract_positioned_row(
                    serial_token,
                    row,
                    page_width,
                    column_ranges,
                )
            )

        return extracted_rows

    @staticmethod
    def _dedupe_serial_tokens(
        serial_tokens: list[_Token],
    ) -> list[_Token]:
        best_by_serial: dict[int, _Token] = {}

        for token in serial_tokens:
            serial_number = int(token.text.strip())
            current = best_by_serial.get(serial_number)

            if current is None or token.confidence > current.confidence:
                best_by_serial[serial_number] = token

        return list(best_by_serial.values())

    @staticmethod
    def _supplement_serials_from_lot_rows(
        serial_tokens: list[_Token],
        body_tokens: list[_Token],
        page_width: float,
        column_ranges: dict[str, tuple[float, float]],
    ) -> list[_Token]:
        """Recover row anchors from lot numbers when serial glyphs are missed."""
        if not serial_tokens:
            return serial_tokens

        lot_start, lot_end = column_ranges["lot"]
        lot_tokens = sorted(
            (
                token
                for token in body_tokens
                if lot_start <= token.center_x / page_width < lot_end
                and re.fullmatch(r"\d{4,6}", token.text.strip())
            ),
            key=lambda token: token.center_y,
        )

        if len(lot_tokens) < 2:
            return serial_tokens

        # Keep one lot anchor per visual row.
        deduped_lots: list[_Token] = []

        for token in lot_tokens:
            if (
                deduped_lots
                and abs(token.center_y - deduped_lots[-1].center_y)
                <= max(token.height, deduped_lots[-1].height) * 0.45
            ):
                if token.confidence > deduped_lots[-1].confidence:
                    deduped_lots[-1] = token
                continue

            deduped_lots.append(token)

        base_candidates: list[int] = []

        for serial in serial_tokens:
            nearest_index, nearest_lot = min(
                enumerate(deduped_lots),
                key=lambda value: abs(
                    value[1].center_y - serial.center_y
                ),
            )

            if (
                abs(nearest_lot.center_y - serial.center_y)
                <= max(nearest_lot.height, serial.height) * 1.5
            ):
                base_candidates.append(
                    int(serial.text) - nearest_index
                )

        if not base_candidates:
            return serial_tokens

        first_serial, votes = Counter(base_candidates).most_common(1)[0]

        # Conflicting anchors indicate that the lot column was misidentified;
        # retain the original serials instead of inventing rows.
        if votes * 2 <= len(base_candidates):
            return serial_tokens

        existing_numbers = {
            int(token.text)
            for token in serial_tokens
        }
        template = max(
            serial_tokens,
            key=lambda token: token.confidence,
        )
        supplemented = list(serial_tokens)

        for index, lot_token in enumerate(deduped_lots):
            serial_number = first_serial + index

            if (
                serial_number in existing_numbers
                or not 1 <= serial_number <= 60
            ):
                continue

            supplemented.append(
                _Token(
                    text=str(serial_number),
                    confidence=0.0,
                    left=template.left,
                    top=lot_token.center_y - (template.height / 2),
                    right=template.right,
                    bottom=lot_token.center_y + (template.height / 2),
                )
            )

        return supplemented

    @classmethod
    def _assign_tokens_to_serial_rows(
        cls,
        serial_tokens: list[_Token],
        body_tokens: list[_Token],
        page_width: float,
        serial_right: float,
    ) -> dict[int, list[_Token]]:
        ordered_serials = sorted(
            serial_tokens,
            key=lambda token: int(token.text),
        )
        serial_centers = [
            (int(token.text), token.center_y)
            for token in ordered_serials
        ]
        rows: dict[int, list[_Token]] = {
            serial_number: []
            for serial_number, _ in serial_centers
        }

        if not serial_centers:
            return rows

        row_steps = [
            abs(right_y - left_y) / max(1, right_number - left_number)
            for (left_number, left_y), (right_number, right_y)
            in zip(serial_centers, serial_centers[1:])
            if right_number > left_number and right_y > left_y
        ]
        typical_step = median(row_steps) if row_steps else max(
            token.height for token in ordered_serials
        )
        row_bands = cls._build_row_bands(serial_centers, typical_step)

        for token in body_tokens:
            if (
                token.center_x < serial_right
                and re.fullmatch(r"\d{1,3}", token.text.strip())
            ):
                continue

            for serial_number, top, bottom in row_bands:
                if top <= token.center_y < bottom:
                    rows[serial_number].append(token)
                    break

        return rows

    @staticmethod
    def _build_row_bands(
        serial_centers: list[tuple[int, float]],
        typical_step: float,
    ) -> list[tuple[int, float, float]]:
        """Split the table into one band per serial anchor.

        Boundaries sit midway between neighbouring anchors, so every band is
        centered on its own printed row. A row whose grower name wraps onto a
        second printed line sits inside a proportionally taller band and stays
        with its own serial. This relies on the page being deskewed first;
        without that, a tilt makes the left-hand serial column drift against the
        right-hand value columns and no fixed band offset can absorb it.
        """
        bands: list[tuple[int, float, float]] = []

        for index, (serial_number, center_y) in enumerate(serial_centers):
            if index == 0:
                top = center_y - (typical_step * 0.5)
            else:
                top = (serial_centers[index - 1][1] + center_y) / 2

            if index + 1 < len(serial_centers):
                bottom = (center_y + serial_centers[index + 1][1]) / 2
            else:
                bottom = center_y + (typical_step * 0.6)

            bands.append((serial_number, top, bottom))

        return bands

    @classmethod
    def _extract_delivery_date(cls, tokens: list[_Token]) -> str:
        for token in tokens:
            inline_match = re.search(
                r"delivery\s*date\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
                token.text,
                re.IGNORECASE,
            )

            if inline_match:
                return inline_match.group(1)

        label = next(
            (
                token
                for token in tokens
                if re.search(r"delivery\s*date", token.text, re.IGNORECASE)
            ),
            None,
        )

        if label is None:
            return ""

        candidates = [
            token
            for token in tokens
            if token is not label
            and re.search(r"\d{1,2}/\d{1,2}/\d{4}", token.text)
            and abs(token.center_y - label.center_y)
            <= max(label.height, token.height) * 1.5
            and token.left >= label.left - (label.height * 2)
        ]

        if not candidates:
            inline_value = re.search(
                r"(\d{1,2}/\d{1,2}/\d{4})",
                label.text,
            )

            if inline_value:
                return inline_value.group(1)

            return ""

        return cls._clean_value(
            min(
                candidates,
                key=lambda token: (
                    abs(token.center_y - label.center_y),
                    token.left - label.right,
                ),
            ).text
        )

    @staticmethod
    def _extract_delivery_date_from_note(value: str) -> str:
        match = re.search(r"/(\d{8})", value)

        if not match:
            return ""

        digits = match.group(1)
        day = int(digits[:2])
        month = int(digits[2:4])
        year = digits[4:]

        if not (1 <= day <= 31 and 1 <= month <= 12):
            return ""

        return f"{day:02d}/{month:02d}/{year}"

    @staticmethod
    def _extract_date_from_printed(value: str) -> str:
        match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", value)
        return match.group(1) if match else ""

    @classmethod
    def _extract_printed_on(cls, tokens: list[_Token]) -> str:
        for token in tokens:
            inline_match = re.search(
                r"print(?:ed)?\s*on\s*[: ]*"
                r"(\d{1,2}/\d{1,2}/\d{4})\s*(\d{1,2}:\d{2})?",
                token.text,
                re.IGNORECASE,
            )

            if inline_match:
                return " ".join(
                    part
                    for part in inline_match.groups()
                    if part
                )

        label = next(
            (
                token
                for token in tokens
                if re.search(r"print(?:ed)?\s*on", token.text, re.IGNORECASE)
            ),
            None,
        )

        if label is None:
            return ""

        candidates = [
            token
            for token in tokens
            if token is not label
            and re.search(
                r"\d{1,2}/\d{1,2}/\d{4}",
                token.text,
            )
            and abs(token.center_y - label.center_y)
            <= max(label.height, token.height) * 1.5
        ]

        if not candidates:
            return ""

        date_token = min(
            candidates,
            key=lambda token: abs(token.center_y - label.center_y),
        )
        time_match = re.search(
            r"(\d{1,2}/\d{1,2}/\d{4})\s*(\d{1,2}:\d{2})?",
            date_token.text,
        )

        if not time_match:
            return ""

        return " ".join(
            part
            for part in time_match.groups()
            if part
        )

    @staticmethod
    def _complete_serial_tokens(
        serial_tokens: list[_Token],
    ) -> list[_Token]:
        """Add row anchors when OCR misses a short sequence of serials."""
        if len(serial_tokens) < 2:
            return serial_tokens

        observed_row_steps = [
            (right.center_y - left.center_y)
            / (int(right.text) - int(left.text))
            for left, right in zip(serial_tokens, serial_tokens[1:])
            if 0 < int(right.text) - int(left.text) <= 6
            and right.center_y > left.center_y
        ]
        typical_row_step = (
            median(observed_row_steps)
            if observed_row_steps
            else None
        )
        completed: list[_Token] = []

        for left, right in zip(serial_tokens, serial_tokens[1:]):
            completed.append(left)
            left_number = int(left.text)
            right_number = int(right.text)
            gap = right_number - left_number

            if not 1 < gap <= 6:
                continue

            y_step = (right.center_y - left.center_y) / gap

            if (
                y_step <= 0
                or (
                    typical_row_step is not None
                    and not (
                        typical_row_step * 0.65
                        <= y_step
                        <= typical_row_step * 1.45
                    )
                )
            ):
                continue

            for offset in range(1, gap):
                center_y = left.center_y + (y_step * offset)
                completed.append(
                    _Token(
                        text=str(left_number + offset),
                        confidence=0.0,
                        left=left.left,
                        top=center_y - (left.height / 2),
                        right=left.right,
                        bottom=center_y + (left.height / 2),
                    )
                )

        completed.append(serial_tokens[-1])
        return sorted(completed, key=lambda token: token.center_y)

    @classmethod
    def _extract_positioned_row(
        cls,
        serial_token: _Token,
        row: list[_Token],
        page_width: float,
        column_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        item = cls._empty_item()
        item["serial_number"] = int(serial_token.text)
        ranges = column_ranges or cls._default_column_ranges()

        text_columns: dict[str, list[str]] = {
            "lot_number": [],
            "grade": [],
            "grower_name": [],
        }
        numeric_columns: dict[str, list[tuple[str, float]]] = {
            "weight": [],
            "second_weight": [],
            "rate_per_kg": [],
            "bale_value": [],
        }

        for token in sorted(row, key=lambda value: value.left):
            relative_x = token.center_x / page_width
            column = cls._column_for_position(relative_x, ranges)

            if column == "left":
                name_part = cls._extract_name_part(
                    token.text,
                    include_name=(
                        relative_x < ranges["grower"][1]
                        and not (
                            relative_x < 0.09
                            and len(
                                re.sub(r"[^A-Za-z]", "", token.text)
                            )
                            <= 2
                        )
                    ),
                )
                cls._merge_left_columns(
                    item,
                    token.text,
                    include_name=False,
                )

                if name_part:
                    text_columns["grower_name"].append(name_part)
            elif column == "lot":
                text_columns["lot_number"].append(token.text)
            elif column == "weight":
                numeric_columns["weight"].append(
                    (token.text, token.center_y)
                )
            elif column == "second_weight":
                numeric_columns["second_weight"].append(
                    (token.text, token.center_y)
                )
            elif column == "grade":
                text_columns["grade"].append(token.text)
            elif column == "rate":
                numeric_columns["rate_per_kg"].append(
                    (token.text, token.center_y)
                )
            elif column == "bale":
                numeric_columns["bale_value"].append(
                    (token.text, token.center_y)
                )

        item["lot_number"] = cls._clean_lot_number(
            " ".join(text_columns["lot_number"])
        )
        item["grade"] = cls._select_row_grade(text_columns["grade"])

        grower_name = " ".join(text_columns["grower_name"]).strip()
        if grower_name:
            item["grower_name_ocr"] = re.sub(r"\s+", " ", grower_name).strip()
            item["grower_name"] = cls._clean_grower_name(grower_name)

        cls._apply_numeric_columns(
            item,
            numeric_columns,
            serial_token.center_y,
        )

        return item

    @classmethod
    def _apply_numeric_columns(
        cls,
        item: dict[str, Any],
        numeric_columns: dict[str, list[tuple[str, float]]],
        serial_center_y: float,
        *,
        weight_hint: float | None = None,
    ) -> None:
        weight_entries = numeric_columns["weight"]
        second_weight_entries = numeric_columns["second_weight"]
        rate_entries = numeric_columns["rate_per_kg"]
        bale_entries = numeric_columns["bale_value"]

        item["weight"] = cls._select_numeric_entry(
            weight_entries,
            anchor_y=serial_center_y + 8,
            repair_weight=True,
        )
        item["second_weight"] = cls._select_numeric_entry(
            second_weight_entries,
            anchor_y=serial_center_y + 8,
            repair_weight=True,
        )

        selected_rate, selected_bale = cls._select_rate_bale_pair(
            rate_entries,
            bale_entries,
            serial_center_y,
            weight_hint=item.get("weight"),
        )
        item["rate_per_kg"] = selected_rate
        item["bale_value"] = selected_bale

    @classmethod
    def _select_numeric_entry(
        cls,
        entries: list[tuple[str, float]],
        *,
        anchor_y: float,
        repair_weight: bool = False,
    ) -> float | None:
        if not entries:
            return None

        if len(entries) == 1:
            return cls._parse_decimal(
                entries[0][0],
                repair_weight=repair_weight,
            )

        best_text = min(
            entries,
            key=lambda entry: abs(entry[1] - anchor_y),
        )[0]
        return cls._parse_decimal(
            best_text,
            repair_weight=repair_weight,
        )

    @classmethod
    def _select_rate_bale_pair(
        cls,
        rate_entries: list[tuple[str, float]],
        bale_entries: list[tuple[str, float]],
        serial_center_y: float,
        *,
        weight_hint: float | None = None,
    ) -> tuple[float | None, float | None]:
        if not rate_entries and not bale_entries:
            return None, None

        if rate_entries and bale_entries:
            best_rate_text = None
            best_bale_text = None
            best_score = float("inf")

            for rate_text, rate_y in rate_entries:
                for bale_text, bale_y in bale_entries:
                    rate_value = cls._parse_decimal(rate_text)
                    bale_value = cls._parse_decimal(bale_text)
                    score = abs(rate_y - bale_y) + abs(rate_y - serial_center_y) * 0.15

                    if rate_value is None or bale_value is None:
                        score += 1000
                    elif not cls._MIN_RATE <= rate_value <= cls._MAX_RATE:
                        score += 500
                    elif (
                        weight_hint is not None
                        and weight_hint > 0
                    ):
                        score += abs(
                            (weight_hint * rate_value) - bale_value
                        ) * 0.35

                    if score < best_score:
                        best_score = score
                        best_rate_text = rate_text
                        best_bale_text = bale_text

            return (
                cls._parse_decimal(best_rate_text or ""),
                cls._parse_decimal(best_bale_text or ""),
            )

        if rate_entries:
            return (
                cls._select_numeric_entry(
                    rate_entries,
                    anchor_y=serial_center_y,
                ),
                None,
            )

        return (
            None,
            cls._select_numeric_entry(
                bale_entries,
                anchor_y=serial_center_y,
            ),
        )

    @classmethod
    def _split_leading_tbgr(cls, text: str) -> tuple[str, str]:
        """Split a left-column string into its TBGR number and the rest."""
        remaining = text.strip()
        tbgr_match = re.match(r"^[^0-9]*(\d{8})(.*)$", remaining)

        if tbgr_match:
            return tbgr_match.group(1), tbgr_match.group(2).strip()

        # OCR sometimes reads a digit as a letter, which hides the number from
        # the plain eight-digit match above. Claim the slot here so the name is
        # split off cleanly; the letters become digits in _recover_tbgr_number,
        # once the note's own prefixes are known.
        # The name is often printed hard against the number, so no separator is
        # required after the window. Demanding almost all of it be digits is
        # what keeps a plain name from being mistaken for a number.
        confused_match = re.match(r"^\s*([0-9A-Za-z]{8})(.*)$", remaining)

        if confused_match:
            window = confused_match.group(1).upper()
            letters = [
                character
                for character in window
                if not character.isdigit()
            ]

            if (
                letters
                and len(letters) <= 2
                and all(
                    character in cls._TBGR_LETTER_DIGITS
                    for character in letters
                )
            ):
                return window, confused_match.group(2).strip()

        return "", remaining

    @classmethod
    def _merge_left_columns(
        cls,
        item: dict[str, Any],
        text: str,
        *,
        include_name: bool,
    ) -> None:
        tbgr_number, remaining = cls._split_leading_tbgr(text)

        if tbgr_number and not item["tbgr_number"]:
            item["tbgr_number"] = tbgr_number

        date_value, date_span = cls._extract_purchase_date(remaining)

        if date_value and not item["date_of_purchase"]:
            item["date_of_purchase"] = date_value

        if date_span is not None:
            remaining = (
                remaining[: date_span[0]]
                + " "
                + remaining[date_span[1] :]
            ).strip()

        if not include_name:
            return

        name = re.sub(r"[^A-Za-z ]", " ", remaining)
        name = re.sub(r"\s+", " ", name).strip()

        if name:
            current_name = item["grower_name"]
            item["grower_name"] = (
                f"{current_name} {name}".strip()
                if current_name
                else name
            )

    @classmethod
    def _extract_name_part(
        cls,
        text: str,
        *,
        include_name: bool,
    ) -> str:
        if not include_name:
            return ""

        _, remaining = cls._split_leading_tbgr(text)
        _, date_span = cls._extract_purchase_date(remaining)

        if date_span is not None:
            remaining = (
                remaining[: date_span[0]]
                + " "
                + remaining[date_span[1] :]
            ).strip()

        name = re.sub(r"[^A-Za-z ]", " ", remaining)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    @classmethod
    def _normalize_buyer_name(cls, value: str) -> str:
        normalized = cls._clean_value(value)
        normalized = re.sub(
            r"\bMs\.\s*",
            "M/s. ",
            normalized,
            flags=re.IGNORECASE,
        )
        return normalized.strip()

    @classmethod
    def _sanitize_purchase_date_text(cls, value: str) -> str:
        cleaned = cls._DATE_PREFIX_PATTERN.sub("", value.strip())
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @classmethod
    def _normalize_purchase_day(cls, day_text: str) -> int | None:
        digits = re.sub(r"\D", "", day_text)

        if not digits:
            return None

        if len(digits) == 1:
            day = int(digits)
        elif len(digits) == 2:
            day = int(digits)
        else:
            candidates = [int(digits[-2:]), int(digits[:2])]
            day = next(
                (candidate for candidate in candidates if 1 <= candidate <= 31),
                int(digits[-2:]),
            )

        if 1 <= day <= 31:
            return day

        return None

    @classmethod
    def _extract_purchase_date(
        cls,
        value: str,
    ) -> tuple[str, tuple[int, int] | None]:
        value = cls._sanitize_purchase_date_text(value)
        date_matches = list(
            re.finditer(r"(\d{1,3})/(\d{1,2})/(\d{2,4})", value)
        )

        if not date_matches:
            date_matches = list(
                re.finditer(
                    r"(\d{1,3})[.](\d{1,2})[./](\d{2,4})",
                    value,
                )
            )

        for match in reversed(date_matches):
            day_text, month_text, year_text = match.groups()
            day = cls._normalize_purchase_day(day_text)
            month = int(month_text)

            if day is not None and 1 <= month <= 12:
                year = year_text[-2:]
                return (
                    f"{day:02d}/{month:02d}/{year}",
                    match.span(),
                )

        digits = "".join(character for character in value if character.isdigit())

        if len(digits) >= 6:
            year = digits[-2:]
            month = int(digits[-4:-2])
            day = cls._normalize_purchase_day(digits[:-4])

            if day is not None and 1 <= month <= 12:
                first_digit = next(
                    (
                        index
                        for index, character in enumerate(value)
                        if character.isdigit()
                    ),
                    0,
                )
                return (
                    f"{day:02d}/{month:02d}/{year}",
                    (first_digit, len(value)),
                )

        return "", None

    @staticmethod
    def _clean_lot_number(value: str) -> str:
        match = re.search(r"\d{4,}", value)
        return match.group(0) if match else value.strip()

    @staticmethod
    def _parse_decimal(
        value: str,
        *,
        repair_weight: bool = False,
    ) -> float | None:
        normalized = (
            value.upper()
            .replace("O", "0")
            .replace("Q", "0")
            .replace("Z", "2")
        )
        numeric = re.sub(r"[^0-9.\-]", "", normalized.replace(",", ""))

        try:
            parsed = float(numeric) if numeric else None
        except ValueError:
            return None

        if parsed is None or not repair_weight:
            return parsed

        if parsed > 300 and "." not in numeric and len(numeric) == 4:
            parsed = float(numeric[-3:]) / 10

        if 0 < parsed < 80:
            parsed += 100

        if repair_weight and not DeliveryNoteExtractor._is_plausible_weight(
            parsed
        ):
            return None

        return parsed

    @classmethod
    def _postprocess_rows(
        cls,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not rows:
            return rows

        cls._deduplicate_rows(rows)
        rows.sort(
            key=lambda row: (
                row["serial_number"]
                if row["serial_number"] is not None
                else 10_000
            )
        )
        cls._repair_lot_numbers(rows)
        cls._repair_purchase_dates(rows)

        tbgr_prefixes = cls._collect_tbgr_prefixes(rows)

        for row in rows:
            cls._recover_tbgr_number(row, tbgr_prefixes)
            cls._recover_numeric_values(row)
            cls._reconcile_row_numerics(row)
            row["grade"] = cls._normalize_grade(str(row.get("grade", "")))

        cls._repair_rate_bale_bleed(rows)

        for row in rows:
            cls._repair_underweight_row(row)
            cls._reconcile_row_numerics(row)
        cls._strip_attached_handwriting_markers(rows)
        cls._restore_grower_name_spacing(rows)
        cls._agree_grower_names_by_tbgr(rows)
        cls._sync_grower_name_ocr(rows)

        return [
            row
            for row in rows
            if row.get("serial_number") is not None
            and str(row.get("grower_name", "")).strip().lower() != "total"
        ]

    @staticmethod
    def _sync_grower_name_ocr(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            ocr = str(row.get("grower_name_ocr") or "").strip()
            name = str(row.get("grower_name") or "").strip()
            if not ocr:
                row["grower_name_ocr"] = name

    # Longest plausible single component of a printed grower name. Anything
    # longer is treated as words OCR ran together rather than as one word.
    _MAX_NAME_WORD_LENGTH = 12

    @classmethod
    def _build_name_vocabulary(
        cls,
        rows: list[dict[str, Any]],
    ) -> set[str]:
        """Learn name words from the note itself.

        Words are harvested only from names OCR already split into several
        words, so its own run-together spellings are not fed back in and made
        to look correct. Growers repeat across rows, which is what lets a name
        read correctly on one row repair the same name joined up on another.

        A word is kept when the note backs it up a second time: it appears on
        more than one row, or anywhere but at the end of a name, or it opens or
        closes another grower's name. The handwritten list markers sit at the
        end of the name column and are different every time, so none of those
        tests hold for them.
        """
        occurrences: Counter[str] = Counter()
        before_last: set[str] = set()
        compact_names: list[str] = []

        for row in rows:
            name = str(row.get("grower_name", ""))
            compact_names.append(re.sub(r"[^A-Za-z]", "", name).upper())
            words = name.split()

            if len(words) < 2:
                continue

            for index, word in enumerate(words):
                # The printed names are upper case, so anything carrying lower
                # case or punctuation is handwriting and not a name word.
                if not word.isalpha() or not word.isupper():
                    continue

                if not 3 <= len(word) <= cls._MAX_NAME_WORD_LENGTH:
                    continue

                occurrences[word] += 1

                if index < len(words) - 1:
                    before_last.add(word)

        def bounds_other_names(word: str) -> bool:
            return (
                sum(
                    compact != word
                    and (
                        compact.startswith(word)
                        or compact.endswith(word)
                    )
                    for compact in compact_names
                )
                >= 2
            )

        return {
            word
            for word, count in occurrences.items()
            if count > 1
            or word in before_last
            or bounds_other_names(word)
        }

    @classmethod
    def _agree_grower_names_by_tbgr(
        cls,
        rows: list[dict[str, Any]],
    ) -> None:
        """Make rows that share a TBGR number agree on the grower's name.

        A TBGR number identifies one grower, so repeated rows are repeated
        readings of the same printed name. The letters are settled by majority
        vote and the spacing by whichever reading OCR split into most words.
        """
        names_by_tbgr: dict[str, list[str]] = defaultdict(list)

        for row in rows:
            tbgr_number = str(row.get("tbgr_number", "")).strip()
            grower_name = str(row.get("grower_name", "")).strip()

            if re.fullmatch(r"\d{8}", tbgr_number) and grower_name:
                names_by_tbgr[tbgr_number].append(grower_name)

        agreed: dict[str, str] = {}

        for tbgr_number, names in names_by_tbgr.items():
            if len(names) < 2:
                continue

            letter_counts = Counter(
                re.sub(r"[^A-Za-z]", "", name).upper() for name in names
            )
            letters, votes = letter_counts.most_common(1)[0]

            if votes * 2 > len(names):
                variants = [
                    name
                    for name in names
                    if re.sub(r"[^A-Za-z]", "", name).upper() == letters
                ]
                agreed[tbgr_number] = max(
                    variants,
                    key=lambda name: (len(name.split()), -len(name)),
                )
                continue

            # Three or more readings of one TBGR can repair a handwritten
            # suffix without any external grower database. Keep the common
            # prefix, then extend it only while a strict character majority
            # agrees. This turns SHEKARPL / SHEKAP / SHEKARF into SHEKAR, but
            # a two-reading pair is accepted only when one is the other's long
            # prefix and the difference is a short attached OCR suffix.
            if len(names) < 3:
                if len(names) == 2:
                    compact_pair = [
                        re.sub(r"[^A-Za-z]", "", name).upper()
                        for name in names
                    ]
                    shorter_index = min(
                        range(2),
                        key=lambda index: len(compact_pair[index]),
                    )
                    shorter = compact_pair[shorter_index]
                    longer = compact_pair[1 - shorter_index]

                    if (
                        len(shorter) >= 8
                        and longer.startswith(shorter)
                        and len(longer) - len(shorter) <= 4
                    ):
                        agreed[tbgr_number] = names[shorter_index]
                continue

            compact = [
                re.sub(r"[^A-Za-z]", "", name).upper()
                for name in names
            ]
            common_length = 0

            for characters in zip(*compact):
                if len(set(characters)) != 1:
                    break
                common_length += 1

            consensus = compact[0][:common_length]
            position = common_length

            while position < max(map(len, compact)):
                counts = Counter(
                    value[position]
                    for value in compact
                    if position < len(value)
                )

                if not counts:
                    break

                character, count = counts.most_common(1)[0]

                if count * 2 <= len(compact):
                    break

                consensus += character
                position += 1

            if (
                len(consensus) < 8
                or len(consensus) < min(map(len, compact)) * 0.8
            ):
                continue

            consensus_variants = [
                name
                for name, value in zip(names, compact)
                if value.startswith(consensus)
            ]
            base = max(
                consensus_variants or names,
                key=lambda name: (len(name.split()), -len(name)),
            )
            agreed[tbgr_number] = cls._truncate_name_letters(
                base,
                len(consensus),
            )

        for row in rows:
            tbgr_number = str(row.get("tbgr_number", "")).strip()
            agreed_name = agreed.get(tbgr_number)

            if not agreed_name:
                continue

            letters = re.sub(
                r"[^A-Za-z]",
                "",
                str(row.get("grower_name", "")),
            ).upper()

            agreed_letters = re.sub(
                r"[^A-Za-z]",
                "",
                agreed_name,
            ).upper()

            # When multiple rows with the same registration number establish
            # one grower, a completely missed OCR cell can be restored without
            # guessing from neighboring rows.
            if not letters:
                row["grower_name"] = agreed_name
                continue

            shared_prefix = 0

            for left, right in zip(letters, agreed_letters):
                if left != right:
                    break
                shared_prefix += 1

            # Exact readings are only respaced. A longer reading may also be
            # replaced when the agreed name is its prefix, which is the shape
            # produced by a handwritten suffix joined to the printed name.
            if (
                letters == agreed_letters
                or (
                    letters.startswith(agreed_letters)
                    and len(letters) - len(agreed_letters) <= 4
                )
                or (
                    abs(len(letters) - len(agreed_letters)) <= 4
                    and shared_prefix >= min(
                        len(letters),
                        len(agreed_letters),
                    ) * 0.85
                )
            ):
                row["grower_name"] = agreed_name

    @staticmethod
    def _truncate_name_letters(name: str, letter_count: int) -> str:
        kept: list[str] = []
        seen = 0

        for character in name:
            if character.isalpha():
                if seen >= letter_count:
                    break
                seen += 1
            kept.append(character)

        return "".join(kept).strip()

    @classmethod
    def _grower_name_needs_refinement(cls, name: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(name)).strip()
        compact = re.sub(r"[^A-Za-z]", "", normalized)

        if len(compact) < 4:
            return True

        final_word = normalized.split()[-1] if normalized.split() else ""
        return bool(
            cls._ATTACHED_HANDWRITING_SUFFIX.search(final_word)
            or len(compact) > 35
        )

    @classmethod
    def _prefer_refined_grower_name(
        cls,
        original: str,
        refined: str,
    ) -> bool:
        original = str(original).strip()
        refined = str(refined).strip()

        if not refined:
            return False
        if not original:
            return True

        original_letters = re.sub(r"[^A-Za-z]", "", original).upper()
        refined_letters = re.sub(r"[^A-Za-z]", "", refined).upper()

        if len(refined_letters) < 4:
            return False

        original_suspicious = cls._grower_name_needs_refinement(original)
        refined_suspicious = cls._grower_name_needs_refinement(refined)

        if original_suspicious and not refined_suspicious:
            return (
                original_letters.startswith(refined_letters)
                and len(original_letters) - len(refined_letters) <= 5
            )

        return (
            len(original_letters) > 35
            and len(refined_letters) < len(original_letters)
            and (
                original_letters.startswith(refined_letters)
                or refined_letters.startswith(original_letters)
            )
        )

    @classmethod
    def _segment_grower_name(
        cls,
        name: str,
        vocabulary: set[str],
    ) -> str | None:
        compact = re.sub(r"[^A-Za-z]", "", name).upper()

        if len(compact) < 4:
            return None

        length = len(compact)
        segments: list[str | None] = [None] * (length + 1)
        segments[0] = ""

        for index in range(1, length + 1):
            for start in range(max(0, index - 24), index):
                if segments[start] is None:
                    continue

                word = compact[start:index]

                if len(word) >= 3 and word in vocabulary:
                    segments[index] = f"{segments[start]} {word}".strip()
                    break

        return segments[length]

    @classmethod
    def _peel_known_words(
        cls,
        word: str,
        vocabulary: set[str],
    ) -> str | None:
        """Split a run-together word by peeling off words seen on this note.

        Used when the word cannot be covered by known words end to end, which
        is common because a grower may appear only once. Whatever cannot be
        matched is left untouched rather than guessed at.
        """
        core = re.sub(r"[^A-Za-z]", "", word).upper()
        prefixes: list[str] = []
        suffixes: list[str] = []
        peeled = True

        while peeled and len(core) > 6:
            peeled = False

            for size in range(min(cls._MAX_NAME_WORD_LENGTH, len(core) - 3), 2, -1):
                if core[:size] in vocabulary:
                    prefixes.append(core[:size])
                    core = core[size:]
                    peeled = True
                    break

            if peeled:
                continue

            for size in range(min(cls._MAX_NAME_WORD_LENGTH, len(core) - 3), 2, -1):
                if core[-size:] in vocabulary:
                    suffixes.insert(0, core[-size:])
                    core = core[:-size]
                    peeled = True
                    break

        parts = prefixes + ([core] if core else []) + suffixes

        return " ".join(parts) if len(parts) > 1 else None

    @classmethod
    def _strip_handwriting_noise(
        cls,
        name: str,
        vocabulary: set[str],
    ) -> str:
        """Drop the handwritten list markers scrawled across the name column.

        The printed names are upper case, so a trailing fragment carrying lower
        case letters is handwriting rather than print. A trailing very short
        word is also dropped unless it appears as a name word elsewhere on the
        note, which keeps real name endings such as RAO.
        """
        words = name.split()

        while words:
            last = words[-1]
            letters = re.sub(r"[^A-Za-z]", "", last)

            if not letters:
                words.pop()
                continue

            has_lower = any(
                character.islower() for character in letters
            )

            if len(letters) <= 4 and has_lower:
                words.pop()
                continue

            if (
                len(letters) <= 3
                and letters.upper() not in vocabulary
                and len(words) > 1
            ):
                words.pop()
                continue

            break

        if not words:
            return name.strip()

        # A marker written hard against the last name is read as one word, so
        # the printed part is recovered when it is a word seen elsewhere.
        last = re.sub(r"[^A-Za-z]", "", words[-1]).upper()

        if last not in vocabulary:
            for size in range(len(last) - 1, 3, -1):
                trailing = last[size:]

                # A trailing part that is itself a name word means two names
                # were run together, which the splitter handles instead.
                if (
                    last[:size] in vocabulary
                    and len(trailing) <= 4
                    and trailing not in vocabulary
                ):
                    words[-1] = last[:size]
                    break

        return " ".join(words).strip()

    @staticmethod
    def _repair_rate_bale_bleed(rows: list[dict[str, Any]]) -> None:
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                row["serial_number"]
                if row["serial_number"] is not None
                else 10_000
            ),
        )

        for index in range(len(ordered_rows) - 1, 0, -1):
            current_row = ordered_rows[index - 1]
            next_row = ordered_rows[index]
            rate = current_row.get("rate_per_kg")
            bale_value = current_row.get("bale_value")
            next_weight = next_row.get("weight")

            if (
                rate is None
                or bale_value is None
                or next_weight is None
                or next_row.get("rate_per_kg") is not None
            ):
                continue

            expected_bale = round(float(next_weight) * float(rate), 2)

            if abs(expected_bale - float(bale_value)) > 2:
                continue

            current_weight = current_row.get("weight")

            if (
                current_weight is not None
                and abs(
                    round(float(current_weight) * float(rate), 2)
                    - float(bale_value)
                )
                <= 2
            ):
                continue

            next_row["rate_per_kg"] = rate
            next_row["bale_value"] = bale_value
            current_row["rate_per_kg"] = None
            current_row["bale_value"] = None
            DeliveryNoteExtractor._recover_numeric_values(current_row)
            DeliveryNoteExtractor._recover_numeric_values(next_row)

    @classmethod
    def _restore_grower_name_spacing(
        cls,
        rows: list[dict[str, Any]],
    ) -> None:
        vocabulary = cls._build_name_vocabulary(rows)

        for row in rows:
            grower_name = str(row.get("grower_name", "")).strip()

            if not grower_name:
                continue

            grower_name = cls._strip_handwriting_noise(
                grower_name,
                vocabulary,
            )

            # OCR joins printed words unpredictably, so each word that is not
            # itself a known name is re-split into known words where possible.
            rebuilt: list[str] = []

            for word in grower_name.split():
                if len(word) <= 3 or word.upper() in vocabulary:
                    rebuilt.append(word)
                    continue

                rebuilt.append(
                    cls._segment_grower_name(word, vocabulary)
                    or cls._peel_known_words(word, vocabulary)
                    or word
                )

            rebuilt_name = " ".join(rebuilt).strip()

            if rebuilt_name:
                row["grower_name"] = rebuilt_name

    @classmethod
    def _strip_attached_handwriting_markers(
        cls,
        rows: list[dict[str, Any]],
    ) -> None:
        """Remove row-list marks that OCR joined to the printed grower name.

        Short endings such as AP or P are stripped only when several stronger
        marker readings on the same note establish that handwriting is present.
        A lone final P additionally requires the same TBGR on at least three
        rows, preventing ordinary one-off names from being shortened.
        """
        names = [
            str(row.get("grower_name", "")).strip()
            for row in rows
        ]
        strong_marker_count = sum(
            bool(re.search(r"(?:APV|PPLB|PLB|AP)$", name, re.IGNORECASE))
            for name in names
        )
        rows_by_tbgr: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for row in rows:
            tbgr_number = str(row.get("tbgr_number", "")).strip()

            if re.fullmatch(r"\d{8}", tbgr_number):
                rows_by_tbgr[tbgr_number].append(row)

        # On a short crop, stronger marker variants may be outside the image.
        # Three differing readings of one grower that all end in P provide
        # enough local evidence that P is the attached handwritten stroke.
        for repeated_rows in rows_by_tbgr.values():
            repeated_names = [
                str(row.get("grower_name", "")).strip()
                for row in repeated_rows
            ]

            if (
                len(repeated_names) >= 3
                and len(set(repeated_names)) > 1
                and all(
                    re.search(r"(?<=[A-Za-z]{5})P$", name)
                    for name in repeated_names
                )
            ):
                for row in repeated_rows:
                    row["grower_name"] = re.sub(
                        r"P$",
                        "",
                        str(row.get("grower_name", "")).strip(),
                    )

        if strong_marker_count < 2:
            return

        tbgr_counts = Counter(
            str(row.get("tbgr_number", "")).strip()
            for row in rows
            if re.fullmatch(
                r"\d{8}",
                str(row.get("tbgr_number", "")).strip(),
            )
        )

        for row in rows:
            name = str(row.get("grower_name", "")).strip()

            if not name:
                continue

            # The printed RAO ending is frequently fused with a PPLB marker,
            # with its O becoming the first stroke of the handwritten suffix.
            cleaned = re.sub(
                r"RAPPLB$",
                "RAO",
                name,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"(?:APV|PPLB|PLB)$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            if cleaned == name:
                cleaned = re.sub(
                    r"AP$",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )

            tbgr_number = str(row.get("tbgr_number", "")).strip()

            if (
                cleaned == name
                and tbgr_counts.get(tbgr_number, 0) >= 3
            ):
                cleaned = re.sub(
                    r"(?<=[A-Za-z]{5})P$",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )

            row["grower_name"] = cleaned.strip() or name

    @staticmethod
    def _collect_tbgr_prefixes(rows: list[dict[str, Any]]) -> list[str]:
        """Learn the TBGR platform prefixes used by this note.

        Every note is printed for one or two auction platforms, so the numbers
        OCR read cleanly tell us which prefixes the damaged ones must share.
        """
        prefix_counts: Counter[str] = Counter()

        for row in rows:
            tbgr_number = str(row.get("tbgr_number", "")).strip()

            if re.fullmatch(r"\d{8}", tbgr_number):
                prefix_counts[tbgr_number[:3]] += 1

        return [prefix for prefix, _ in prefix_counts.most_common()]

    @classmethod
    def _recover_tbgr_from_text(
        cls,
        text: str,
        prefixes: list[str],
    ) -> str:
        """Recover a TBGR number whose digits OCR partly read as letters.

        Candidates are accepted only when they match a prefix taken from the
        rows OCR read cleanly, which keeps the substitutions unambiguous.
        """
        if not prefixes:
            return ""

        compact = re.sub(r"[^0-9A-Za-z]", "", text).upper()

        for start in range(max(0, len(compact) - 7)):
            window = compact[start : start + 8]
            letter_positions = [
                index
                for index, character in enumerate(window)
                if not character.isdigit()
            ]

            if not letter_positions or len(letter_positions) > 2:
                continue

            choices = [
                cls._TBGR_LETTER_DIGITS.get(window[index], "")
                for index in letter_positions
            ]

            if not all(choices):
                continue

            for replacement in product(*choices):
                characters = list(window)

                for index, digit in zip(letter_positions, replacement):
                    characters[index] = digit

                candidate = "".join(characters)

                if candidate.startswith(tuple(prefixes)):
                    return candidate

        return ""

    @classmethod
    def _recover_tbgr_number(
        cls,
        row: dict[str, Any],
        prefixes: list[str],
    ) -> None:
        tbgr_number = str(row.get("tbgr_number", "")).strip()

        if re.fullmatch(r"\d{8}", tbgr_number):
            return

        combined = " ".join(
            str(row.get(field, ""))
            for field in ("tbgr_number", "grower_name", "date_of_purchase")
        )
        compact = combined.replace(" ", "")

        # The serial or the purchase date can end up joined to the number, so
        # every eight-digit window inside a run of digits is considered and the
        # note's own prefixes decide which one is the TBGR.
        for run in re.findall(r"\d+", compact):
            for start in range(len(run) - 7):
                candidate = run[start : start + 8]

                if not prefixes or candidate.startswith(tuple(prefixes)):
                    row["tbgr_number"] = candidate
                    cls._strip_tbgr_digits_from_name(row, candidate)
                    return

        recovered = cls._recover_tbgr_from_text(combined, prefixes)
        row["tbgr_number"] = recovered

        if recovered:
            cls._strip_tbgr_digits_from_name(row, recovered)

    @staticmethod
    def _strip_tbgr_digits_from_name(
        row: dict[str, Any],
        tbgr_number: str,
    ) -> None:
        """Remove a TBGR that was read as part of the grower name."""
        name = str(row.get("grower_name", ""))

        if not name:
            return

        cleaned = name.replace(tbgr_number, " ")
        cleaned = re.sub(r"\d", " ", cleaned)
        row["grower_name"] = re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _deduplicate_rows(rows: list[dict[str, Any]]) -> None:
        best_by_serial: dict[int, dict[str, Any]] = {}
        rows_without_serial: list[dict[str, Any]] = []

        for row in rows:
            serial_number = row.get("serial_number")

            if serial_number is None:
                rows_without_serial.append(row)
                continue

            current = best_by_serial.get(serial_number)
            populated_fields = sum(
                value not in ("", None)
                for name, value in row.items()
                if name != "serial_number"
            )
            current_fields = (
                sum(
                    value not in ("", None)
                    for name, value in current.items()
                    if name != "serial_number"
                )
                if current is not None
                else -1
            )

            if populated_fields > current_fields:
                best_by_serial[serial_number] = row

        if any(serial_number <= 5 for serial_number in best_by_serial):
            plausible_limit = max(100, len(best_by_serial) * 2)
            best_by_serial = {
                serial_number: row
                for serial_number, row in best_by_serial.items()
                if serial_number <= plausible_limit
            }

        rows[:] = list(best_by_serial.values()) + rows_without_serial

    @staticmethod
    def _repair_lot_numbers(rows: list[dict[str, Any]]) -> None:
        lengths = [
            len(str(row["lot_number"]))
            for row in rows
            if str(row["lot_number"]).isdigit()
            and 4 <= len(str(row["lot_number"])) <= 6
        ]

        if not lengths:
            return

        expected_length = int(median(lengths))

        for row in rows:
            lot_number = str(row["lot_number"])

            if (
                lot_number.isdigit()
                and len(lot_number) > expected_length
            ):
                row["lot_number"] = lot_number[:expected_length]

    @classmethod
    def _repair_purchase_dates(cls, rows: list[dict[str, Any]]) -> None:
        """Keep valid OCR dates and expose malformed values for refinement."""
        for row in rows:
            try:
                parsed = datetime.strptime(
                    str(row["date_of_purchase"]),
                    "%d/%m/%y",
                )
            except ValueError:
                row["date_of_purchase"] = ""
                continue

            row["date_of_purchase"] = parsed.strftime("%d/%m/%y")

    @staticmethod
    def _is_plausible_weight(value: float | None) -> bool:
        return (
            value is not None
            and DeliveryNoteExtractor._MIN_WEIGHT
            <= value
            <= DeliveryNoteExtractor._MAX_WEIGHT
        )

    @classmethod
    def _typical_rate(cls, rows: list[dict[str, Any]]) -> float | None:
        rates = [
            float(row["rate_per_kg"])
            for row in rows
            if row.get("rate_per_kg") is not None
            and 100 <= float(row["rate_per_kg"]) <= 500
        ]

        if not rates:
            return None

        rounded_rates = [round(rate) for rate in rates]
        return float(max(set(rounded_rates), key=rounded_rates.count))

    @classmethod
    def _select_row_grade(cls, values: list[str]) -> str:
        if not values:
            return ""

        normalized = [cls._normalize_grade(value) for value in values]
        valid = [
            grade
            for grade in normalized
            if cls._GRADE_PATTERN.fullmatch(grade)
        ]

        if valid:
            return valid[0]

        return cls._normalize_grade(values[0])

    @staticmethod
    def _recover_numeric_values(
        row: dict[str, Any],
    ) -> None:
        weight = row["weight"]
        second_weight = row["second_weight"]
        rate = row["rate_per_kg"]
        bale_value = row["bale_value"]

        if weight is not None and not DeliveryNoteExtractor._is_plausible_weight(
            weight
        ):
            row["weight"] = None
            weight = None

        if (
            second_weight is not None
            and not DeliveryNoteExtractor._is_plausible_weight(second_weight)
        ):
            row["second_weight"] = None
            second_weight = None

        if (
            rate is not None
            and rate > 1000
            and float(rate).is_integer()
        ):
            repaired_rate = int(rate) % 1000

            if 100 <= repaired_rate <= 500:
                row["rate_per_kg"] = float(repaired_rate)
                rate = float(repaired_rate)

        if (
            weight is None
            and second_weight is not None
            and DeliveryNoteExtractor._is_plausible_weight(second_weight)
        ):
            row["weight"] = second_weight
            weight = second_weight

        candidate_weight: float | None = None

        if rate is not None and bale_value is not None and rate > 0:
            candidate = round(bale_value / rate, 1)

            if DeliveryNoteExtractor._is_plausible_weight(candidate):
                candidate_weight = candidate

        if weight is None and candidate_weight is not None:
            row["weight"] = candidate_weight
            weight = candidate_weight
        elif (
            weight is not None
            and second_weight is not None
            and candidate_weight is not None
            and (
                weight < second_weight
                or abs(weight - second_weight) > 5
            )
            and candidate_weight >= second_weight
            and abs(candidate_weight - second_weight) <= 5
        ):
            row["weight"] = candidate_weight
            weight = candidate_weight

        if (
            rate is None
            and weight is not None
            and bale_value is not None
            and weight > 0
        ):
            inferred_rate = round(bale_value / weight)

            if (
                DeliveryNoteExtractor._MIN_RATE
                <= inferred_rate
                <= DeliveryNoteExtractor._MAX_RATE
            ):
                row["rate_per_kg"] = float(inferred_rate)
                rate = float(inferred_rate)

        if (
            weight is not None
            and rate is not None
            and bale_value is None
        ):
            row["bale_value"] = round(weight * rate, 2)

    @staticmethod
    def _reconcile_row_numerics(row: dict[str, Any]) -> None:
        """Force weight, rate, and bale value to satisfy bale = weight x rate.

        The printed bale value is the product of the other two cells, so when
        the three disagree the bale value is the best evidence of the true
        weight. The weight column is also the least reliable one to read: it is
        narrow and the notes carry hand-drawn strokes across it.
        """
        weight = row.get("weight")
        rate = row.get("rate_per_kg")
        bale_value = row.get("bale_value")
        second_weight = row.get("second_weight")

        if (
            rate is None
            and weight is not None
            and bale_value is not None
            and float(weight) > 0
        ):
            inferred_rate = round(float(bale_value) / float(weight))

            if (
                DeliveryNoteExtractor._MIN_RATE
                <= inferred_rate
                <= DeliveryNoteExtractor._MAX_RATE
            ):
                row["rate_per_kg"] = float(inferred_rate)
                rate = row["rate_per_kg"]

        if rate is None or float(rate) <= 0:
            return

        rate_value = float(rate)

        if bale_value is None:
            if weight is not None and float(weight) > 0:
                row["bale_value"] = round(float(weight) * rate_value, 2)
            return

        implied_weight = round(float(bale_value) / rate_value, 1)
        implied_is_usable = (
            DeliveryNoteExtractor._is_plausible_weight(implied_weight)
            and (
                second_weight is None
                or abs(implied_weight - float(second_weight)) <= 5.0
            )
        )

        if weight is None or float(weight) <= 0:
            if implied_is_usable:
                row["weight"] = implied_weight
            return

        weight_value = float(weight)

        # Any disagreement beyond rounding means a cell was misread. Trust the
        # bale value when the implied weight also agrees with the second weight.
        if abs(implied_weight - weight_value) > 0.05 and implied_is_usable:
            row["weight"] = implied_weight
            return

        row["bale_value"] = round(weight_value * rate_value, 2)

    @staticmethod
    def _repair_underweight_row(row: dict[str, Any]) -> None:
        """Recover a weight that OCR dropped or read outside the valid range."""
        weight = row.get("weight")
        bale_value = row.get("bale_value")
        rate = row.get("rate_per_kg")

        if bale_value is None or rate is None or float(rate) <= 0:
            return

        if DeliveryNoteExtractor._is_plausible_weight(weight):
            return

        repaired_weight = round(float(bale_value) / float(rate), 1)

        if DeliveryNoteExtractor._is_plausible_weight(repaired_weight):
            row["weight"] = repaired_weight

    @staticmethod
    def _default_column_ranges() -> dict[str, tuple[float, float]]:
        return {
            "left": (0.0, 0.37),
            "grower": (0.0, 0.30),
            "lot": (0.37, 0.42),
            "weight": (0.42, 0.465),
            "second_weight": (0.465, 0.505),
            "grade": (0.505, 0.555),
            "rate": (0.555, 0.615),
            "bale": (0.615, 1.01),
        }

    @classmethod
    def _resolve_layout_name(
        cls,
        headers: list[tuple[str, _Token]],
        page_width: float,
        delivery_note_number: str = "",
    ) -> str:
        header_x = {
            name: token.center_x / page_width
            for name, token in headers
        }
        lot_x = header_x.get("lot_number", 0.5)
        grade_x = header_x.get("grade", 0.5)

        # Compact layout shifts lot/grade/rate columns right (lot ~0.54, grade ~0.80).
        if lot_x >= 0.52 and grade_x >= 0.72:
            return "compact"

        # Wide layout keeps lot near the left margin (lot ~0.40).
        if lot_x < 0.52:
            return "wide"

        return "compact"

    @staticmethod
    def _build_totals_check(
        rows: list[dict[str, Any]],
        totals: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Reconcile the extracted rows against the printed footer totals.

        The footer is an independent copy of the same figures, so agreement is
        strong evidence that every row was read correctly.
        """
        def column_sum(field: str) -> float | None:
            values = [
                float(row[field])
                for row in rows
                if row.get(field) is not None
            ]

            if len(values) != len(rows) or not values:
                return None

            return round(sum(values), 2)

        computed_weight = column_sum("weight")
        computed_second_weight = column_sum("second_weight")
        computed_bale = column_sum("bale_value")
        printed_weight = (totals or {}).get("total_weight")
        printed_bale = (totals or {}).get("total_bale_value")
        printed_row_count = (totals or {}).get("printed_row_count")

        def compare(
            computed: float | None,
            printed: float | None,
            tolerance: float,
        ) -> bool | None:
            if computed is None or printed is None:
                return None

            return abs(computed - float(printed)) <= tolerance

        return {
            "computed_total_weight": computed_weight,
            "printed_total_weight": printed_weight,
            "total_weight_matches": compare(
                computed_weight,
                printed_weight,
                0.5,
            ),
            "computed_total_bale_value": computed_bale,
            "printed_total_bale_value": printed_bale,
            "total_bale_value_matches": compare(
                computed_bale,
                printed_bale,
                1.0,
            ),
            "printed_row_count": printed_row_count,
            "row_count_matches": (
                None
                if printed_row_count is None
                else int(printed_row_count) == len(rows)
            ),
            "computed_total_second_weight": computed_second_weight,
        }

    @classmethod
    def _build_extraction_meta(
        cls,
        rows: list[dict[str, Any]],
        *,
        layout_name: str = "unknown",
        totals: dict[str, Any] | None = None,
        source: str = "image",
    ) -> dict[str, Any]:
        if not rows:
            return {
                "layout": layout_name,
                "source": source,
                "rows_extracted": 0,
                "rows_complete": 0,
                "quality_percent": 0.0,
                "issue_rows": [],
                "totals_check": cls._build_totals_check([], totals),
                "canonical_ocr_width": ImageService.CANONICAL_OCR_WIDTH,
                "row_source": "token_geometry",
            }

        complete_rows = 0
        issue_rows: list[dict[str, Any]] = []

        for row in rows:
            missing_fields: list[str] = []
            uncertain_fields: list[str] = []

            if not row.get("tbgr_number"):
                missing_fields.append("tbgr_number")
            if not row.get("grower_name"):
                missing_fields.append("grower_name")
            elif cls._grower_name_needs_refinement(
                str(row.get("grower_name", ""))
            ):
                uncertain_fields.append("grower_name")
            if row.get("rate_per_kg") is None:
                missing_fields.append("rate_per_kg")
            if row.get("bale_value") is None:
                missing_fields.append("bale_value")
            if row.get("weight") is None:
                missing_fields.append("weight")

            if missing_fields or uncertain_fields:
                issue = {
                    "serial_number": row.get("serial_number"),
                    "missing_fields": missing_fields,
                }

                if uncertain_fields:
                    issue["uncertain_fields"] = uncertain_fields

                issue_rows.append(issue)
            elif (
                row.get("tbgr_number")
                and row.get("grower_name")
                and row.get("rate_per_kg") is not None
                and row.get("bale_value") is not None
                and row.get("weight") is not None
            ):
                complete_rows += 1

        row_count = len(rows)
        quality_percent = (
            round(100 * complete_rows / row_count, 1)
            if row_count
            else 0.0
        )

        return {
            "layout": layout_name,
            "source": source,
            "rows_extracted": row_count,
            "rows_complete": complete_rows,
            "quality_percent": quality_percent,
            "issue_rows": issue_rows[:12],
            "totals_check": cls._build_totals_check(rows, totals),
            "canonical_ocr_width": ImageService.CANONICAL_OCR_WIDTH,
            "row_source": "token_geometry",
        }

    @classmethod
    def _build_column_ranges(
        cls,
        headers: list[tuple[str, _Token]],
        page_width: float,
        *,
        layout_name: str | None = None,
    ) -> dict[str, tuple[float, float]]:
        resolved_layout = layout_name or cls._resolve_layout_name(
            headers,
            page_width,
        )
        template = cls._LAYOUT_TEMPLATES.get(resolved_layout)

        if template is not None:
            return dict(template)

        header_x = {
            name: token.center_x / page_width
            for name, token in headers
        }
        defaults = cls._default_column_ranges()

        def pick(name: str, fallback: float) -> float:
            return header_x.get(name, fallback)

        lot_x = pick("lot_number", defaults["lot"][0])
        weight_x = pick("weight", defaults["weight"][0])
        second_x = pick("second_weight", defaults["second_weight"][0])
        date_x = pick("date_of_purchase", defaults["left"][1] - 0.04)

        if (
            lot_x < 0.56
            and weight_x < 0.63
            and pick("grade", defaults["grade"][0]) < 0.78
        ):
            return dict(cls._LAYOUT_TEMPLATES["compact"])

        if abs(weight_x - second_x) < 0.03:
            second_x = weight_x + 0.045

        grade_x = pick("grade", defaults["grade"][0])
        rate_x = pick("rate_per_kg", defaults["rate"][0])
        bale_x = pick("bale_value", defaults["bale"][0])

        def midpoint(left: float, right: float) -> float:
            return (left + right) / 2

        left_end = lot_x - 0.015

        return {
            "left": (0.0, left_end),
            "grower": (0.0, max(date_x, left_end * 0.82)),
            "lot": (left_end, midpoint(lot_x, weight_x)),
            "weight": (midpoint(lot_x, weight_x), midpoint(weight_x, second_x)),
            "second_weight": (
                midpoint(weight_x, second_x),
                midpoint(second_x, grade_x),
            ),
            "grade": (midpoint(second_x, grade_x), midpoint(grade_x, rate_x)),
            "rate": (midpoint(grade_x, rate_x), midpoint(rate_x, bale_x)),
            "bale": (midpoint(rate_x, bale_x), 1.01),
        }

    @staticmethod
    def _column_for_position(
        relative_x: float,
        ranges: dict[str, tuple[float, float]],
    ) -> str:
        ordered = (
            "lot",
            "weight",
            "second_weight",
            "grade",
            "rate",
            "bale",
        )

        for name in ordered:
            start, end = ranges[name]
            if start <= relative_x < end:
                return name

        if relative_x < ranges["left"][1]:
            return "left"

        return "bale"

    @staticmethod
    def _clean_grower_name(value: str) -> str:
        cleaned = DeliveryNoteExtractor._HANDWRITTEN_NAME_MARKERS.sub(
            " ",
            value,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"LIST$", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"L2[LO]$", "", cleaned, flags=re.IGNORECASE).strip()

        # Each row carries a hand-written marker between the name and the date.
        # OCR often fuses it onto the final printed word, so strip a trailing
        # marker only when enough letters remain in front of it to be a name.
        cleaned = re.sub(
            r"(?<=[A-Za-z]{3})(?:L[I1][A-Z]{0,2}|LAST[O]?|L[1-4][A-Z]{1,2})$",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"\s+(?:L[I1][A-Z]{0,2}|LAST[O]?|L[1-4][A-Z]{1,2}|VB|LA|AST[O]?)$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        cleaned = re.sub(r"\s+l$", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"RACL$", "RAO", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+[A-Z]$", "", cleaned).strip()
        cleaned = re.sub(r"[^A-Za-z ]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _extract_totals(
        cls,
        tokens: list[_Token],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        totals: dict[str, Any] = {
            "row_count": len(rows),
            "printed_row_count": None,
            "total_weight": None,
            "total_second_weight": None,
            "total_bale_value": None,
        }

        if not rows:
            return totals

        page_width = max(token.right for token in tokens)
        serial_numbers = {
            int(row["serial_number"])
            for row in rows
            if row.get("serial_number") is not None
        }
        last_serial = max(serial_numbers) if serial_numbers else None
        last_row_bottom = 0.0

        if last_serial is not None:
            serial_bottoms = [
                token.bottom
                for token in tokens
                if token.text.strip() == str(last_serial)
                and token.center_x < page_width * 0.12
            ]
            last_row_bottom = max(serial_bottoms) if serial_bottoms else 0.0

        footer_tokens = [
            token
            for token in tokens
            if token.top >= last_row_bottom - 5
            and re.search(r"total", token.text, re.IGNORECASE)
        ]

        if not footer_tokens:
            return totals

        # The footer figures are typeset slightly above the "Total" label, so the
        # cut-off is scaled to the label height rather than a fixed pixel margin.
        footer_label = min(footer_tokens, key=lambda token: token.top)
        footer_top = footer_label.top - (footer_label.height * 0.75)
        footer_area = [
            token
            for token in tokens
            if token.center_y >= footer_top
        ]
        numeric_values: list[tuple[str, float, float]] = []

        for token in footer_area:
            relative_x = token.center_x / page_width
            numeric = re.sub(r"[^0-9.\-]", "", token.text.replace(",", ""))

            if not numeric or not re.search(r"\d", numeric):
                continue

            try:
                parsed = float(numeric)
            except ValueError:
                continue

            if parsed <= 0:
                continue

            numeric_values.append((token.text, parsed, relative_x))

        row_count_candidates = [
            parsed
            for _, parsed, relative_x in numeric_values
            if 1 <= parsed <= 100
            and relative_x < 0.65
        ]

        if row_count_candidates:
            totals["row_count"] = int(row_count_candidates[0])
            totals["printed_row_count"] = int(row_count_candidates[0])

        weight_candidates = [
            parsed
            for _, parsed, relative_x in numeric_values
            if 500 <= parsed <= 10000
            and 0.55 <= relative_x <= 0.72
        ]
        second_weight_candidates = [
            parsed
            for _, parsed, relative_x in numeric_values
            if 500 <= parsed <= 10000
            and 0.68 <= relative_x <= 0.78
        ]
        bale_candidates = [
            parsed
            for _, parsed, relative_x in numeric_values
            if parsed >= 10000
            and relative_x >= 0.85
        ]

        if weight_candidates:
            totals["total_weight"] = cls._align_total_with_rows(
                weight_candidates[0],
                rows,
                "weight",
            )

        if second_weight_candidates:
            totals["total_second_weight"] = cls._align_total_with_rows(
                second_weight_candidates[0],
                rows,
                "second_weight",
            )

        if bale_candidates:
            totals["total_bale_value"] = cls._align_total_with_rows(
                max(bale_candidates),
                rows,
                "bale_value",
            )

        return totals

    @staticmethod
    def _align_total_with_rows(
        printed: float,
        rows: list[dict[str, Any]],
        field: str,
    ) -> float:
        """Repair a footer total whose decimal digits OCR misread.

        Only the fraction is replaced, and only when the integer part already
        matches the sum of the rows, so a total that genuinely disagrees is
        still reported exactly as printed and can be flagged.
        """
        values = [
            float(row[field])
            for row in rows
            if row.get(field) is not None
        ]

        if not values or len(values) != len(rows):
            return printed

        computed = round(sum(values), 2)

        if int(computed) == int(printed) and computed != printed:
            return computed

        return printed

    @staticmethod
    def _table_body_start(headers: list[tuple[str, _Token]]) -> float:
        header_tops = [token.top for _, token in headers]
        header_heights = [token.height for _, token in headers]
        return max(header_tops) + (median(header_heights) * 0.35)

    @classmethod
    def _normalize_grade(cls, value: str) -> str:
        """Normalize a grade cell for either printed grade family.

        The F family prints digits (F02, F02M, F03) while the L/X family ends
        in a letter (L2L, L2O, X2L), so zero/letter-O handling must differ.
        """
        compact = re.sub(r"[^A-Za-z0-9]", "", value).upper()

        if not compact:
            return ""

        if compact in cls._GRADE_DIRECT_MAP:
            return cls._GRADE_DIRECT_MAP[compact]

        # OCR commonly swaps the leading grade letter.
        compact = re.sub(r"^E(?=[0-9OC])", "F", compact)
        compact = re.sub(r"^1(?=[0-9O])", "L", compact)

        if compact.startswith("F"):
            body = compact[1:]
            suffix = ""

            if len(body) == 3 and body[2].isalpha():
                suffix = body[2]
                body = body[:2]

            digits = body.translate(cls._DIGIT_CONFUSIONS)

            if re.fullmatch(r"\d{1,2}", digits):
                return f"F{digits.zfill(2)}{suffix}"
        elif compact[0] in {"L", "X"}:
            body = compact[1:]

            if body:
                digit = body[0].translate(cls._DIGIT_CONFUSIONS)
                suffix = body[1:2].replace("0", "O")

                if digit.isdigit() and suffix in {"", "L", "O"}:
                    return f"{compact[0]}{digit}{suffix}"

        match = cls._GRADE_PATTERN.search(compact)

        if match:
            return match.group(0).upper()

        return value.strip()

    @classmethod
    def _match_column_header(cls, normalized: str) -> str | None:
        compact = normalized.replace(" ", "")

        keyword_rules = (
            ("tbgr_number", lambda: "tbgr" in compact and "no" in compact),
            (
                "grower_name",
                lambda: "grower" in compact
                or "nameofthegrower" in compact
                or "nameofthe" in compact,
            ),
            (
                "date_of_purchase",
                lambda: "purchase" in compact or normalized == "date of",
            ),
            (
                "lot_number",
                lambda: compact.startswith("lot")
                or compact.startswith("lol"),
            ),
            ("second_weight", lambda: compact.startswith("second")),
            ("weight", lambda: normalized == "weight"),
            ("grade", lambda: "grade" in compact),
            (
                "rate_per_kg",
                lambda: "rateper" in compact
                or "perkg" in compact
                or normalized == "kg.",
            ),
            (
                "bale_value",
                lambda: "bale" in compact or "value(rs" in compact,
            ),
            (
                "serial_number",
                lambda: normalized in {"s. no", "s no", "s.no", "s.no."},
            ),
        )

        for name, predicate in keyword_rules:
            if predicate():
                return name

        return None

    @classmethod
    def _find_table_headers(
        cls,
        tokens: list[_Token],
    ) -> list[tuple[str, _Token]]:
        matches: list[tuple[str, _Token]] = []

        for token in tokens:
            normalized = cls._normalize(token.text)
            matched_name: str | None = None

            for name, expressions in cls._COLUMN_LABELS.items():
                if any(
                    re.search(expression, normalized, re.IGNORECASE)
                    for expression in expressions
                ):
                    matched_name = name
                    break

            if matched_name is None:
                matched_name = cls._match_column_header(normalized)

            if matched_name is not None:
                matches.append((matched_name, token))

        if not matches:
            return []

        tolerance = max(median(token.height for _, token in matches), 10.0)
        groups: list[list[tuple[str, _Token]]] = []

        for match in sorted(matches, key=lambda value: value[1].center_y):
            for group in groups:
                group_center = median(
                    token.center_y for _, token in group
                )
                if abs(match[1].center_y - group_center) <= tolerance:
                    group.append(match)
                    break
            else:
                groups.append([match])

        best_group = max(groups, key=len)
        unique_headers: dict[str, _Token] = {}

        for name, token in best_group:
            current = unique_headers.get(name)
            if current is None or token.confidence > current.confidence:
                unique_headers[name] = token

        return list(unique_headers.items())

    @staticmethod
    def _group_rows(
        tokens: list[_Token],
        tolerance: float,
    ) -> list[list[_Token]]:
        rows: list[list[_Token]] = []

        for token in sorted(tokens, key=lambda value: value.center_y):
            for row in rows:
                row_center = median(value.center_y for value in row)
                if abs(token.center_y - row_center) <= tolerance:
                    row.append(token)
                    break
            else:
                rows.append([token])

        return [
            sorted(row, key=lambda value: value.left)
            for row in rows
        ]

    @classmethod
    def _is_known_label(cls, text: str) -> bool:
        normalized = cls._normalize(text).rstrip(":=- ")
        expressions = (
            expression
            for labels in (
                *cls._FIELD_LABELS.values(),
                *cls._COLUMN_LABELS.values(),
            )
            for expression in labels
        )
        return any(
            re.fullmatch(expression, normalized, re.IGNORECASE)
            for expression in expressions
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().lower()

    @staticmethod
    def _clean_value(value: str) -> str:
        return value.strip().strip(".:=-| ")

    @staticmethod
    def _normalize_printed_date(value: str) -> str:
        match = re.search(
            r"(\d{1,2}/\d{1,2}/\d{4})\s*(\d{1,2}:\d{2})?",
            value,
        )

        if not match:
            return value

        return " ".join(part for part in match.groups() if part)

    @staticmethod
    def _digits_only(value: str) -> str:
        return "".join(character for character in value if character.isdigit())

    @staticmethod
    def _empty_item() -> dict[str, Any]:
        return {
            "serial_number": None,
            "tbgr_number": "",
            "grower_name": "",
            "grower_name_ocr": "",
            "date_of_purchase": "",
            "lot_number": "",
            "weight": None,
            "second_weight": None,
            "grade": "",
            "rate_per_kg": None,
            "bale_value": None,
        }

    @staticmethod
    def _convert_item_value(name: str, value: str) -> Any:
        cleaned = value.strip()

        if name == "serial_number":
            digits = re.sub(r"\D", "", cleaned)
            return int(digits) if digits else None

        if name in {
            "weight",
            "second_weight",
            "rate_per_kg",
            "bale_value",
        }:
            numeric = re.sub(r"[^0-9.\-]", "", cleaned.replace(",", ""))
            try:
                return float(numeric) if numeric else None
            except ValueError:
                return None

        return cleaned
