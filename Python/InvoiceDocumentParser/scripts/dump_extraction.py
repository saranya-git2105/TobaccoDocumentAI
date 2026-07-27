"""Dump the full extraction result for each sample note as JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.image_service import ImageService
from app.services.ocr_service import OcrService
from scripts.audit_extraction import JUNE_IMAGE, MAY_IMAGE
from scripts.ground_truth import FIELD_NAMES, JUNE_ROWS, MAY_ROWS, as_dicts

OUTPUT = Path("build_output")


def main() -> None:
    service = OcrService()
    OUTPUT.mkdir(exist_ok=True)
    summary: dict[str, dict] = {}

    for label, path, truth_rows in (
        ("june", JUNE_IMAGE, JUNE_ROWS),
        ("may", MAY_IMAGE, MAY_ROWS),
    ):
        image, _ = ImageService.prepare_document_array(path)
        document = service.extract_text_from_image(image)
        expected = as_dicts(truth_rows)
        rows = []

        for row in document["items"]:
            want = expected.get(row["serial_number"], {})
            rows.append(
                {
                    "row": row,
                    "wrong": sorted(
                        field
                        for field in FIELD_NAMES
                        if field in want and row.get(field) != want[field]
                    ),
                    "expected": want,
                }
            )

        summary[label] = {
            "header": {
                key: document.get(key)
                for key in (
                    "document_type",
                    "delivery_note_number",
                    "delivery_date",
                    "buyer_name",
                    "auction_platform_number",
                    "auction_platform_name",
                    "code_number",
                    "printed_date",
                )
            },
            "totals": document.get("totals"),
            "extraction_meta": document.get("extraction_meta"),
            "rows": rows,
        }

    (OUTPUT / "extraction.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT / 'extraction.json'}")

    for label, data in summary.items():
        wrong_rows = sum(1 for row in data["rows"] if row["wrong"])
        print(
            f"{label}: rows={len(data['rows'])} "
            f"rows_with_any_mismatch={wrong_rows}"
        )


if __name__ == "__main__":
    main()
