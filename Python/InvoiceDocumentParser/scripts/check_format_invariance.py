"""Verify the same note extracts identically from JPG, PNG, TIFF, and PDF."""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import img2pdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.image_service import ImageService
from app.services.ocr_service import OcrService
from scripts.audit_extraction import JUNE_IMAGE, MAY_IMAGE

COMPARED_FIELDS = (
    "delivery_note_number",
    "delivery_date",
    "buyer_name",
    "auction_platform_number",
    "auction_platform_name",
    "code_number",
    "items",
    "totals",
)


def fingerprint(document: dict) -> str:
    return json.dumps(
        {field: document.get(field) for field in COMPARED_FIELDS},
        sort_keys=True,
    )


def build_variants(source: Path, directory: Path) -> dict[str, Path]:
    image = ImageService.load_image_bgr(source)
    variants: dict[str, Path] = {}

    png_path = directory / "note.png"
    cv2.imwrite(str(png_path), image)
    variants["png"] = png_path

    jpg_path = directory / "note.jpg"
    cv2.imwrite(str(jpg_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    variants["jpg"] = jpg_path

    tif_path = directory / "note.tif"
    cv2.imwrite(str(tif_path), image)
    variants["tif"] = tif_path

    pdf_path = directory / "note.pdf"
    pdf_path.write_bytes(img2pdf.convert(str(png_path)))
    variants["pdf"] = pdf_path

    return variants


def check(label: str, source: Path, service: OcrService) -> bool:
    with tempfile.TemporaryDirectory() as directory:
        variants = build_variants(source, Path(directory))
        results = {}

        for name, path in variants.items():
            image, _ = ImageService.prepare_document_array(path)
            document = service.extract_text_from_image(image)
            results[name] = document

        baseline_name = "png"
        baseline = fingerprint(results[baseline_name])
        identical = True

        print(f"===== {label}")
        for name, document in results.items():
            same = fingerprint(document) == baseline
            identical = identical and same
            meta = document.get("extraction_meta", {})
            print(
                f"  {name:4s} rows={len(document.get('items', [])):2d} "
                f"complete={meta.get('rows_complete'):2d} "
                f"quality={meta.get('quality_percent')}% "
                f"identical_to_{baseline_name}={same}"
            )

        if not identical:
            reference = results[baseline_name]

            for name, document in results.items():
                if fingerprint(document) == baseline:
                    continue

                for field in COMPARED_FIELDS:
                    if field == "items":
                        continue

                    if reference.get(field) != document.get(field):
                        print(
                            f"    {name} {field}: "
                            f"png={reference.get(field)!r} "
                            f"{name}={document.get(field)!r}"
                        )

                for expected_row, actual_row in zip(
                    reference["items"],
                    document["items"],
                ):
                    if expected_row != actual_row:
                        print(
                            f"    {name} row "
                            f"{expected_row.get('serial_number')} differs:"
                        )
                        for key in expected_row:
                            if expected_row[key] != actual_row.get(key):
                                print(
                                    f"      {key}: png={expected_row[key]!r} "
                                    f"{name}={actual_row.get(key)!r}"
                                )
        print()
        return identical


def main() -> None:
    service = OcrService()
    ok = check("JUNE", JUNE_IMAGE, service)
    ok = check("MAY", MAY_IMAGE, service) and ok
    print("ALL FORMATS IDENTICAL" if ok else "FORMAT DIFFERENCES FOUND")


if __name__ == "__main__":
    main()
