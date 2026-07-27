"""Score extraction against the printed ground truth for the sample notes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.image_service import ImageService
from app.services.ocr_service import OcrService
from scripts.ground_truth import (
    FIELD_NAMES,
    JUNE_ROWS,
    MAY_ROWS,
    as_dicts,
)

ASSETS = Path(
    r"C:\Users\saranya\.cursor\projects\d-Projects-RDProjects-Dotnet-projects-TobaccoDocumentAI\assets"
)
JUNE_IMAGE = ASSETS / (
    "c__Users_saranya_AppData_Roaming_Cursor_User_workspaceStorage_"
    "dd5bbada8f6663f13de97ab8b818b01b_images_tnote-"
    "6e9a6165-27c9-43f7-bcb6-7a838f6afaf0.png"
)
MAY_IMAGE = ASSETS / (
    "c__Users_saranya_AppData_Roaming_Cursor_User_workspaceStorage_"
    "dd5bbada8f6663f13de97ab8b818b01b_images_T-Note_1-"
    "6372c8eb-94c8-4bbc-a393-c917458f2306.png"
)

NUMERIC_FIELDS = {"weight", "second_weight", "rate_per_kg", "bale_value"}


def values_match(field: str, expected, actual) -> bool:
    if actual is None:
        return False

    if field in NUMERIC_FIELDS:
        return abs(float(expected) - float(actual)) < 0.051

    return str(expected).strip().upper() == str(actual).strip().upper()


def audit(label: str, image_path: Path, expected_rows) -> None:
    image, _ = ImageService.prepare_document_array(image_path)
    result = OcrService().extract_text_from_image(image)
    actual = {
        row["serial_number"]: row
        for row in result["items"]
        if row.get("serial_number") is not None
    }
    expected = as_dicts(expected_rows)

    field_hits = dict.fromkeys(FIELD_NAMES, 0)
    total_cells = 0
    matched_cells = 0
    row_errors = []

    for serial, expected_row in expected.items():
        actual_row = actual.get(serial, {})
        mismatches = []

        for field in FIELD_NAMES:
            total_cells += 1

            if values_match(field, expected_row[field], actual_row.get(field)):
                field_hits[field] += 1
                matched_cells += 1
            else:
                mismatches.append(
                    f"{field}: want {expected_row[field]!r} "
                    f"got {actual_row.get(field)!r}"
                )

        if mismatches:
            row_errors.append((serial, mismatches))

    meta = result.get("extraction_meta", {})
    accuracy = 100 * matched_cells / total_cells if total_cells else 0.0
    perfect_rows = len(expected) - len(row_errors)

    check = meta.get("totals_check", {})

    print(f"===== {label}")
    print(f"  layout           {meta.get('layout')}")
    print(f"  rows found       {len(actual)} / {len(expected)}")
    print(f"  perfect rows     {perfect_rows} / {len(expected)}")
    print(f"  cell accuracy    {accuracy:.1f}%  ({matched_cells}/{total_cells})")
    print(
        f"  totals check     weight={check.get('total_weight_matches')} "
        f"bale={check.get('total_bale_value_matches')} "
        f"rows={check.get('row_count_matches')}"
    )
    print(
        f"                   computed {check.get('computed_total_weight')} kg"
        f" / {check.get('computed_total_bale_value')}"
        f"  printed {check.get('printed_total_weight')} kg"
        f" / {check.get('printed_total_bale_value')}"
    )
    print("  per-field hits:")
    for field in FIELD_NAMES:
        print(f"    {field:16s} {field_hits[field]:3d} / {len(expected)}")

    if row_errors:
        print("  mismatches:")
        for serial, mismatches in row_errors:
            print(f"    row {serial}:")
            for mismatch in mismatches:
                print(f"      {mismatch}")
    print()


def main() -> None:
    audit("JUNE (32 rows)", JUNE_IMAGE, JUNE_ROWS)
    audit("MAY (24 rows)", MAY_IMAGE, MAY_ROWS)


if __name__ == "__main__":
    main()
