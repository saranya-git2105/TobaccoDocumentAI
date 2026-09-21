from app.services.tnote_contract import to_pascal_keys


def test_renames_keys_without_changing_shape() -> None:
    payload = to_pascal_keys(
        {
            "document_type": "Tobacco Board Delivery Note",
            "delivery_note_number": "28/130720260250A",
            "delivery_date": "13/07/2026",
            "buyer_name": "M/s. Maruthi Tobacco Suppliers",
            "auction_platform_number": "28",
            "auction_platform_name": "KALIGIRI",
            "code_number": "11",
            "printed_date": "17/08/2026 07:44",
            "items": [
                {
                    "serial_number": 1,
                    "tbgr_number": "28095067",
                    "grower_name": "KOPPOLU MURA",
                    "grower_name_ocr": "KOPPOLU MURA",
                    "date_of_purchase": "24/06/26",
                    "lot_number": "17140",
                    "weight": 132.1,
                    "second_weight": 132,
                    "grade": "X3L",
                    "rate_per_kg": 162,
                    "bale_value": 21400.2,
                }
            ],
            "totals": {
                "row_count": 1,
                "total_bale_value": 21400.2,
            },
            "extraction_meta": {
                "quality_percent": 98.2,
                "issue_rows": [
                    {
                        "serial_number": 18,
                        "missing_fields": ["weight"],
                    }
                ],
            },
            "processing_time_seconds": 5.1,
            "page_count": 2,
        }
    )

    assert payload["DeliveryNoteNumber"] == "28/130720260250A"
    assert payload["AuctionPlatformName"] == "KALIGIRI"
    assert payload["AuctionPlatformNumber"] == "28"
    assert payload["BuyerName"] == "M/s. Maruthi Tobacco Suppliers"
    assert payload["CodeNumber"] == "11"
    assert payload["PrintedDate"] == "17/08/2026 07:44"
    assert "Bales" not in payload
    assert "HeaderError" not in payload
    assert "Validation" not in payload
    assert "TNoteNumber" not in payload
    assert len(payload["Items"]) == 1

    item = payload["Items"][0]
    assert item["SerialNumber"] == 1
    assert item["TbgrNumber"] == "28095067"
    assert item["GrowerName"] == "KOPPOLU MURA"
    assert item["GrowerNameOcr"] == "KOPPOLU MURA"
    assert item["DateOfPurchase"] == "24/06/26"
    assert item["LotNumber"] == "17140"
    assert item["Weight"] == 132.1
    assert item["SecondWeight"] == 132
    assert item["Grade"] == "X3L"
    assert item["RatePerKg"] == 162
    assert item["BaleValue"] == 21400.2
    assert "Error" not in item
    assert "IsValueMismatch" not in item
    assert "TbgrNo" not in item
    assert "LotNo" not in item

    assert payload["Totals"]["RowCount"] == 1
    assert payload["Totals"]["TotalBaleValue"] == 21400.2
    assert payload["ExtractionMeta"]["QualityPercent"] == 98.2
    assert payload["ExtractionMeta"]["IssueRows"][0]["MissingFields"] == ["weight"]
    assert payload["ProcessingTimeSeconds"] == 5.1
    assert payload["PageCount"] == 2


if __name__ == "__main__":
    test_renames_keys_without_changing_shape()
    print("tnote contract checks passed")
