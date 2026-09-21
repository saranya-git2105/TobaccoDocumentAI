namespace TobaccoDocumentAI.API.Models;

public class DeliveryNote
{
    public string DocumentType { get; set; } = "Tobacco Board Delivery Note";

    public string DeliveryNoteNumber { get; set; } = string.Empty;

    public string DeliveryDate { get; set; } = string.Empty;

    public string BuyerName { get; set; } = string.Empty;

    public string AuctionPlatformNumber { get; set; } = string.Empty;

    public string AuctionPlatformName { get; set; } = string.Empty;

    public string CodeNumber { get; set; } = string.Empty;

    public string PrintedDate { get; set; } = string.Empty;

    public List<DeliveryNoteItem> Items { get; set; } = [];

    public DeliveryNoteTotals Totals { get; set; } = new();

    public ExtractionMeta ExtractionMeta { get; set; } = new();

    public double ProcessingTimeSeconds { get; set; }

    public int PageCount { get; set; }
}

public class DeliveryNoteTotals
{
    public int RowCount { get; set; }

    public int? PrintedRowCount { get; set; }

    public decimal? TotalWeight { get; set; }

    public decimal? TotalSecondWeight { get; set; }

    public decimal? TotalBaleValue { get; set; }
}

public class ExtractionMeta
{
    public string Layout { get; set; } = "custom_extractor";

    public string Source { get; set; } = "image";

    public int RowsExtracted { get; set; }

    public int RowsComplete { get; set; }

    public double QualityPercent { get; set; }

    public List<ExtractionIssueRow> IssueRows { get; set; } = [];

    public TotalsCheck TotalsCheck { get; set; } = new();

    public int CanonicalOcrWidth { get; set; }

    public string RowSource { get; set; } = "document_ai_entities";
}

public class ExtractionIssueRow
{
    public int? SerialNumber { get; set; }

    public List<string> MissingFields { get; set; } = [];
}

public class TotalsCheck
{
    public decimal? ComputedTotalWeight { get; set; }

    public decimal? PrintedTotalWeight { get; set; }

    public bool? TotalWeightMatches { get; set; }

    public decimal? ComputedTotalBaleValue { get; set; }

    public decimal? PrintedTotalBaleValue { get; set; }

    public bool? TotalBaleValueMatches { get; set; }

    public int? PrintedRowCount { get; set; }

    public bool? RowCountMatches { get; set; }

    public decimal? ComputedTotalSecondWeight { get; set; }
}
