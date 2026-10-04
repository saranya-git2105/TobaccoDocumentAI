using System.Diagnostics;
using Google.Cloud.DocumentAI.V1;
using Google.Protobuf;
using Microsoft.Extensions.Options;
using TobaccoDocumentAI.API.Configuration;
using TobaccoDocumentAI.API.Models;

namespace TobaccoDocumentAI.API.Services;

public class DocumentAIService : IDocumentAIService
{
    private readonly GoogleDocumentAIOptions _options;
    private readonly ILogger<DocumentAIService> _logger;

    public DocumentAIService(
        IOptions<GoogleDocumentAIOptions> options,
        ILogger<DocumentAIService> logger)
    {
        _options = options.Value;
        _logger = logger;
    }

    public async Task<DeliveryNote> ProcessDocumentAsync(
        Stream fileStream,
        string mimeType,
        CancellationToken cancellationToken = default)
    {
        var started = Stopwatch.StartNew();

        // 1. Set credentials for Google SDK
        Environment.SetEnvironmentVariable(
            "GOOGLE_APPLICATION_CREDENTIALS",
            _options.CredentialsPath);

        // 2. Create Document AI client
        var client = await DocumentProcessorServiceClient.CreateAsync(cancellationToken);

        // 3. Build processor resource name
        var processorName = ProcessorName.FromProjectLocationProcessor(
            _options.ProjectId,
            _options.Location,
            _options.ProcessorId);

        // 4. Read uploaded file into bytes
        using var memoryStream = new MemoryStream();
        await fileStream.CopyToAsync(memoryStream, cancellationToken);
        var fileBytes = memoryStream.ToArray();

        // 5. Build the request
        var request = new ProcessRequest
        {
            Name = processorName.ToString(),
            RawDocument = new RawDocument
            {
                Content = ByteString.CopyFrom(fileBytes),
                MimeType = mimeType
            }
        };

        // 6. Call Document AI
        var response = await client.ProcessDocumentAsync(request, cancellationToken);
        var document = response.Document;
        LogExtractorEntities(document);

        // 7. Map extractor entities only. Do not derive missing numbers or totals.
        var deliveryNote = MapToDeliveryNote(document);
        deliveryNote = DeliveryNotePostProcessor.Process(deliveryNote);
        FinalizeContract(deliveryNote, document, mimeType, started);
        return deliveryNote;
    }

    private void LogExtractorEntities(Document document)
    {
        _logger.LogInformation(
            "Document AI returned {EntityCount} entities across {PageCount} pages.",
            document.Entities.Count,
            document.Pages.Count);

        foreach (var entity in document.Entities)
        {
            _logger.LogInformation(
                "Entity {Type} confidence={Confidence} mention={Mention} normalized={Normalized}",
                entity.Type,
                entity.Confidence,
                entity.MentionText,
                entity.NormalizedValue?.Text);

            foreach (var property in entity.Properties)
            {
                _logger.LogInformation(
                    "  Property {Type} confidence={Confidence} mention={Mention} normalized={Normalized}",
                    property.Type,
                    property.Confidence,
                    property.MentionText,
                    property.NormalizedValue?.Text);
            }
        }
    }

    private DeliveryNote MapToDeliveryNote(Document document)
    {
        var result = new DeliveryNote();

        foreach (var entity in document.Entities)
        {
            var type = GetSimpleType(entity.Type);
            var value = GetValue(entity);

            switch (type)
            {
                case "delivery_note_number":
                    result.DeliveryNoteNumber = value;
                    break;
                case "delivery_date":
                    result.DeliveryDate = value;
                    break;
                case "buyer_name":
                    result.BuyerName = value;
                    break;
                case "auction_platform_number":
                    result.AuctionPlatformNumber =
                        new string(value.Where(char.IsDigit).ToArray());
                    break;
                case "auction_platform_name":
                    result.AuctionPlatformName = value;
                    break;
                case "code_number":
                    result.CodeNumber = value;
                    break;
                case "printed_date":
                    result.PrintedDate = value;
                    break;
                case "items":
                case "line_item":
                    result.Items.Add(MapItem(entity, result.Items.Count));
                    break;
                case "totals":
                case "total":
                case "totals_row":
                case "total_row":
                case "row_count":
                case "printed_row_count":
                case "total_weight":
                case "total_second_weight":
                case "total_bale_value":
                    MapTotals(entity, result.Totals);
                    break;
            }
        }

        return result;
    }

    private DeliveryNoteItem MapItem(Document.Types.Entity entity, int itemIndex)
    {
        var item = new DeliveryNoteItem();

        foreach (var property in entity.Properties)
        {
            var type = GetSimpleType(property.Type);
            var value = GetValue(property);

            _logger.LogTrace(
                "Document AI item {ItemIndex}: Type={Type}, Confidence={Confidence}, Mention={Mention}, Normalized={Normalized}",
                itemIndex,
                property.Type,
                property.Confidence,
                property.MentionText,
                property.NormalizedValue?.Text);

            switch (type)
            {
                case "serial_number":
                    item.SerialNumber = ParseInt(value);
                    break;
                case "tbgr_number":
                case "tbgr_no":
                case "grower_name":
                case "grower":
                    // Text fields are selected below by confidence.
                    break;
                case "purchase_date":
                    item.DateOfPurchase = GetMentionText(property);
                    break;
                case "date_of_purchase":
                    item.DateOfPurchase = value;
                    break;
                case "lot_number":
                    item.LotNumber = value;
                    break;
                case "weight":
                    item.Weight = ParseFirstDecimal(property);
                    break;
                case "second_weight":
                    item.SecondWeight = ParseFirstDecimal(property);
                    break;
                case "grade":
                    item.Grade = value;
                    break;
                case "rate_per_kg":
                    item.RatePerKg = ParseFirstDecimal(property);
                    break;
                case "bale_value":
                    item.BaleValue = ParseMoney(property);
                    break;
            }
        }

        item.TbgrNumber = GetBestTextProperty(
            entity.Properties,
            "tbgr_number",
            "tbgr_no");
        item.GrowerName = GetBestTextProperty(
            entity.Properties,
            "grower_name",
            "grower");
        item.GrowerNameOcr = item.GrowerName;

        return item;
    }

    private static void MapTotals(Document.Types.Entity entity, DeliveryNoteTotals totals)
    {
        if (entity.Properties.Count > 0)
        {
            foreach (var property in entity.Properties)
            {
                ApplyTotalField(totals, GetSimpleType(property.Type), property);
            }

            return;
        }

        ApplyTotalField(totals, GetSimpleType(entity.Type), entity);
    }

    private static void ApplyTotalField(
        DeliveryNoteTotals totals,
        string type,
        Document.Types.Entity entity)
    {
        switch (type)
        {
            case "row_count":
            case "printed_row_count":
            case "no_of_rows":
            case "number_of_rows":
            case "total_rows":
                var rowCount = ParseInt(GetValue(entity));
                if (rowCount is null)
                {
                    break;
                }

                if (type == "printed_row_count")
                {
                    totals.PrintedRowCount = rowCount;
                }
                else
                {
                    totals.RowCount = rowCount.Value;
                }

                totals.PrintedRowCount ??= rowCount;
                if (totals.RowCount == 0)
                {
                    totals.RowCount = rowCount.Value;
                }

                break;
            case "total_weight":
            case "weight":
                if (ParseFirstDecimal(entity) is decimal totalWeight)
                {
                    totals.TotalWeight = totalWeight;
                }

                break;
            case "total_second_weight":
            case "second_weight":
                if (ParseFirstDecimal(entity) is decimal totalSecondWeight)
                {
                    totals.TotalSecondWeight = totalSecondWeight;
                }

                break;
            case "total_bale_value":
            case "bale_value":
                if (ParseFirstDecimal(entity) is decimal totalBaleValue)
                {
                    totals.TotalBaleValue = totalBaleValue;
                }

                break;
        }
    }

    private static void FinalizeContract(
        DeliveryNote deliveryNote,
        Document document,
        string mimeType,
        Stopwatch started)
    {
        foreach (var item in deliveryNote.Items)
        {
            if (string.IsNullOrWhiteSpace(item.GrowerNameOcr))
            {
                item.GrowerNameOcr = item.GrowerName;
            }
        }

        var source = string.Equals(
            mimeType,
            "application/pdf",
            StringComparison.OrdinalIgnoreCase)
            ? "pdf"
            : "image";

        deliveryNote.ExtractionMeta = BuildExtractionMeta(
            deliveryNote.Items,
            deliveryNote.Totals,
            source);
        deliveryNote.ProcessingTimeSeconds = Math.Round(started.Elapsed.TotalSeconds, 3);
        deliveryNote.PageCount = Math.Max(document.Pages.Count, 1);
    }

    private static ExtractionMeta BuildExtractionMeta(
        IReadOnlyList<DeliveryNoteItem> items,
        DeliveryNoteTotals totals,
        string source)
    {
        var issueRows = new List<ExtractionIssueRow>();
        var completeRows = 0;

        foreach (var item in items)
        {
            var missingFields = new List<string>();

            if (string.IsNullOrWhiteSpace(item.TbgrNumber))
            {
                missingFields.Add("tbgr_number");
            }

            if (string.IsNullOrWhiteSpace(item.GrowerName))
            {
                missingFields.Add("grower_name");
            }

            if (item.RatePerKg is null)
            {
                missingFields.Add("rate_per_kg");
            }

            if (item.BaleValue is null)
            {
                missingFields.Add("bale_value");
            }

            if (item.Weight is null)
            {
                missingFields.Add("weight");
            }

            if (missingFields.Count > 0)
            {
                issueRows.Add(new ExtractionIssueRow
                {
                    SerialNumber = item.SerialNumber,
                    MissingFields = missingFields,
                });
            }
            else
            {
                completeRows++;
            }
        }

        var rowCount = items.Count;

        return new ExtractionMeta
        {
            Layout = "custom_extractor",
            Source = source,
            RowsExtracted = rowCount,
            RowsComplete = completeRows,
            QualityPercent = rowCount == 0
                ? 0
                : Math.Round(100.0 * completeRows / rowCount, 1),
            IssueRows = issueRows.Take(12).ToList(),
            TotalsCheck = new TotalsCheck
            {
                PrintedTotalWeight = totals.TotalWeight,
                PrintedTotalBaleValue = totals.TotalBaleValue,
                PrintedRowCount = totals.PrintedRowCount,
            },
            CanonicalOcrWidth = 0,
            RowSource = "document_ai_entities",
        };
    }

    private static string GetBestTextProperty(
        IEnumerable<Document.Types.Entity> properties,
        params string[] acceptedTypes)
    {
        var acceptedTypeSet = acceptedTypes.ToHashSet(StringComparer.OrdinalIgnoreCase);

        var bestProperty = properties
            .Where(property => acceptedTypeSet.Contains(GetSimpleType(property.Type)))
            .Where(property => !string.IsNullOrWhiteSpace(GetTextValue(property)))
            .OrderByDescending(property => property.Confidence)
            .FirstOrDefault();

        return bestProperty is null ? string.Empty : GetTextValue(bestProperty);
    }

    private static string GetSimpleType(string type)
    {
        return type.Split('/', StringSplitOptions.RemoveEmptyEntries)
                   .Last()
                   .Trim()
                   .Replace(' ', '_')
                   .Replace('-', '_')
                   .ToLowerInvariant();
    }

    private static string GetMentionText(Document.Types.Entity entity)
    {
        return entity.MentionText?.Trim() ?? string.Empty;
    }

    private static string GetTextValue(Document.Types.Entity entity)
    {
        return !string.IsNullOrWhiteSpace(entity.MentionText)
            ? entity.MentionText.Trim()
            : entity.NormalizedValue?.Text?.Trim() ?? string.Empty;
    }

    private static string GetNumericText(Document.Types.Entity entity)
    {
        return !string.IsNullOrWhiteSpace(entity.NormalizedValue?.Text)
            ? entity.NormalizedValue.Text
            : entity.MentionText?.Trim() ?? string.Empty;
    }

    private static string GetValue(Document.Types.Entity entity)
    {
        return !string.IsNullOrWhiteSpace(entity.NormalizedValue?.Text)
            ? entity.NormalizedValue.Text.Trim()
            : entity.MentionText?.Trim() ?? string.Empty;
    }

    private static int? ParseInt(string value)
    {
        return int.TryParse(value.Trim(), out var result) ? result : null;
    }

    private static decimal? ParseDecimal(string value)
    {
        var cleaned = value.Replace(",", "").Trim();

        return decimal.TryParse(
            cleaned,
            System.Globalization.NumberStyles.Number,
            System.Globalization.CultureInfo.InvariantCulture,
            out var result)
            ? result
            : null;
    }

    private static decimal? ParseFirstDecimal(Document.Types.Entity entity)
    {
        foreach (var candidate in new[] { entity.MentionText, entity.NormalizedValue?.Text })
        {
            if (string.IsNullOrWhiteSpace(candidate))
            {
                continue;
            }

            var match = System.Text.RegularExpressions.Regex.Match(
                candidate,
                @"\d+(?:,\d{3})*(?:\.\d+)?");

            if (!match.Success)
            {
                continue;
            }

            var parsed = ParseDecimal(match.Value);
            if (parsed is not null)
            {
                return parsed;
            }
        }

        return null;
    }

    private static decimal? ParseMoney(Document.Types.Entity entity)
    {
        foreach (var candidate in new[] { entity.MentionText, entity.NormalizedValue?.Text })
        {
            if (string.IsNullOrWhiteSpace(candidate))
            {
                continue;
            }

            var match = System.Text.RegularExpressions.Regex.Match(
                candidate,
                @"\d+(?:,\d{3})*(?:\.\d+)?");

            if (!match.Success)
            {
                continue;
            }

            var number = match.Value.Replace(",", "");
            var decimalPoint = number.IndexOf('.');
            if (decimalPoint >= 0 && number.Length - decimalPoint - 1 > 2)
            {
                number = number[..(decimalPoint + 3)];
            }

            var parsed = ParseDecimal(number);
            if (parsed is not null)
            {
                return parsed;
            }
        }

        return null;
    }
}