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

        // 7. Map entities and safely repair deterministic OCR errors.
        var deliveryNote = MapToDeliveryNote(document);
        return DeliveryNotePostProcessor.Process(deliveryNote);
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
                    item.Weight = ParseDecimal(GetNumericText(property));
                    break;
                case "second_weight":
                    item.SecondWeight = ParseDecimal(value);
                    break;
                case "grade":
                    item.Grade = value;
                    break;
                case "rate_per_kg":
                    item.RatePerKg = ParseDecimal(value);
                    break;
                case "bale_value":
                    item.BaleValue = ParseDecimal(GetNumericText(property));
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

        return item;
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
}