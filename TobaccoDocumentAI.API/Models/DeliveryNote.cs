using System.Text.Json.Serialization;

namespace TobaccoDocumentAI.API.Models;

public class DeliveryNote
{
    [JsonPropertyName("document_type")]
    public string DocumentType { get; set; } = "Tobacco Board Delivery Note";

    [JsonPropertyName("delivery_note_number")]
    public string DeliveryNoteNumber { get; set; } = string.Empty;

    [JsonPropertyName("delivery_date")]
    public string DeliveryDate { get; set; } = string.Empty;

    [JsonPropertyName("buyer_name")]
    public string BuyerName { get; set; } = string.Empty;

    [JsonPropertyName("auction_platform_number")]
    public string AuctionPlatformNumber { get; set; } = string.Empty;

    [JsonPropertyName("auction_platform_name")]
    public string AuctionPlatformName { get; set; } = string.Empty;

    [JsonPropertyName("code_number")]
    public string CodeNumber { get; set; } = string.Empty;

    [JsonPropertyName("printed_date")]
    public string PrintedDate { get; set; } = string.Empty;

    [JsonPropertyName("items")]
    public List<DeliveryNoteItem> Items { get; set; } = [];
}