using System.Text.Json.Serialization;

namespace TobaccoDocumentAI.API.Models;

public class DeliveryNoteItem
{
    [JsonPropertyName("serial_number")]
    public int? SerialNumber { get; set; }

    [JsonPropertyName("tbgr_number")]
    public string TbgrNumber { get; set; } = string.Empty;

    [JsonPropertyName("grower_name")]
    public string GrowerName { get; set; } = string.Empty;

    [JsonPropertyName("date_of_purchase")]
    public string DateOfPurchase { get; set; } = string.Empty;

    [JsonPropertyName("lot_number")]
    public string LotNumber { get; set; } = string.Empty;

    [JsonPropertyName("weight")]
    public decimal? Weight { get; set; }

    [JsonPropertyName("second_weight")]
    public decimal? SecondWeight { get; set; }

    [JsonPropertyName("grade")]
    public string Grade { get; set; } = string.Empty;

    [JsonPropertyName("rate_per_kg")]
    public decimal? RatePerKg { get; set; }

    [JsonPropertyName("bale_value")]
    public decimal? BaleValue { get; set; }
}