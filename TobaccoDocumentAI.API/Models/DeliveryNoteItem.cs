namespace TobaccoDocumentAI.API.Models;

public class DeliveryNoteItem
{
    public int? SerialNumber { get; set; }

    public string TbgrNumber { get; set; } = string.Empty;

    public string GrowerName { get; set; } = string.Empty;

    public string GrowerNameOcr { get; set; } = string.Empty;

    public string DateOfPurchase { get; set; } = string.Empty;

    public string LotNumber { get; set; } = string.Empty;

    public decimal? Weight { get; set; }

    public decimal? SecondWeight { get; set; }

    public string Grade { get; set; } = string.Empty;

    public decimal? RatePerKg { get; set; }

    public decimal? BaleValue { get; set; }
}
