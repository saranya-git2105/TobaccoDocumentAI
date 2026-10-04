using System.Text.RegularExpressions;
using TobaccoDocumentAI.API.Models;

namespace TobaccoDocumentAI.API.Services;

internal static partial class DeliveryNotePostProcessor
{
    public static DeliveryNote Process(DeliveryNote deliveryNote)
    {
        foreach (var item in deliveryNote.Items)
        {
            item.TbgrNumber = NormalizeTbgrNumber(item.TbgrNumber);
            item.GrowerName = NormalizeGrowerName(item.GrowerName);
            SeparateLotFromGrowerName(item);
            item.DateOfPurchase = CollapseWhitespace(item.DateOfPurchase);
            SeparateWeightFromLotNumber(item);
            item.Grade = CollapseWhitespace(item.Grade).ToUpperInvariant();
        }

        deliveryNote.Items = deliveryNote.Items
            .OrderBy(item => item.SerialNumber ?? int.MaxValue)
            .ToList();

        return deliveryNote;
    }

    private static string NormalizeTbgrNumber(string value)
    {
        var digits = DigitsRegex().Replace(value ?? string.Empty, string.Empty);
        return digits.Length == 8 ? digits : CollapseWhitespace(value);
    }

    private static string NormalizeGrowerName(string value)
    {
        return CollapseWhitespace(value)
            .Replace('\u039A', 'K')
            .Replace('\u03BA', 'K')
            .Replace('\u03A1', 'P')
            .Replace('\u03C1', 'P')
            .ToUpperInvariant();
    }

    private static void SeparateLotFromGrowerName(DeliveryNoteItem item)
    {
        var match = GrowerWithLotRegex().Match(item.GrowerName ?? string.Empty);
        if (!match.Success)
        {
            return;
        }

        item.GrowerName = CollapseWhitespace(match.Groups["name"].Value);
        if (string.IsNullOrWhiteSpace(item.LotNumber))
        {
            item.LotNumber = match.Groups["lot"].Value;
        }
    }

    private static void SeparateWeightFromLotNumber(DeliveryNoteItem item)
    {
        var match = LotWithWeightRegex().Match(item.LotNumber ?? string.Empty);
        if (!match.Success)
        {
            item.LotNumber = CollapseWhitespace(item.LotNumber);
            return;
        }

        item.LotNumber = match.Groups["lot"].Value;
        if (item.Weight is not null)
        {
            return;
        }

        var weightText = match.Groups["glued"].Success
            ? match.Groups["glued"].Value
            : match.Groups["split"].Value;

        if (decimal.TryParse(
                weightText,
                System.Globalization.NumberStyles.Number,
                System.Globalization.CultureInfo.InvariantCulture,
                out var weight))
        {
            item.Weight = weight;
        }
    }

    private static string CollapseWhitespace(string? value)
    {
        return WhitespaceRegex().Replace(value?.Trim() ?? string.Empty, " ");
    }

    [GeneratedRegex(@"\D")]
    private static partial Regex DigitsRegex();

    [GeneratedRegex(@"^(?<name>[\p{L}][\p{L} .'-]*?)(?<lot>\d{5})(?:\D.*)?$")]
    private static partial Regex GrowerWithLotRegex();

    [GeneratedRegex(@"^\D*(?<lot>\d{5})(?:(?<glued>\d+\.\d+)|[^\d]+(?<split>\d+(?:\.\d+)?))?")]
    private static partial Regex LotWithWeightRegex();

    [GeneratedRegex(@"\s+")]
    private static partial Regex WhitespaceRegex();
}
