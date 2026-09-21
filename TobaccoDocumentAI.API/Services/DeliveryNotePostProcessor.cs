using System.Text.RegularExpressions;
using TobaccoDocumentAI.API.Models;

namespace TobaccoDocumentAI.API.Services;

internal static partial class DeliveryNotePostProcessor
{
    private const decimal MinimumExpectedWeight = 50m;
    private const decimal MaximumExpectedWeight = 250m;
    private const decimal CalculationTolerance = 0.05m;

    public static DeliveryNote Process(DeliveryNote deliveryNote)
    {
        foreach (var item in deliveryNote.Items)
        {
            item.TbgrNumber = NormalizeTbgrNumber(item.TbgrNumber);
            item.GrowerName = NormalizeGrowerName(item.GrowerName);
            item.DateOfPurchase = CollapseWhitespace(item.DateOfPurchase);
            item.LotNumber = NormalizeLotNumber(item.LotNumber);
            item.Grade = CollapseWhitespace(item.Grade).ToUpperInvariant();

            RecoverMissingWeight(item);
        }

        FillRepeatedTextFields(deliveryNote.Items);

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

    private static string NormalizeLotNumber(string value)
    {
        var match = LotNumberRegex().Match(value ?? string.Empty);
        return match.Success
            ? match.Groups["lot"].Value
            : CollapseWhitespace(value);
    }

    private static void RecoverMissingWeight(DeliveryNoteItem item)
    {
        if (item.Weight.HasValue ||
            !item.RatePerKg.HasValue ||
            !item.BaleValue.HasValue ||
            item.RatePerKg.Value <= 0)
        {
            return;
        }

        var calculatedWeight = decimal.Round(
            item.BaleValue.Value / item.RatePerKg.Value,
            1,
            MidpointRounding.AwayFromZero);

        if (calculatedWeight is < MinimumExpectedWeight or > MaximumExpectedWeight)
        {
            return;
        }

        var calculatedBaleValue = calculatedWeight * item.RatePerKg.Value;

        if (decimal.Abs(calculatedBaleValue - item.BaleValue.Value) <= CalculationTolerance)
        {
            item.Weight = calculatedWeight;
        }
    }

    private static void FillRepeatedTextFields(IReadOnlyCollection<DeliveryNoteItem> items)
    {
        var growerByTbgr = BuildUniqueLookup(
            items,
            item => item.TbgrNumber,
            item => item.GrowerName,
            IsValidTbgrNumber);

        foreach (var item in items.Where(item =>
                     string.IsNullOrWhiteSpace(item.GrowerName) &&
                     IsValidTbgrNumber(item.TbgrNumber)))
        {
            if (growerByTbgr.TryGetValue(item.TbgrNumber, out var growerName))
            {
                item.GrowerName = growerName;
                if (string.IsNullOrWhiteSpace(item.GrowerNameOcr))
                {
                    item.GrowerNameOcr = growerName;
                }
            }
        }

        var tbgrByGrower = BuildUniqueLookup(
            items,
            item => item.GrowerName,
            item => item.TbgrNumber,
            key => !string.IsNullOrWhiteSpace(key));

        foreach (var item in items.Where(item =>
                     !IsValidTbgrNumber(item.TbgrNumber) &&
                     !string.IsNullOrWhiteSpace(item.GrowerName)))
        {
            if (tbgrByGrower.TryGetValue(item.GrowerName, out var tbgrNumber))
            {
                item.TbgrNumber = tbgrNumber;
            }
        }

        var dateByTbgr = BuildUniqueLookup(
            items,
            item => item.TbgrNumber,
            item => item.DateOfPurchase,
            IsValidTbgrNumber);

        foreach (var item in items.Where(item =>
                     string.IsNullOrWhiteSpace(item.DateOfPurchase) &&
                     IsValidTbgrNumber(item.TbgrNumber)))
        {
            if (dateByTbgr.TryGetValue(item.TbgrNumber, out var purchaseDate))
            {
                item.DateOfPurchase = purchaseDate;
            }
        }
    }

    private static Dictionary<string, string> BuildUniqueLookup(
        IEnumerable<DeliveryNoteItem> items,
        Func<DeliveryNoteItem, string> keySelector,
        Func<DeliveryNoteItem, string> valueSelector,
        Func<string, bool> validKey)
    {
        return items
            .Select(item => new
            {
                Key = keySelector(item),
                Value = valueSelector(item)
            })
            .Where(pair =>
                validKey(pair.Key) &&
                !string.IsNullOrWhiteSpace(pair.Value))
            .GroupBy(pair => pair.Key, StringComparer.OrdinalIgnoreCase)
            .Select(group => new
            {
                group.Key,
                Values = group
                    .Select(pair => pair.Value)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList()
            })
            .Where(group => group.Values.Count == 1)
            .ToDictionary(
                group => group.Key,
                group => group.Values[0],
                StringComparer.OrdinalIgnoreCase);
    }

    private static bool IsValidTbgrNumber(string value)
    {
        return TbgrNumberRegex().IsMatch(value ?? string.Empty);
    }

    private static string CollapseWhitespace(string? value)
    {
        return WhitespaceRegex().Replace(value?.Trim() ?? string.Empty, " ");
    }

    [GeneratedRegex(@"\D")]
    private static partial Regex DigitsRegex();

    [GeneratedRegex(@"^\D*(?<lot>\d{5})")]
    private static partial Regex LotNumberRegex();

    [GeneratedRegex(@"^\d{8}$")]
    private static partial Regex TbgrNumberRegex();

    [GeneratedRegex(@"\s+")]
    private static partial Regex WhitespaceRegex();
}
