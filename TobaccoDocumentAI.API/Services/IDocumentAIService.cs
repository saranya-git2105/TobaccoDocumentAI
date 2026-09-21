using TobaccoDocumentAI.API.Models;

namespace TobaccoDocumentAI.API.Services;

public interface IDocumentAIService
{
    Task<DeliveryNote> ProcessDocumentAsync(
        Stream fileStream,
        string mimeType,
        CancellationToken cancellationToken = default);
}