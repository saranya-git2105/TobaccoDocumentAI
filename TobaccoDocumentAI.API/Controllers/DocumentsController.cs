using Microsoft.AspNetCore.Mvc;
using TobaccoDocumentAI.API.Services;

namespace TobaccoDocumentAI.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class DocumentsController : ControllerBase
{
    private readonly IDocumentAIService _documentAIService;

    public DocumentsController(IDocumentAIService documentAIService)
    {
        _documentAIService = documentAIService;
    }

    [HttpPost("extract")]
    [RequestSizeLimit(10 * 1024 * 1024)] // 10 MB
    public async Task<IActionResult> Extract(IFormFile file, CancellationToken cancellationToken)
    {
        if (file == null || file.Length == 0)
            return BadRequest(new { error = "No file uploaded." });

        var allowedTypes = new[]
        {
            "application/pdf",
            "image/png",
            "image/jpeg",
            "image/jpg"
        };

        if (!allowedTypes.Contains(file.ContentType))
            return BadRequest(new { error = "Only PDF and image files are allowed." });

        using var stream = file.OpenReadStream();

        var result = await _documentAIService.ProcessDocumentAsync(
            stream,
            file.ContentType,
            cancellationToken);

        return Ok(result);
    }
}