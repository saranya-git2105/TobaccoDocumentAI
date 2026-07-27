namespace TobaccoDocumentAI.API.Configuration;

public class GoogleDocumentAIOptions
{
    public const string SectionName = "GoogleDocumentAI";

    public string ProjectId { get; set; } = string.Empty;
    public string Location { get; set; } = string.Empty;
    public string ProcessorId { get; set; } = string.Empty;
    public string CredentialsPath { get; set; } = string.Empty;
}