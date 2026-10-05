namespace TobaccoDocumentAI.API.Configuration;

public class BasicAuthOptions
{
    public const string SectionName = "BasicAuth";

    public string Username { get; set; } = string.Empty;

    public string Password { get; set; } = string.Empty;
}
