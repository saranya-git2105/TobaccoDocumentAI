using System.Security.Cryptography;
using System.Text;
using Microsoft.Extensions.Options;
using TobaccoDocumentAI.API.Configuration;

namespace TobaccoDocumentAI.API.Authentication;

public class BasicAuthenticationMiddleware
{
    private readonly RequestDelegate _next;
    private readonly BasicAuthOptions _options;

    public BasicAuthenticationMiddleware(RequestDelegate next, IOptions<BasicAuthOptions> options)
    {
        _next = next;
        _options = options.Value;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        if (!IsAuthorized(context.Request))
        {
            context.Response.StatusCode = StatusCodes.Status401Unauthorized;
            context.Response.Headers.WWWAuthenticate = "Basic realm=\"TobaccoDocumentAI\"";
            await context.Response.WriteAsJsonAsync(new { error = "Unauthorized" });
            return;
        }

        await _next(context);
    }

    private bool IsAuthorized(HttpRequest request)
    {
        if (string.IsNullOrEmpty(_options.Username) || string.IsNullOrEmpty(_options.Password))
        {
            return false;
        }

        if (!request.Headers.TryGetValue("Authorization", out var headerValue))
        {
            return false;
        }

        var header = headerValue.ToString();
        const string prefix = "Basic ";
        if (!header.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        string decoded;
        try
        {
            decoded = Encoding.UTF8.GetString(Convert.FromBase64String(header[prefix.Length..].Trim()));
        }
        catch (FormatException)
        {
            return false;
        }

        var separator = decoded.IndexOf(':');
        if (separator < 0)
        {
            return false;
        }

        var username = decoded[..separator];
        var password = decoded[(separator + 1)..];
        return FixedEquals(username, _options.Username) && FixedEquals(password, _options.Password);
    }

    private static bool FixedEquals(string actual, string expected)
    {
        var actualBytes = Encoding.UTF8.GetBytes(actual);
        var expectedBytes = Encoding.UTF8.GetBytes(expected);
        if (actualBytes.Length != expectedBytes.Length)
        {
            return false;
        }

        return CryptographicOperations.FixedTimeEquals(actualBytes, expectedBytes);
    }
}
