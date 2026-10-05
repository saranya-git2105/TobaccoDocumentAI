using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Options;
using TobaccoDocumentAI.API.Configuration;

namespace TobaccoDocumentAI.API.Authentication;

public class StringValueJsonMiddleware
{
    private readonly RequestDelegate _next;
    private readonly bool _enabled;

    public StringValueJsonMiddleware(RequestDelegate next, IOptions<StringValueJsonOptions> options)
    {
        _next = next;
        _enabled = options.Value.Enabled;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        if (!_enabled || IsDocumentationRequest(context.Request.Path))
        {
            await _next(context);
            return;
        }

        var originalBody = context.Response.Body;
        await using var buffer = new MemoryStream();
        context.Response.Body = buffer;

        try
        {
            await _next(context);

            buffer.Position = 0;
            if (!IsJsonResponse(context.Response) || buffer.Length == 0)
            {
                await CopyAsync(buffer, originalBody);
                return;
            }

            JsonNode converted;
            try
            {
                using var document = await JsonDocument.ParseAsync(buffer);
                converted = ToStringValues(document.RootElement);
            }
            catch (JsonException)
            {
                buffer.Position = 0;
                await CopyAsync(buffer, originalBody);
                return;
            }

            var json = converted.ToJsonString();
            var bytes = Encoding.UTF8.GetBytes(json);
            context.Response.ContentLength = bytes.Length;
            context.Response.Body = originalBody;
            await context.Response.Body.WriteAsync(bytes);
        }
        finally
        {
            context.Response.Body = originalBody;
        }
    }

    private static bool IsDocumentationRequest(PathString path)
    {
        return path.StartsWithSegments("/swagger") || path.StartsWithSegments("/openapi");
    }

    private static bool IsJsonResponse(HttpResponse response)
    {
        return response.ContentType?.Contains("application/json", StringComparison.OrdinalIgnoreCase) == true;
    }

    private static async Task CopyAsync(Stream source, Stream destination)
    {
        source.Position = 0;
        await source.CopyToAsync(destination);
    }

    private static JsonNode ToStringValues(JsonElement element)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                var obj = new JsonObject();
                foreach (var property in element.EnumerateObject())
                {
                    obj[property.Name] = ToStringValues(property.Value);
                }

                return obj;
            case JsonValueKind.Array:
                var array = new JsonArray();
                foreach (var item in element.EnumerateArray())
                {
                    array.Add(ToStringValues(item));
                }

                return array;
            case JsonValueKind.String:
                return JsonValue.Create(element.GetString())!;
            case JsonValueKind.Null:
            case JsonValueKind.Undefined:
                return JsonValue.Create(string.Empty)!;
            default:
                return JsonValue.Create(element.GetRawText())!;
        }
    }
}
