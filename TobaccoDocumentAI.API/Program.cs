using TobaccoDocumentAI.API.Configuration;
using TobaccoDocumentAI.API.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.Configure<GoogleDocumentAIOptions>(
    builder.Configuration.GetSection(GoogleDocumentAIOptions.SectionName));

builder.Services.AddScoped<IDocumentAIService, DocumentAIService>();
builder.Services.AddOpenApi();

builder.Services.AddCors(options =>
{
    options.AddPolicy("ReactPolicy", policy =>
    {
        policy.AllowAnyHeader()
              .AllowAnyMethod()
              .AllowAnyOrigin();
    });
});

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.UseSwaggerUI(options =>
    {
        options.SwaggerEndpoint("/openapi/v1.json", "v1");
    });
}

app.UseCors("ReactPolicy");

app.MapControllers();

app.Run();
