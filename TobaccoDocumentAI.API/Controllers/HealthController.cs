using Microsoft.AspNetCore.Mvc;
using System;

namespace TobaccoDocumentAI.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class HealthController : ControllerBase
{
    [HttpGet]
    public IActionResult Get()
    {
        return Ok(new
        {
            Status = "Backend Running",
            Time = DateTime.Now
        });
    }
}