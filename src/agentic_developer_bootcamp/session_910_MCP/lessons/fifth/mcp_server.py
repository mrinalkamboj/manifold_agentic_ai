# MCP Server — same incident tool, hosted for another process to load
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Incident MCP Server")


@mcp.tool()
def calculate_incident_impact(failed_requests: int, total_requests: int, baseline_error_rate_pct: float) -> dict:
    """Calculate request failure percentage and percentage-point change from baseline.
    Use counts from the same observation window.
    """
    rate = failed_requests / total_requests * 100
    return {"error_rate_pct": rate, "change_percentage_points": rate - baseline_error_rate_pct}


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b


if __name__ == "__main__":
    mcp.run()
