# MCP Server — same incident tool, hosted for another process to load
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MCP Server")

import requests
from urllib.request import urlopen

import os #import os utility
#load environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Placeholder API key for the weather provider (e.g. OpenWeatherMap).
# Fill this in or export WEATHER_API_KEY in the environment / .env file.
# ---------------------------------------------------------------------------
WEATHER_API_KEY: str | None = os.environ["WEATHER_API_KEY"]  # TODO: placeholder - put your weather API key here

@mcp.tool()
def calculate_incident_impact(failed_requests: int, total_requests: int, baseline_error_rate_pct: float) -> dict:
    """Calculate request failure percentage and percentage-point change from baseline.
    Use counts from the same observation window.
    """
    rate = failed_requests / total_requests * 100
    return {"error_rate_pct": rate, "change_percentage_points": rate - baseline_error_rate_pct}

@mcp.tool()
def get_live_weather(city: str) -> str:
    """
    Get the current weather conditions (temperature, humidity, sky state, wind) for a given city anywhere in the world.

    Use this tool whenever the user asks about the weather, temperature, forecast or climate conditions in a specific location. The city name
    should be plain text, e.g. 'San Francisco' or 'London'. Requires a valid weather provider API key (WEATHER_API_KEY) to perform the live API call.
    """
    if not WEATHER_API_KEY:
        return ("Weather lookup unavailable: no API key configured. "
                "Set WEATHER_API_KEY in Prompts/Tools.py or the environment.")

  #  --- Real API call (OpenWeatherMap-style, untested without a key) ---
    
    url = "https://api.openweathermap.org/data/2.5/weather"
    resp = requests.get(url, params={"q": city, "appid": WEATHER_API_KEY, "units": "metric"}, timeout=10)
    data = resp.json()
    if resp.status_code != 200:
        return f"Weather lookup failed for '{city}': {data.get('message', resp.status_code)}"
    main = data["main"]; wind = data["wind"]; sky = data["weather"][0]["description"]
    return (f"Weather in {city}: {sky}, {main['temp']}°C "
            f"(feels like {main['feels_like']}°C), humidity {main['humidity']}%, "
            f"wind {wind.get('speed', 0)} m/s")


if __name__ == "__main__":
    mcp.run()
