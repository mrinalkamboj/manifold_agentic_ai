
'''
import json
from urllib.request import urlopen
'''

from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# ---------------------------------------------------------------------------
# Weather tool (API call with a placeholder API key)
# ---------------------------------------------------------------------------
@tool
def get_live_weather(city: str) -> str:
    """Get the current weather conditions (temperature, humidity, sky state, wind)
    for a given city anywhere in the world.

    Use this tool whenever the user asks about the weather, temperature,
    forecast or climate conditions in a specific location. The city name
    should be plain text, e.g. 'San Francisco' or 'London'. Requires a valid
    weather provider API key (WEATHER_API_KEY) to perform the live API call.
    """
    if not WEATHER_API_KEY:
        return ("Weather lookup unavailable: no API key configured. "
                "Set WEATHER_API_KEY in Prompts/Tools.py or the environment.")

  #  --- Real API call (OpenWeatherMap-style, untested without a key) ---
    import requests
    url = "https://api.openweathermap.org/data/2.5/weather"
    resp = requests.get(url, params={"q": city, "appid": WEATHER_API_KEY, "units": "metric"}, timeout=10)
    data = resp.json()
    if resp.status_code != 200:
        return f"Weather lookup failed for '{city}': {data.get('message', resp.status_code)}"
    main = data["main"]; wind = data["wind"]; sky = data["weather"][0]["description"]
    return (f"Weather in {city}: {sky}, {main['temp']}°C "
            f"(feels like {main['feels_like']}°C), humidity {main['humidity']}%, "
            f"wind {wind.get('speed', 0)} m/s")

    # # Placeholder response until the API key is configured
    # return f"It's always sunny in {city}! (placeholder - configure WEATHER_API_KEY for live data)"


'''
@tool
def get_live_weather(latitude: float, longitude: float) -> dict:
    """Fetch current weather from the Open-Meteo public API. No API key.
    Bangalore is latitude 12.97 longitude 77.59. Use decimal degrees.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&current=temperature_2m,weather_code"
    )
    with urlopen(url, timeout=10) as response:
        data = json.loads(response.read())
    return data.get("current", data)
'''

TOOLS = [get_live_weather]