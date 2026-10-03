import json
import logging
from urllib.request import urlopen

from langchain_core.tools import tool
from pydantic import BaseModel, FiniteFloat

WEATHER_URL = "https://api.open-meteo.com/v1/forecast?latitude=19.08&longitude=72.88&current=temperature_2m&timezone=Asia%2FKolkata"


class CurrentWeather(BaseModel):
    time: str
    temperature_2m: FiniteFloat


def fetch_mumbai_weather(timeout: float) -> dict:
    with urlopen(WEATHER_URL, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError("Weather provider returned a non-success response")
        payload = json.loads(response.read(100_000))
    current = CurrentWeather.model_validate(payload["current"])
    if payload.get("current_units", {}).get("temperature_2m") != "°C":
        raise ValueError("Unexpected temperature unit")
    return {"status": "available", "city": "Mumbai", "temperature_c": current.temperature_2m,
            "observed_at": current.time, "source": WEATHER_URL, "fixture": False}


def make_weather_tool(settings, offline: bool = False):
    @tool
    def get_mumbai_weather() -> dict:
        """Get current temperature in Mumbai from Open-Meteo. No other city is supported."""
        if offline:
            return {"status": "available", "city": "Mumbai", "temperature_c": 29.5,
                    "observed_at": "DEMO-FIXTURE", "source": "offline fixture, not current weather", "fixture": True}
        try:
            return fetch_mumbai_weather(min(settings.timeout_seconds, 10))
        except Exception as exc:
            logging.getLogger("weather_tool").warning("weather_failed error_type=%s", type(exc).__name__)
            return {"status": "unavailable", "message": "Current Mumbai weather could not be fetched. Please try again later.", "fixture": False}
    return get_mumbai_weather
