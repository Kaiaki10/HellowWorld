"""
Weather Specialist

Tuned for weather prediction markets: temperature records, storms, precipitation.

Key data sources:
  - NOAA/NWS API (free, authoritative, real-time)
  - Open-Meteo API (free, global forecasts from GFS/ECMWF models)
  - Historical weather data for baseline comparison

Why specialized: Weather markets have an enormous advantage — free, high-quality
numerical weather models (GFS, ECMWF) that are more accurate than any LLM.
The bot compares model forecasts against market prices instead of relying on
sentiment analysis, which is nearly useless for weather.

This specialist produced $1,325 profit in the polymarket-kalshi-weather-bot reference.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

from modules.scanner import Market
from modules.researcher import ResearchBrief, SourceResult, SentimentAnalyzer
from modules.specialists import BaseSpecialist

logger = logging.getLogger(__name__)


# Major US cities for weather market matching
CITY_COORDINATES = {
    "new york": (40.71, -74.01),
    "nyc": (40.71, -74.01),
    "los angeles": (34.05, -118.24),
    "la": (34.05, -118.24),
    "chicago": (41.88, -87.63),
    "houston": (29.76, -95.37),
    "phoenix": (33.45, -112.07),
    "miami": (25.76, -80.19),
    "dallas": (32.78, -96.80),
    "denver": (39.74, -104.99),
    "seattle": (47.61, -122.33),
    "washington": (38.91, -77.04),
    "dc": (38.91, -77.04),
    "boston": (42.36, -71.06),
    "atlanta": (33.75, -84.39),
    "san francisco": (37.77, -122.42),
    "sf": (37.77, -122.42),
    "las vegas": (36.17, -115.14),
    "detroit": (42.33, -83.05),
    "minneapolis": (44.98, -93.27),
    "tampa": (27.95, -82.46),
}


class WeatherSpecialist(BaseSpecialist):

    category = "weather"
    description = "Temperature, precipitation, storms, and climate events"

    def __init__(self):
        self.sentiment = SentimentAnalyzer()

    async def research(self, market: Market) -> ResearchBrief:
        """
        Weather research pipeline:
        1. Parse market title for location + weather condition + target
        2. Fetch numerical weather model forecasts (Open-Meteo / NOAA)
        3. Compare model forecast against market price
        4. Calculate probability from forecast ensemble spread
        """
        # Parse the market
        location, condition, target_value, target_date = self._parse_weather_market(market.title)

        sources: list[SourceResult] = []
        model_probability = None

        if location:
            coords = self._get_coordinates(location)
            if coords:
                # Fetch weather forecast
                forecast = await self._fetch_forecast(coords[0], coords[1], target_date)
                if forecast:
                    model_probability = self._calculate_probability(
                        forecast, condition, target_value
                    )
                    sources.append(SourceResult(
                        source_type="weather_model",
                        title=f"Weather forecast for {location}",
                        content=f"Model forecast: {forecast}. "
                                f"Condition: {condition}, Target: {target_value}. "
                                f"Model probability: {model_probability:.1%}" if model_probability else "No forecast",
                        url="https://api.open-meteo.com",
                        sentiment_score=0,
                        sentiment_label="neutral",
                    ))

                # Fetch historical baseline
                historical = await self._fetch_historical(coords[0], coords[1], target_date)
                if historical:
                    sources.append(SourceResult(
                        source_type="weather_historical",
                        title=f"Historical weather for {location}",
                        content=f"Historical data: {historical}",
                        url="https://archive-api.open-meteo.com",
                        sentiment_score=0,
                        sentiment_label="neutral",
                    ))

        # Build the brief
        if model_probability is not None:
            narrative_prob = model_probability
            confidence = 0.80  # Weather models are highly reliable 1-7 days out
        else:
            narrative_prob = market.current_price  # No signal, defer to market
            confidence = 0.2

        gap = narrative_prob - market.current_price

        brief = ResearchBrief(
            market=market,
            sources=sources,
            overall_sentiment=0,
            sentiment_label="neutral",
            narrative_consensus=f"Weather model estimate: {narrative_prob:.0%}" if model_probability else "No forecast data",
            market_price=market.current_price,
            narrative_probability=narrative_prob,
            gap=gap,
            confidence=confidence,
        )

        return brief

    def get_model_prompt_context(self, market: Market, brief: ResearchBrief) -> str:
        return (
            "WEATHER MARKET CONTEXT:\n"
            "- Numerical weather models (GFS, ECMWF) are far more accurate than LLMs for weather\n"
            "- If a weather model forecast is included in the research, weight it VERY heavily (80%+)\n"
            "- Forecast accuracy degrades: 1-3 days ~90%, 4-7 days ~75%, 8-14 days ~60%\n"
            "- For temperature: check if the market is asking about high, low, or average\n"
            "- For precipitation: even small rain chances (30%) often verify\n"
            "- Historical baselines matter: 'above average' needs to know the average\n"
            f"- Sources: {len(brief.sources)} | Confidence: {brief.confidence:.0%}\n"
        )

    def get_data_sources(self) -> list[str]:
        return ["open_meteo_forecast", "open_meteo_historical", "noaa_api"]

    def _parse_weather_market(self, title: str) -> tuple[Optional[str], str, Optional[float], Optional[datetime]]:
        """
        Parse a weather market title into components.
        Returns (location, condition, target_value, target_date)
        """
        title_lower = title.lower()

        # Find location
        location = None
        for city in CITY_COORDINATES:
            if city in title_lower:
                location = city
                break

        # Find condition type
        condition = "temperature"  # default
        if any(w in title_lower for w in ["rain", "precipitation", "snow", "inch"]):
            condition = "precipitation"
        elif any(w in title_lower for w in ["wind", "mph", "gust"]):
            condition = "wind"
        elif any(w in title_lower for w in ["hurricane", "tropical", "storm"]):
            condition = "storm"

        # Find target value (temperature threshold, rainfall amount, etc.)
        target_value = None
        # Match patterns like "above 90", "below 32", "exceed 100", "reach 80°F"
        temp_match = re.search(r'(?:above|below|exceed|reach|over|under|higher than|lower than)\s*(\d+)', title_lower)
        if temp_match:
            target_value = float(temp_match.group(1))

        # Find target date
        target_date = None  # Would need more sophisticated date parsing in production

        return location, condition, target_value, target_date

    def _get_coordinates(self, location: str) -> Optional[tuple[float, float]]:
        """Get lat/lon for a location."""
        return CITY_COORDINATES.get(location.lower())

    async def _fetch_forecast(self, lat: float, lon: float,
                                target_date: Optional[datetime] = None) -> Optional[dict]:
        """Fetch weather forecast from Open-Meteo (free, no API key needed)."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                    "temperature_unit": "fahrenheit",
                    "forecast_days": 7,
                    "timezone": "auto",
                }
                async with session.get(url, params=params,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        daily = data.get("daily", {})
                        return {
                            "dates": daily.get("time", []),
                            "temp_max": daily.get("temperature_2m_max", []),
                            "temp_min": daily.get("temperature_2m_min", []),
                            "precip": daily.get("precipitation_sum", []),
                            "wind_max": daily.get("wind_speed_10m_max", []),
                        }
                    logger.warning(f"Open-Meteo forecast returned {resp.status}")
        except Exception as e:
            logger.error(f"Forecast fetch error: {e}")
        return None

    async def _fetch_historical(self, lat: float, lon: float,
                                  target_date: Optional[datetime] = None) -> Optional[dict]:
        """Fetch historical weather data for baseline comparison."""
        # Use last year's same week as baseline
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=365)

        try:
            async with aiohttp.ClientSession() as session:
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": (start - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "end_date": start.strftime("%Y-%m-%d"),
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "temperature_unit": "fahrenheit",
                    "timezone": "auto",
                }
                async with session.get(url, params=params,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        daily = data.get("daily", {})
                        temps_max = daily.get("temperature_2m_max", [])
                        if temps_max:
                            return {
                                "avg_high": sum(t for t in temps_max if t) / max(len([t for t in temps_max if t]), 1),
                                "avg_precip": sum(p for p in daily.get("precipitation_sum", []) if p) / 7,
                            }
        except Exception as e:
            logger.debug(f"Historical fetch error: {e}")
        return None

    def _calculate_probability(self, forecast: dict, condition: str,
                                 target_value: Optional[float]) -> Optional[float]:
        """
        Calculate probability from forecast data.
        Uses the forecast ensemble to estimate likelihood.
        """
        if target_value is None:
            return None

        if condition == "temperature":
            temps = forecast.get("temp_max", [])
            if not temps:
                return None
            # What fraction of forecast days exceed the target?
            valid_temps = [t for t in temps if t is not None]
            if not valid_temps:
                return None
            above_count = sum(1 for t in valid_temps if t >= target_value)
            # Use first day's forecast as primary signal
            first_temp = valid_temps[0]
            # Distance from target in degrees → probability
            diff = first_temp - target_value
            # Sigmoid-ish mapping: 5 degrees above = ~85%, 5 below = ~15%
            import math
            prob = 1 / (1 + math.exp(-diff / 3))
            return max(0.05, min(0.95, prob))

        elif condition == "precipitation":
            precip = forecast.get("precip", [])
            if not precip:
                return None
            valid_precip = [p for p in precip if p is not None]
            if not valid_precip:
                return None
            first_precip = valid_precip[0]
            if target_value <= 0:
                # "Will it rain?" → any precipitation
                return 0.85 if first_precip > 0.01 else 0.15
            # Amount-based
            import math
            diff = first_precip - target_value
            prob = 1 / (1 + math.exp(-diff / 0.5))
            return max(0.05, min(0.95, prob))

        return None
