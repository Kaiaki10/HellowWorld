"""
Economics Specialist

Tuned for macroeconomic prediction markets: Fed rates, CPI, jobs reports, GDP.

Key data sources:
  - FRED API (Federal Reserve Economic Data — free, authoritative)
  - BLS API (Bureau of Labor Statistics — free, jobs/CPI data)
  - CME FedWatch (implied rate probabilities from futures)
  - Treasury yield curves (recession signal)
  - Economic calendar RSS feeds

Why specialized: Economic markets resolve on hard government data releases.
Leading indicators (initial claims → jobs report, Cleveland Fed nowcast → CPI)
give a quantitative edge that generic sentiment analysis completely misses.
The data is free, authoritative, and most prediction market traders don't
build models on it — they trade on vibes and headlines.

Edge examples:
  - Cleveland Fed CPI Nowcast predicts CPI within 0.1% days before release
  - Initial jobless claims trend predicts NFP direction ~70% of the time
  - CME FedWatch implied probabilities are the gold standard for rate decisions
"""

import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

from modules.scanner import Market
from modules.researcher import (
    ResearchBrief, SourceResult, SentimentAnalyzer,
    RSSFetcher, MarketResearcher,
)
from modules.specialists import BaseSpecialist

logger = logging.getLogger(__name__)


# Key economic indicators and their release patterns
ECONOMIC_INDICATORS = {
    "cpi": {
        "name": "Consumer Price Index",
        "fred_series": ["CPIAUCSL", "CPILFESL"],  # All items, Core
        "keywords": ["cpi", "inflation", "consumer price"],
        "leading_indicators": ["CPIAUCSL", "T10YIE"],  # Cleveland Fed nowcast, breakeven inflation
    },
    "jobs": {
        "name": "Nonfarm Payrolls / Employment",
        "fred_series": ["PAYEMS", "UNRATE", "ICSA"],
        "keywords": ["jobs", "employment", "payroll", "unemployment", "nonfarm"],
        "leading_indicators": ["ICSA", "CCSA"],  # Initial claims, continuing claims
    },
    "gdp": {
        "name": "Gross Domestic Product",
        "fred_series": ["GDP", "GDPC1"],
        "keywords": ["gdp", "growth", "recession"],
        "leading_indicators": ["GDPNOW"],  # Atlanta Fed GDPNow
    },
    "fed_rate": {
        "name": "Federal Funds Rate",
        "fred_series": ["FEDFUNDS", "DFEDTARU"],
        "keywords": ["fed", "federal reserve", "interest rate", "fomc", "rate cut", "rate hike"],
        "leading_indicators": ["T10Y2Y", "DFF"],  # Yield curve, effective FFR
    },
    "housing": {
        "name": "Housing Market",
        "fred_series": ["HOUST", "MSPUS", "MORTGAGE30US"],
        "keywords": ["housing", "home", "mortgage", "real estate"],
        "leading_indicators": ["PERMIT", "MORTGAGE30US"],  # Building permits, mortgage rates
    },
    "retail": {
        "name": "Retail Sales",
        "fred_series": ["RSAFS"],
        "keywords": ["retail", "consumer spending", "sales"],
        "leading_indicators": ["UMCSENT"],  # Consumer sentiment
    },
}


class EconomicsSpecialist(BaseSpecialist):

    category = "economics"
    description = "Fed rates, CPI/inflation, jobs reports, GDP, and macro events"

    ECONOMICS_RSS_FEEDS = [
        "https://news.google.com/rss/search?q={query}+economy+federal+reserve&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/economics/search/.rss?q={query}&sort=new&t=week",
    ]

    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.base_researcher = MarketResearcher()

    async def research(self, market: Market) -> ResearchBrief:
        """
        Economics research pipeline:
        1. Detect which economic indicator the market is about
        2. Fetch current + historical data from FRED
        3. Fetch leading indicators that predict the target
        4. Fetch economic news for context
        5. Calculate probability from data trends
        """
        # Run base research for general sentiment
        brief = await self.base_researcher.research_market(market)

        indicator = self._detect_indicator(market.title)
        query = self._extract_economics_query(market.title)
        target_value = self._extract_target_value(market.title)

        # Fetch economic data in parallel
        fred_task = self._fetch_fred_data(indicator)
        leading_task = self._fetch_leading_indicators(indicator)
        treasury_task = self._fetch_treasury_yields()
        news_task = self._fetch_economics_news(query)

        results = await asyncio.gather(
            fred_task, leading_task, treasury_task, news_task,
            return_exceptions=True,
        )

        fred_data = results[0] if isinstance(results[0], dict) else None
        leading_data = results[1] if isinstance(results[1], dict) else None
        treasury_data = results[2] if isinstance(results[2], dict) else None
        news_sources = results[3] if isinstance(results[3], list) else []

        # Add FRED data
        if fred_data:
            for series_id, series_data in fred_data.items():
                if series_data:
                    latest = series_data[-1] if series_data else {}
                    trend = self._calculate_trend(series_data)
                    brief.sources.append(SourceResult(
                        source_type="fred",
                        title=f"FRED {series_id}: {latest.get('value', 'N/A')}",
                        content=(
                            f"Latest: {latest.get('value', 'N/A')} ({latest.get('date', 'N/A')}). "
                            f"Trend: {trend}. "
                            f"Previous values: {[d.get('value') for d in series_data[-5:]]}"
                        ),
                        url=f"https://fred.stlouisfed.org/series/{series_id}",
                        sentiment_score=0,
                        sentiment_label="neutral",
                    ))

        # Add leading indicators
        if leading_data:
            for series_id, series_data in leading_data.items():
                if series_data:
                    latest = series_data[-1] if series_data else {}
                    trend = self._calculate_trend(series_data)
                    brief.sources.append(SourceResult(
                        source_type="fred_leading",
                        title=f"Leading indicator {series_id}: {latest.get('value', 'N/A')}",
                        content=(
                            f"Latest: {latest.get('value', 'N/A')} ({latest.get('date', 'N/A')}). "
                            f"Trend: {trend}. "
                            f"This is a leading indicator for {indicator.get('name', 'economic data')}."
                        ),
                        url=f"https://fred.stlouisfed.org/series/{series_id}",
                        sentiment_score=0,
                        sentiment_label="neutral",
                    ))

        # Add treasury yield curve (recession signal)
        if treasury_data:
            spread = treasury_data.get("10y2y_spread")
            brief.sources.append(SourceResult(
                source_type="treasury",
                title=f"Treasury yield curve: 10Y-2Y spread = {spread}",
                content=(
                    f"10Y yield: {treasury_data.get('10y', 'N/A')}%, "
                    f"2Y yield: {treasury_data.get('2y', 'N/A')}%, "
                    f"Spread: {spread}bp. "
                    f"{'INVERTED (recession signal)' if spread and spread < 0 else 'Normal'}"
                ),
                url="https://fred.stlouisfed.org/series/T10Y2Y",
                sentiment_score=-0.3 if spread and spread < 0 else 0.1,
                sentiment_label="bearish" if spread and spread < 0 else "neutral",
            ))

        # Add news
        for source in news_sources:
            score, label = self.sentiment.analyze(f"{source.title} {source.content}")
            source.sentiment_score = score
            source.sentiment_label = label
        brief.sources.extend(news_sources)

        # Calculate data-driven probability
        data_probability = self._calculate_data_probability(
            fred_data, leading_data, treasury_data, target_value, market.title, indicator,
        )
        if data_probability is not None:
            # Weight hard data heavily — it's more predictive than sentiment
            brief.narrative_probability = (
                brief.narrative_probability * 0.30 + data_probability * 0.70
            )
            brief.gap = brief.narrative_probability - brief.market_price
            brief.confidence = min(brief.confidence + 0.20, 0.90)

        return brief

    def get_model_prompt_context(self, market: Market, brief: ResearchBrief) -> str:
        return (
            "ECONOMICS/MACRO MARKET CONTEXT:\n"
            "- Government economic data (CPI, jobs, GDP) is the definitive resolution source\n"
            "- Leading indicators predict releases: initial claims → jobs, nowcasts → CPI/GDP\n"
            "- CME FedWatch probabilities are the gold standard for Fed rate predictions\n"
            "- Inverted yield curve (10Y-2Y < 0) is a recession signal with 12-18 month lead\n"
            "- CPI: Cleveland Fed Nowcast predicts within 0.1% days before release\n"
            "- Jobs: initial claims trend predicts NFP direction ~70% of the time\n"
            "- GDP: Atlanta Fed GDPNow updates in real-time with incoming data\n"
            "- Markets often overreact to single data points — look at the trend\n"
            "- Fed communications (dot plot, minutes, speeches) signal future policy\n"
            f"- Sources: {len(brief.sources)} | Confidence: {brief.confidence:.0%}\n"
        )

    def get_data_sources(self) -> list[str]:
        return ["fred_api", "treasury_yields", "economics_rss", "bls_data"]

    def _detect_indicator(self, title: str) -> dict:
        """Detect which economic indicator a market is about."""
        title_lower = title.lower()
        best_indicator = {}
        best_count = 0

        for key, config in ECONOMIC_INDICATORS.items():
            count = sum(1 for kw in config["keywords"] if kw in title_lower)
            if count > best_count:
                best_count = count
                best_indicator = config

        return best_indicator

    def _extract_economics_query(self, title: str) -> str:
        """Build search query for economic content."""
        terms = re.findall(
            r'\b(?:Fed|CPI|GDP|inflation|unemployment|recession|'
            r'interest rate|FOMC|treasury|tariff|jobs|payroll|'
            r'housing|retail|[A-Z][a-z]+)\b',
            title,
        )
        if terms:
            return " ".join(terms[:5])
        return title.split("?")[0].strip()[:50]

    def _extract_target_value(self, title: str) -> Optional[float]:
        """Extract numeric target from market title."""
        # Match patterns like "above 3%", "below 4.5%", "exceed 200k", "reach 3.5"
        patterns = [
            r'(?:above|below|exceed|reach|over|under|higher than|lower than)\s*(\d+(?:\.\d+)?)\s*%',
            r'(?:above|below|exceed|reach|over|under)\s*(\d+(?:\.\d+)?)\s*[kK]',
            r'(?:above|below|exceed|reach|over|under)\s*(\d+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if "k" in title[match.start():match.end()].lower():
                        value *= 1000
                    return value
                except ValueError:
                    continue
        return None

    async def _fetch_fred_data(self, indicator: dict) -> Optional[dict]:
        """
        Fetch economic data from FRED API (free, 120 requests/min).
        FRED is the gold standard for US economic data.
        """
        series_ids = indicator.get("fred_series", [])
        if not series_ids:
            return None

        from config import get_env
        api_key = get_env("FRED_API_KEY", "")
        if not api_key:
            logger.debug("No FRED_API_KEY set, skipping FRED data")
            return None

        result = {}
        try:
            async with aiohttp.ClientSession() as session:
                for series_id in series_ids[:3]:  # Limit to 3 series
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        "series_id": series_id,
                        "api_key": api_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 12,  # Last 12 observations
                    }
                    async with session.get(url, params=params,
                                             timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            observations = data.get("observations", [])
                            # Convert to usable format (most recent first)
                            parsed = []
                            for obs in observations:
                                try:
                                    parsed.append({
                                        "date": obs["date"],
                                        "value": float(obs["value"]) if obs["value"] != "." else None,
                                    })
                                except (ValueError, KeyError):
                                    continue
                            result[series_id] = list(reversed(parsed))  # Chronological order
                        else:
                            logger.debug(f"FRED {series_id} returned {resp.status}")
        except Exception as e:
            logger.debug(f"FRED fetch error: {e}")

        return result if result else None

    async def _fetch_leading_indicators(self, indicator: dict) -> Optional[dict]:
        """Fetch leading indicators that predict the target economic data."""
        series_ids = indicator.get("leading_indicators", [])
        if not series_ids:
            return None

        from config import get_env
        api_key = get_env("FRED_API_KEY", "")
        if not api_key:
            return None

        result = {}
        try:
            async with aiohttp.ClientSession() as session:
                for series_id in series_ids[:2]:
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        "series_id": series_id,
                        "api_key": api_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 8,
                    }
                    async with session.get(url, params=params,
                                             timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            observations = data.get("observations", [])
                            parsed = []
                            for obs in observations:
                                try:
                                    parsed.append({
                                        "date": obs["date"],
                                        "value": float(obs["value"]) if obs["value"] != "." else None,
                                    })
                                except (ValueError, KeyError):
                                    continue
                            result[series_id] = list(reversed(parsed))
        except Exception as e:
            logger.debug(f"Leading indicators fetch error: {e}")

        return result if result else None

    async def _fetch_treasury_yields(self) -> Optional[dict]:
        """Fetch treasury yield curve data from FRED."""
        from config import get_env
        api_key = get_env("FRED_API_KEY", "")
        if not api_key:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                yields = {}
                for series_id, label in [("DGS10", "10y"), ("DGS2", "2y"), ("T10Y2Y", "10y2y_spread")]:
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        "series_id": series_id,
                        "api_key": api_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 1,
                    }
                    async with session.get(url, params=params,
                                             timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            observations = data.get("observations", [])
                            if observations and observations[0]["value"] != ".":
                                yields[label] = float(observations[0]["value"])

                return yields if yields else None
        except Exception as e:
            logger.debug(f"Treasury yields fetch error: {e}")
        return None

    async def _fetch_economics_news(self, query: str) -> list[SourceResult]:
        """Fetch economic news from RSS feeds."""
        sources = []
        rss = RSSFetcher()
        rss.DEFAULT_FEEDS = self.ECONOMICS_RSS_FEEDS
        results = await rss.fetch(query, max_results=8)
        for entry in results:
            sources.append(SourceResult(
                source_type="economics_rss",
                title=entry.get("title", ""),
                content=entry.get("summary", "")[:500],
                url=entry.get("link", ""),
                sentiment_score=0,
                sentiment_label="neutral",
            ))
        return sources

    def _calculate_trend(self, data: list[dict]) -> str:
        """Calculate simple trend direction from time series data."""
        values = [d["value"] for d in data if d.get("value") is not None]
        if len(values) < 3:
            return "insufficient data"

        recent_avg = sum(values[-3:]) / 3
        older_avg = sum(values[:3]) / 3

        if older_avg == 0:
            return "flat"

        change_pct = (recent_avg - older_avg) / abs(older_avg) * 100

        if change_pct > 2:
            return f"rising (+{change_pct:.1f}%)"
        elif change_pct < -2:
            return f"falling ({change_pct:.1f}%)"
        else:
            return f"flat ({change_pct:+.1f}%)"

    def _calculate_data_probability(self, fred_data: Optional[dict],
                                      leading_data: Optional[dict],
                                      treasury_data: Optional[dict],
                                      target_value: Optional[float],
                                      market_title: str,
                                      indicator: dict) -> Optional[float]:
        """
        Calculate probability from economic data and leading indicators.
        Uses trend analysis and distance-from-target to estimate likelihood.
        """
        title_lower = market_title.lower()
        is_above = any(w in title_lower for w in ["above", "over", "exceed", "higher", "increase", "rise"])
        is_below = any(w in title_lower for w in ["below", "under", "lower", "decrease", "fall", "cut"])

        # --- Fed rate decision markets ---
        if any(kw in title_lower for kw in ["fed", "rate cut", "rate hike", "fomc", "interest rate"]):
            return self._fed_rate_probability(fred_data, treasury_data, title_lower)

        # --- Numeric target markets (CPI above X%, unemployment below Y%) ---
        if target_value is not None and fred_data:
            return self._target_probability(fred_data, target_value, is_above, is_below)

        # --- Trend-based markets (recession, growth) ---
        if any(w in title_lower for w in ["recession", "contraction"]):
            return self._recession_probability(fred_data, treasury_data)

        return None

    def _fed_rate_probability(self, fred_data: Optional[dict],
                                treasury_data: Optional[dict],
                                title_lower: str) -> Optional[float]:
        """Estimate probability of Fed rate action."""
        import math

        # Use yield curve as primary signal
        if treasury_data:
            spread = treasury_data.get("10y2y_spread")
            if spread is not None:
                if "cut" in title_lower or "lower" in title_lower:
                    # Inverted/flat curve → higher probability of cuts
                    # spread < 0 = inverted, strong cut signal
                    prob = 1 / (1 + math.exp(spread * 2))
                    return max(0.10, min(0.90, prob))
                elif "hike" in title_lower or "raise" in title_lower:
                    # Steep curve → higher probability of hikes
                    prob = 1 / (1 + math.exp(-spread * 2))
                    return max(0.10, min(0.90, prob))
                elif "hold" in title_lower or "unchanged" in title_lower:
                    # Flat curve → hold is likely
                    prob = 1 / (1 + math.exp(abs(spread) * 3 - 1))
                    return max(0.10, min(0.90, prob))

        return None

    def _target_probability(self, fred_data: dict, target_value: float,
                              is_above: bool, is_below: bool) -> Optional[float]:
        """Estimate probability of hitting a numeric target."""
        import math

        # Get the most recent values from any available series
        for series_id, data in fred_data.items():
            values = [d["value"] for d in data if d.get("value") is not None]
            if len(values) < 3:
                continue

            current = values[-1]
            # Calculate recent volatility (standard deviation of changes)
            changes = [values[i] - values[i-1] for i in range(1, len(values))]
            if not changes:
                continue
            vol = max(abs(sum(changes) / len(changes)), 0.01)

            distance = target_value - current
            z_score = distance / vol

            if is_above:
                # Probability of going above target
                prob = 1 / (1 + math.exp(z_score * 0.8))
            elif is_below:
                # Probability of going below target
                prob = 1 / (1 + math.exp(-z_score * 0.8))
            else:
                # Default: treat as "above"
                prob = 1 / (1 + math.exp(z_score * 0.8))

            return max(0.05, min(0.95, prob))

        return None

    def _recession_probability(self, fred_data: Optional[dict],
                                 treasury_data: Optional[dict]) -> Optional[float]:
        """Estimate recession probability from multiple signals."""
        signals = []

        # Yield curve inversion
        if treasury_data:
            spread = treasury_data.get("10y2y_spread")
            if spread is not None:
                if spread < -0.5:
                    signals.append(0.70)  # Deeply inverted — strong recession signal
                elif spread < 0:
                    signals.append(0.55)  # Inverted — moderate signal
                else:
                    signals.append(0.20)  # Normal — low recession risk

        # Unemployment trend (rising = recession risk)
        if fred_data:
            for series_id in ["UNRATE", "ICSA"]:
                if series_id in fred_data:
                    values = [d["value"] for d in fred_data[series_id] if d.get("value") is not None]
                    if len(values) >= 3:
                        trend = (values[-1] - values[-3]) / max(abs(values[-3]), 0.1)
                        if trend > 0.05:
                            signals.append(0.60)  # Rising unemployment
                        else:
                            signals.append(0.25)

        if signals:
            return sum(signals) / len(signals)

        return None
