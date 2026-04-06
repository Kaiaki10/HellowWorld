"""
Step 1: Scanner - Find Markets Worth Trading

Scans Polymarket (CLOB API) and Kalshi (REST API) for tradeable markets.
Filters by volume, liquidity, time-to-expiry, and flags anomalies.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp
import numpy as np

from config import get_setting, get_env

logger = logging.getLogger(__name__)


@dataclass
class Market:
    """Represents a tradeable prediction market."""
    platform: str               # "polymarket" or "kalshi"
    market_id: str
    title: str
    url: str
    current_price: float        # 0-1 probability (e.g., 0.49 = 49 cents)
    volume_24h: float
    total_volume: float
    liquidity_usd: float
    spread: float               # bid-ask spread in cents
    expiry: Optional[datetime] = None
    days_to_expiry: float = 0.0
    price_change_24h: float = 0.0
    volume_7d_avg: float = 0.0
    anomaly_flags: list = field(default_factory=list)
    opportunity_score: float = 0.0
    raw_data: dict = field(default_factory=dict)


class PolymarketClient:
    """Client for Polymarket CLOB API."""

    BASE_URL = get_setting("platforms", "polymarket", "base_url") or "https://clob.polymarket.com"

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.api_key = get_env("POLYMARKET_API_KEY")

    async def get_markets(self) -> list[dict]:
        """Fetch active markets from Polymarket."""
        markets = []
        next_cursor = "MA=="
        try:
            while next_cursor:
                url = f"{self.BASE_URL}/markets"
                params = {"next_cursor": next_cursor, "active": "true"}
                async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.error(f"Polymarket API returned {resp.status}")
                        break
                    data = await resp.json()
                    batch = data.get("data", data) if isinstance(data, dict) else data
                    if isinstance(batch, list):
                        markets.extend(batch)
                    next_cursor = data.get("next_cursor") if isinstance(data, dict) else None
                    if next_cursor == "LTE=" or len(markets) > 500:
                        break
        except Exception as e:
            logger.error(f"Error fetching Polymarket markets: {e}")
        return markets

    async def get_orderbook(self, token_id: str) -> dict:
        """Fetch orderbook for a specific market token."""
        try:
            url = f"{self.BASE_URL}/book"
            params = {"token_id": token_id}
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Error fetching orderbook for {token_id}: {e}")
        return {}

    def parse_market(self, raw: dict) -> Optional[Market]:
        """Parse raw Polymarket API data into a Market object."""
        try:
            tokens = raw.get("tokens", [])
            if not tokens:
                return None

            # Use the first token (YES outcome) for pricing
            token = tokens[0]
            price = float(token.get("price", 0))
            volume = float(raw.get("volume", 0) or 0)
            liquidity = float(raw.get("liquidity", 0) or 0)

            end_date = raw.get("end_date_iso") or raw.get("end_date")
            expiry = None
            days_to_expiry = 999
            if end_date:
                if isinstance(end_date, str):
                    expiry = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    days_to_expiry = (expiry - datetime.now(timezone.utc)).days

            return Market(
                platform="polymarket",
                market_id=raw.get("condition_id", ""),
                title=raw.get("question", raw.get("title", "Unknown")),
                url=f"https://polymarket.com/event/{raw.get('slug', '')}",
                current_price=price,
                volume_24h=float(raw.get("volume_24h", 0) or 0),
                total_volume=volume,
                liquidity_usd=liquidity,
                spread=0.0,  # Will be updated from orderbook
                expiry=expiry,
                days_to_expiry=days_to_expiry,
                raw_data=raw,
            )
        except Exception as e:
            logger.debug(f"Error parsing Polymarket market: {e}")
            return None


class KalshiClient:
    """Client for Kalshi REST API (demo environment)."""

    BASE_URL = get_setting("platforms", "kalshi", "base_url") or "https://demo-api.kalshi.co/trade-api/v2"

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.api_key = get_env("KALSHI_API_KEY")
        self.token = None

    async def authenticate(self):
        """Authenticate with Kalshi API."""
        api_key = get_env("KALSHI_API_KEY")
        api_secret = get_env("KALSHI_API_SECRET")
        if not api_key or not api_secret:
            logger.warning("Kalshi API credentials not set; skipping authentication")
            return

        try:
            url = f"{self.BASE_URL}/login"
            payload = {"email": api_key, "password": api_secret}
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.token = data.get("token")
                    logger.info("Kalshi authentication successful")
                else:
                    logger.error(f"Kalshi auth failed: {resp.status}")
        except Exception as e:
            logger.error(f"Kalshi authentication error: {e}")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get_markets(self) -> list[dict]:
        """Fetch active markets from Kalshi."""
        markets = []
        try:
            url = f"{self.BASE_URL}/markets"
            params = {"status": "open", "limit": 200}
            async with self.session.get(url, params=params, headers=self._headers(),
                                         timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    markets = data.get("markets", [])
                else:
                    logger.error(f"Kalshi API returned {resp.status}")
        except Exception as e:
            logger.error(f"Error fetching Kalshi markets: {e}")
        return markets

    def parse_market(self, raw: dict) -> Optional[Market]:
        """Parse raw Kalshi API data into a Market object."""
        try:
            yes_price = float(raw.get("yes_ask", 0) or 0) / 100.0
            no_price = float(raw.get("no_ask", 0) or 0) / 100.0
            spread = abs(yes_price + no_price - 1.0)

            volume = float(raw.get("volume", 0) or 0)

            close_time = raw.get("close_time") or raw.get("expiration_time")
            expiry = None
            days_to_expiry = 999
            if close_time:
                if isinstance(close_time, str):
                    expiry = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                    days_to_expiry = (expiry - datetime.now(timezone.utc)).days

            return Market(
                platform="kalshi",
                market_id=raw.get("ticker", ""),
                title=raw.get("title", "Unknown"),
                url=f"https://kalshi.com/markets/{raw.get('ticker', '')}",
                current_price=yes_price,
                volume_24h=float(raw.get("volume_24h", 0) or 0),
                total_volume=volume,
                liquidity_usd=float(raw.get("open_interest", 0) or 0),
                spread=spread,
                expiry=expiry,
                days_to_expiry=days_to_expiry,
                raw_data=raw,
            )
        except Exception as e:
            logger.debug(f"Error parsing Kalshi market: {e}")
            return None


class MarketScanner:
    """
    Main scanner that aggregates markets from all platforms,
    applies filters, detects anomalies, and ranks opportunities.
    """

    def __init__(self):
        self.filters = get_setting("scanner", "filters") or {}
        self.anomaly_thresholds = get_setting("scanner", "anomaly_thresholds") or {}
        self.min_volume = self.filters.get("min_volume", 200)
        self.max_days = self.filters.get("max_days_to_expiry", 30)
        self.min_liquidity = self.filters.get("min_liquidity_usd", 500)

    async def scan(self) -> list[Market]:
        """Run a full scan across all platforms."""
        logger.info("Starting market scan...")
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            poly_client = PolymarketClient(session)
            kalshi_client = KalshiClient(session)

            # Authenticate Kalshi if credentials available
            await kalshi_client.authenticate()

            # Fetch markets from both platforms in parallel
            poly_raw, kalshi_raw = await asyncio.gather(
                poly_client.get_markets(),
                kalshi_client.get_markets(),
            )

            logger.info(f"Fetched {len(poly_raw)} Polymarket, {len(kalshi_raw)} Kalshi markets")

            # Parse all markets
            markets: list[Market] = []
            for raw in poly_raw:
                m = poly_client.parse_market(raw)
                if m:
                    markets.append(m)
            for raw in kalshi_raw:
                m = kalshi_client.parse_market(raw)
                if m:
                    markets.append(m)

            # Apply filters
            filtered = self._apply_filters(markets)
            logger.info(f"Filtered to {len(filtered)} markets from {len(markets)} total")

            # Detect anomalies
            for market in filtered:
                self._detect_anomalies(market)

            # Score and rank
            ranked = self._rank_markets(filtered)

            elapsed = time.time() - start_time
            logger.info(f"Scan complete: {len(ranked)} tradeable markets in {elapsed:.1f}s")

            return ranked

    def _apply_filters(self, markets: list[Market]) -> list[Market]:
        """Filter markets by volume, liquidity, and time-to-expiry."""
        filtered = []
        for m in markets:
            if m.total_volume < self.min_volume:
                continue
            if m.days_to_expiry > self.max_days:
                continue
            if m.days_to_expiry < 0:
                continue  # Already expired
            if m.liquidity_usd < self.min_liquidity:
                continue
            if m.current_price <= 0.02 or m.current_price >= 0.98:
                continue  # Too close to resolution, no edge
            filtered.append(m)
        return filtered

    def _detect_anomalies(self, market: Market):
        """Flag markets with unusual activity."""
        thresholds = self.anomaly_thresholds

        # Sudden price move
        if abs(market.price_change_24h) > thresholds.get("price_move_pct", 10):
            market.anomaly_flags.append(f"price_move_{market.price_change_24h:+.1f}%")

        # Wide spread
        spread_limit = thresholds.get("spread_cents", 5) / 100.0
        if market.spread > spread_limit:
            market.anomaly_flags.append(f"wide_spread_{market.spread:.2f}")

        # Volume spike vs 7-day average
        spike_ratio = thresholds.get("volume_spike_ratio", 2.0)
        if market.volume_7d_avg > 0 and market.volume_24h > market.volume_7d_avg * spike_ratio:
            ratio = market.volume_24h / market.volume_7d_avg
            market.anomaly_flags.append(f"volume_spike_{ratio:.1f}x")

    def _rank_markets(self, markets: list[Market]) -> list[Market]:
        """Rank markets by estimated opportunity score."""
        for m in markets:
            # Opportunity score: higher volume + liquidity + anomalies = more interesting
            volume_score = min(m.total_volume / 10000, 1.0)
            liquidity_score = min(m.liquidity_usd / 5000, 1.0)
            time_score = max(0, 1.0 - m.days_to_expiry / 30)
            anomaly_bonus = len(m.anomaly_flags) * 0.15

            # Prefer mid-range prices (more room for edge)
            price_score = 1.0 - abs(m.current_price - 0.5) * 2

            m.opportunity_score = (
                volume_score * 0.25
                + liquidity_score * 0.25
                + time_score * 0.15
                + price_score * 0.20
                + anomaly_bonus * 0.15
            )

        # Sort descending by opportunity score
        markets.sort(key=lambda m: m.opportunity_score, reverse=True)
        return markets

    def format_results(self, markets: list[Market], top_n: int = 20) -> str:
        """Format scan results for display."""
        lines = [f"=== Market Scan Results ({len(markets)} tradeable markets) ===\n"]
        for i, m in enumerate(markets[:top_n], 1):
            flags = ", ".join(m.anomaly_flags) if m.anomaly_flags else "none"
            lines.append(
                f"{i:2d}. [{m.platform.upper():11s}] {m.title[:60]}\n"
                f"    Price: {m.current_price:.2f} | Vol: {m.total_volume:,.0f} | "
                f"Liq: ${m.liquidity_usd:,.0f} | Exp: {m.days_to_expiry:.0f}d | "
                f"Score: {m.opportunity_score:.3f}\n"
                f"    Anomalies: {flags}\n"
            )
        return "\n".join(lines)


async def run_scan() -> list[Market]:
    """Entry point for running a market scan."""
    scanner = MarketScanner()
    markets = await scanner.scan()
    print(scanner.format_results(markets))
    return markets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_scan())
