"""
Real-Time Market Monitor via WebSocket Feeds

Connects to Polymarket and Kalshi WebSocket APIs for live orderbook updates.
Triggers the research/predict/execute pipeline ONLY when anomalies are detected,
instead of blindly polling on a schedule.

Architecture:
    WebSocket Feed → Anomaly Detection → Pipeline Trigger (event-driven)
    Scheduled Scan (every 30-60 min) → Market Discovery (find NEW markets to watch)

This hybrid approach gives sub-5-second reaction times while keeping
LLM costs low (only called when something actually happens).
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import aiohttp
import websockets

from config import get_setting, get_env
from modules.scanner import Market

logger = logging.getLogger(__name__)


@dataclass
class PriceUpdate:
    """A single price update from a WebSocket feed."""
    platform: str
    market_id: str
    timestamp: datetime
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    volume: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass
class AnomalyEvent:
    """Triggered when real-time data shows something worth investigating."""
    event_type: str        # "price_spike", "spread_blow", "volume_surge", "rapid_move"
    platform: str
    market_id: str
    market_title: str
    current_price: float
    previous_price: float
    change_pct: float
    spread: float
    details: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 1      # 1=high, 2=medium, 3=low

    def summary(self) -> str:
        return (
            f"[{self.event_type.upper()}] {self.market_title[:50]} | "
            f"Price: {self.previous_price:.3f}→{self.current_price:.3f} "
            f"({self.change_pct:+.1f}%) | {self.details}"
        )


class PriceTracker:
    """
    Tracks price history for each market to detect anomalies in real-time.
    Keeps a rolling window of recent prices for comparison.
    """

    def __init__(self, window_size: int = 60):
        self.window_size = window_size  # Number of updates to keep
        self.prices: dict[str, deque[PriceUpdate]] = defaultdict(lambda: deque(maxlen=window_size))
        self.market_titles: dict[str, str] = {}

    def register_market(self, market_id: str, title: str):
        self.market_titles[market_id] = title

    def update(self, price_update: PriceUpdate) -> Optional[AnomalyEvent]:
        """
        Record a price update and check for anomalies.
        Returns an AnomalyEvent if something noteworthy is detected.
        """
        key = f"{price_update.platform}:{price_update.market_id}"
        history = self.prices[key]

        anomaly = None

        if len(history) > 0:
            prev = history[-1]

            # Check for price spike (>5% move since last update)
            if prev.mid_price > 0:
                change_pct = ((price_update.mid_price - prev.mid_price) / prev.mid_price) * 100
                if abs(change_pct) >= 5.0:
                    anomaly = AnomalyEvent(
                        event_type="price_spike",
                        platform=price_update.platform,
                        market_id=price_update.market_id,
                        market_title=self.market_titles.get(price_update.market_id, "Unknown"),
                        current_price=price_update.mid_price,
                        previous_price=prev.mid_price,
                        change_pct=change_pct,
                        spread=price_update.spread,
                        details=f"Moved {change_pct:+.1f}% in one update",
                        priority=1 if abs(change_pct) >= 10 else 2,
                    )

            # Check for spread blowout (>5 cents)
            if price_update.spread > 0.05 and prev.spread <= 0.05:
                anomaly = anomaly or AnomalyEvent(
                    event_type="spread_blow",
                    platform=price_update.platform,
                    market_id=price_update.market_id,
                    market_title=self.market_titles.get(price_update.market_id, "Unknown"),
                    current_price=price_update.mid_price,
                    previous_price=prev.mid_price,
                    change_pct=0,
                    spread=price_update.spread,
                    details=f"Spread widened to {price_update.spread:.3f} (was {prev.spread:.3f})",
                    priority=2,
                )

            # Check for rapid movement (>10% over rolling window)
            if len(history) >= 10:
                window_start = history[0]
                if window_start.mid_price > 0:
                    window_change = ((price_update.mid_price - window_start.mid_price)
                                     / window_start.mid_price) * 100
                    if abs(window_change) >= 10.0:
                        anomaly = anomaly or AnomalyEvent(
                            event_type="rapid_move",
                            platform=price_update.platform,
                            market_id=price_update.market_id,
                            market_title=self.market_titles.get(price_update.market_id, "Unknown"),
                            current_price=price_update.mid_price,
                            previous_price=window_start.mid_price,
                            change_pct=window_change,
                            spread=price_update.spread,
                            details=f"Moved {window_change:+.1f}% over last {len(history)} updates",
                            priority=1,
                        )

        # Check for volume surge
        if len(history) >= 5 and price_update.volume > 0:
            recent_volumes = [u.volume for u in list(history)[-5:] if u.volume > 0]
            if recent_volumes:
                avg_vol = sum(recent_volumes) / len(recent_volumes)
                if avg_vol > 0 and price_update.volume > avg_vol * 3:
                    anomaly = anomaly or AnomalyEvent(
                        event_type="volume_surge",
                        platform=price_update.platform,
                        market_id=price_update.market_id,
                        market_title=self.market_titles.get(price_update.market_id, "Unknown"),
                        current_price=price_update.mid_price,
                        previous_price=history[-1].mid_price if history else 0,
                        change_pct=0,
                        spread=price_update.spread,
                        details=f"Volume {price_update.volume:.0f} is {price_update.volume/avg_vol:.1f}x average",
                        priority=2,
                    )

        history.append(price_update)
        return anomaly


class PolymarketWebSocket:
    """
    Connects to Polymarket's WebSocket API for live orderbook updates.
    Docs: https://docs.polymarket.com/#websocket-api
    """

    WS_URL = get_setting("platforms", "polymarket", "ws_url") or \
        "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, on_update: Callable[[PriceUpdate], None]):
        self.on_update = on_update
        self.subscribed_tokens: list[str] = []
        self._ws = None
        self._running = False

    async def connect(self, token_ids: list[str]):
        """Connect to WebSocket and subscribe to market feeds."""
        self.subscribed_tokens = token_ids
        self._running = True

        while self._running:
            try:
                logger.info(f"Connecting to Polymarket WebSocket ({len(token_ids)} markets)...")
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws

                    # Subscribe to each market's orderbook
                    for token_id in token_ids:
                        subscribe_msg = json.dumps({
                            "type": "subscribe",
                            "channel": "book",
                            "assets_ids": [token_id],
                        })
                        await ws.send(subscribe_msg)

                    logger.info("Polymarket WebSocket connected and subscribed")

                    # Listen for updates
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            update = self._parse_message(data)
                            if update:
                                self.on_update(update)
                        except json.JSONDecodeError:
                            continue

            except websockets.ConnectionClosed as e:
                logger.warning(f"Polymarket WebSocket disconnected: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Polymarket WebSocket error: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10)

    def _parse_message(self, data: dict) -> Optional[PriceUpdate]:
        """Parse a Polymarket WebSocket message into a PriceUpdate."""
        msg_type = data.get("type") or data.get("event_type", "")

        if msg_type in ("book", "price_change", "book_update"):
            market_id = data.get("asset_id") or data.get("market", "")
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            best_bid = float(bids[0].get("price", 0)) if bids else 0
            best_ask = float(asks[0].get("price", 0)) if asks else 0

            if best_bid <= 0 and best_ask <= 0:
                return None

            mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else max(best_bid, best_ask)
            spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0

            return PriceUpdate(
                platform="polymarket",
                market_id=market_id,
                timestamp=datetime.now(timezone.utc),
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=mid,
                spread=spread,
                volume=float(data.get("volume", 0) or 0),
                raw=data,
            )
        return None

    async def disconnect(self):
        self._running = False
        if self._ws:
            await self._ws.close()


class KalshiWebSocket:
    """
    Connects to Kalshi's WebSocket API for live updates.
    Kalshi uses a different protocol — falls back to REST polling
    if WebSocket is not available in the environment.
    """

    WS_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"

    def __init__(self, on_update: Callable[[PriceUpdate], None]):
        self.on_update = on_update
        self._running = False

    async def connect(self, tickers: list[str]):
        """Connect to Kalshi WebSocket or fall back to polling."""
        self._running = True
        api_key = get_env("KALSHI_API_KEY")

        if not api_key:
            logger.info("Kalshi API key not set, using REST polling fallback")
            await self._poll_fallback(tickers)
            return

        while self._running:
            try:
                logger.info(f"Connecting to Kalshi WebSocket ({len(tickers)} markets)...")
                async with websockets.connect(
                    self.WS_URL,
                    extra_headers={"Authorization": f"Bearer {api_key}"},
                    ping_interval=30,
                ) as ws:
                    # Subscribe to orderbook channels
                    for ticker in tickers:
                        msg = json.dumps({
                            "type": "subscribe",
                            "channels": ["orderbook"],
                            "market_tickers": [ticker],
                        })
                        await ws.send(msg)

                    logger.info("Kalshi WebSocket connected")

                    async for message in ws:
                        try:
                            data = json.loads(message)
                            update = self._parse_message(data)
                            if update:
                                self.on_update(update)
                        except json.JSONDecodeError:
                            continue

            except websockets.ConnectionClosed:
                logger.warning("Kalshi WebSocket disconnected. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.warning(f"Kalshi WebSocket failed: {e}. Falling back to polling...")
                await self._poll_fallback(tickers)
                break

    async def _poll_fallback(self, tickers: list[str]):
        """REST polling fallback if WebSocket is unavailable."""
        base_url = get_setting("platforms", "kalshi", "base_url") or "https://demo-api.kalshi.co/trade-api/v2"

        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    for ticker in tickers:
                        url = f"{base_url}/markets/{ticker}"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                market = data.get("market", data)
                                yes_ask = float(market.get("yes_ask", 0) or 0) / 100
                                no_ask = float(market.get("no_ask", 0) or 0) / 100
                                mid = yes_ask if yes_ask > 0 else 0.5
                                spread = abs(yes_ask + no_ask - 1.0) if yes_ask > 0 and no_ask > 0 else 0

                                update = PriceUpdate(
                                    platform="kalshi",
                                    market_id=ticker,
                                    timestamp=datetime.now(timezone.utc),
                                    best_bid=mid - spread / 2,
                                    best_ask=mid + spread / 2,
                                    mid_price=mid,
                                    spread=spread,
                                    volume=float(market.get("volume", 0) or 0),
                                )
                                self.on_update(update)
            except Exception as e:
                logger.error(f"Kalshi polling error: {e}")

            await asyncio.sleep(30)  # Poll every 30 seconds as fallback

    def _parse_message(self, data: dict) -> Optional[PriceUpdate]:
        """Parse Kalshi WebSocket message."""
        msg_type = data.get("type", "")
        if msg_type in ("orderbook_snapshot", "orderbook_update"):
            ticker = data.get("market_ticker", "")
            yes_bids = data.get("yes", {}).get("bids", [])
            yes_asks = data.get("yes", {}).get("asks", [])

            best_bid = float(yes_bids[0][0]) / 100 if yes_bids else 0
            best_ask = float(yes_asks[0][0]) / 100 if yes_asks else 0

            mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else max(best_bid, best_ask)
            spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0

            return PriceUpdate(
                platform="kalshi",
                market_id=ticker,
                timestamp=datetime.now(timezone.utc),
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=mid,
                spread=spread,
                raw=data,
            )
        return None

    async def disconnect(self):
        self._running = False


class RealtimeMonitor:
    """
    Main real-time monitoring orchestrator.

    Hybrid architecture:
    1. Scheduled scan (every 30-60 min) discovers NEW markets to watch
    2. WebSocket feeds monitor known markets in real-time
    3. Anomaly detection triggers the expensive pipeline ONLY when needed

    This gives sub-5-second reaction times at a fraction of the polling cost.
    """

    def __init__(self, on_anomaly: Callable[[AnomalyEvent, Market], None]):
        """
        Args:
            on_anomaly: Callback fired when an anomaly is detected.
                        Receives the event and the associated Market object.
                        This should trigger the research→predict→execute pipeline.
        """
        self.on_anomaly = on_anomaly
        self.tracker = PriceTracker(window_size=120)
        self.watched_markets: dict[str, Market] = {}  # market_id -> Market
        self.poly_ws = PolymarketWebSocket(on_update=self._handle_update)
        self.kalshi_ws = KalshiWebSocket(on_update=self._handle_update)
        self.anomaly_cooldown: dict[str, float] = {}  # Prevent duplicate triggers
        self.cooldown_seconds = 120  # Min 2 min between triggers for same market
        self._stats = {"updates": 0, "anomalies": 0, "triggers": 0}

    def add_markets(self, markets: list[Market]):
        """Register markets to watch via WebSocket."""
        for m in markets:
            self.watched_markets[m.market_id] = m
            self.tracker.register_market(m.market_id, m.title)
        logger.info(f"Watching {len(self.watched_markets)} markets in real-time")

    async def start(self):
        """Start all WebSocket connections."""
        poly_ids = [m.market_id for m in self.watched_markets.values() if m.platform == "polymarket"]
        kalshi_ids = [m.market_id for m in self.watched_markets.values() if m.platform == "kalshi"]

        tasks = []
        if poly_ids:
            tasks.append(asyncio.create_task(self.poly_ws.connect(poly_ids)))
        if kalshi_ids:
            tasks.append(asyncio.create_task(self.kalshi_ws.connect(kalshi_ids)))

        if tasks:
            logger.info(f"Real-time monitor started: {len(poly_ids)} Polymarket, {len(kalshi_ids)} Kalshi")
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.warning("No markets to watch")

    async def stop(self):
        """Disconnect all WebSocket connections."""
        await self.poly_ws.disconnect()
        await self.kalshi_ws.disconnect()
        logger.info(f"Real-time monitor stopped. Stats: {self._stats}")

    def _handle_update(self, update: PriceUpdate):
        """Process a price update from any WebSocket feed."""
        self._stats["updates"] += 1

        # Run through anomaly detection
        anomaly = self.tracker.update(update)

        if anomaly:
            self._stats["anomalies"] += 1

            # Check cooldown to prevent rapid re-triggering
            key = f"{anomaly.platform}:{anomaly.market_id}"
            now = time.time()
            last_trigger = self.anomaly_cooldown.get(key, 0)

            if now - last_trigger < self.cooldown_seconds:
                logger.debug(f"Anomaly cooldown active for {key}, skipping")
                return

            self.anomaly_cooldown[key] = now
            self._stats["triggers"] += 1

            logger.warning(f"ANOMALY DETECTED: {anomaly.summary()}")

            # Get the Market object for this anomaly
            market = self.watched_markets.get(anomaly.market_id)
            if market:
                # Update market's current price
                market.current_price = anomaly.current_price
                # Fire the callback to trigger the pipeline
                self.on_anomaly(anomaly, market)

    def stats(self) -> str:
        return (
            f"Real-Time Monitor Stats:\n"
            f"  Updates received: {self._stats['updates']}\n"
            f"  Anomalies detected: {self._stats['anomalies']}\n"
            f"  Pipeline triggers: {self._stats['triggers']}\n"
            f"  Markets watched: {len(self.watched_markets)}\n"
            f"  Cooldown: {self.cooldown_seconds}s between triggers"
        )
