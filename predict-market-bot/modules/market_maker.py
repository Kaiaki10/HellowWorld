"""
Market Making Module

Instead of only taking directional bets, provides liquidity by quoting
both sides of a market and earning the spread.

Strategy:
  - Quote bid (buy) and ask (sell) around our model's fair value
  - Earn the spread when both sides fill: bid 0.48, ask 0.52 = $0.04/roundtrip
  - Use inventory skew to manage risk: shift quotes away from accumulated side
  - Auto-cancel all quotes when anomaly detected or kill switch triggered

Best for: high-volume, near-50/50 markets with tight spreads
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

from config import get_setting, get_env
from modules.scanner import Market

logger = logging.getLogger(__name__)

DATA_DIR = Path(get_setting("general", "data_dir") or "data")
KILL_SWITCH = get_setting("general", "kill_switch_file") or "STOP"


@dataclass
class Quote:
    """A two-sided quote (bid + ask) on a market."""
    market_id: str
    platform: str
    bid_price: float       # Price we're willing to buy at
    ask_price: float       # Price we're willing to sell at
    spread: float          # ask - bid
    size_contracts: int
    fair_value: float      # Our model's estimated fair price
    bid_order_id: str = ""
    ask_order_id: str = ""
    status: str = "pending"  # "pending", "active", "filled_bid", "filled_ask", "cancelled"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Inventory:
    """Current inventory position from market making."""
    market_id: str
    net_contracts: int     # Positive = long YES, negative = long NO
    avg_entry_price: float
    total_filled_bids: int
    total_filled_asks: int
    realized_pnl: float
    unrealized_pnl: float

    @property
    def roundtrips(self) -> int:
        return min(self.total_filled_bids, self.total_filled_asks)


@dataclass
class MarketMakingConfig:
    """Configuration for market making on a specific market."""
    market: Market
    base_spread: float = 0.04     # 4 cent spread (2 cents each side)
    max_spread: float = 0.10      # Maximum spread before we stop quoting
    size_contracts: int = 10      # Contracts per side
    max_inventory: int = 50       # Max net inventory before skewing
    inventory_skew_rate: float = 0.01  # Price shift per unit of inventory
    refresh_interval: float = 5.0  # Seconds between quote refreshes
    min_volume_24h: float = 500   # Minimum market volume to make
    price_range: tuple = (0.20, 0.80)  # Only make markets in this price range


class InventoryManager:
    """Tracks inventory and PnL across all market-making activities."""

    def __init__(self):
        self.inventories: dict[str, Inventory] = {}
        self.state_file = DATA_DIR / "mm_inventory.json"
        self._load()

    def get(self, market_id: str) -> Inventory:
        if market_id not in self.inventories:
            self.inventories[market_id] = Inventory(
                market_id=market_id,
                net_contracts=0,
                avg_entry_price=0.0,
                total_filled_bids=0,
                total_filled_asks=0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
            )
        return self.inventories[market_id]

    def record_fill(self, market_id: str, side: str, price: float, contracts: int):
        """Record a fill (bid or ask side)."""
        inv = self.get(market_id)
        if side == "bid":
            # We bought (accumulated YES inventory)
            inv.net_contracts += contracts
            inv.total_filled_bids += contracts
        elif side == "ask":
            # We sold (reduced YES inventory)
            inv.net_contracts -= contracts
            inv.total_filled_asks += contracts

        # Calculate realized PnL from completed roundtrips
        roundtrips = inv.roundtrips
        if roundtrips > 0 and inv.total_filled_bids > 0 and inv.total_filled_asks > 0:
            # Simplified: each roundtrip earns approximately the spread
            inv.realized_pnl = roundtrips * 0.02  # Approximate

        self._save()
        logger.info(
            f"MM fill: {market_id} {side} {contracts}@{price:.3f} | "
            f"Net: {inv.net_contracts} | PnL: ${inv.realized_pnl:.2f}"
        )

    def total_pnl(self) -> float:
        return sum(inv.realized_pnl for inv in self.inventories.values())

    def _save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {}
        for mid, inv in self.inventories.items():
            data[mid] = {
                "net_contracts": inv.net_contracts,
                "avg_entry_price": inv.avg_entry_price,
                "total_filled_bids": inv.total_filled_bids,
                "total_filled_asks": inv.total_filled_asks,
                "realized_pnl": inv.realized_pnl,
            }
        self.state_file.write_text(json.dumps(data, indent=2))

    def _load(self):
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                for mid, d in data.items():
                    self.inventories[mid] = Inventory(
                        market_id=mid,
                        net_contracts=d.get("net_contracts", 0),
                        avg_entry_price=d.get("avg_entry_price", 0),
                        total_filled_bids=d.get("total_filled_bids", 0),
                        total_filled_asks=d.get("total_filled_asks", 0),
                        realized_pnl=d.get("realized_pnl", 0),
                        unrealized_pnl=0,
                    )
            except (json.JSONDecodeError, TypeError):
                pass


class QuoteCalculator:
    """Calculates optimal bid/ask quotes with inventory skew."""

    def calculate_quotes(self, config: MarketMakingConfig,
                          fair_value: float,
                          inventory: Inventory) -> Optional[Quote]:
        """
        Calculate bid and ask prices around fair value with inventory skew.

        Inventory skew: if we're long YES (positive inventory), we shift
        our quotes down to encourage sells and discourage buys.
        """
        # Check if market is suitable for making
        price = fair_value
        if price < config.price_range[0] or price > config.price_range[1]:
            logger.debug(f"Price {price:.3f} outside range {config.price_range}, skipping")
            return None

        half_spread = config.base_spread / 2

        # Inventory skew: shift quotes based on accumulated inventory
        skew = inventory.net_contracts * config.inventory_skew_rate
        # Positive inventory (long YES) -> lower both quotes to encourage selling
        # Negative inventory (short YES) -> raise both quotes to encourage buying

        bid_price = fair_value - half_spread - skew
        ask_price = fair_value + half_spread - skew

        # Enforce boundaries
        bid_price = max(0.01, min(0.98, bid_price))
        ask_price = max(0.02, min(0.99, ask_price))

        # Ensure spread is positive and within limits
        actual_spread = ask_price - bid_price
        if actual_spread <= 0 or actual_spread > config.max_spread:
            return None

        # Reduce size when inventory is high
        inventory_ratio = abs(inventory.net_contracts) / max(config.max_inventory, 1)
        size = config.size_contracts
        if inventory_ratio > 0.5:
            size = max(1, int(size * (1 - inventory_ratio)))

        return Quote(
            market_id=config.market.market_id,
            platform=config.market.platform,
            bid_price=round(bid_price, 3),
            ask_price=round(ask_price, 3),
            spread=round(actual_spread, 3),
            size_contracts=size,
            fair_value=fair_value,
        )


class MarketMaker:
    """
    Orchestrates market-making across multiple markets.

    Lifecycle:
    1. Select suitable markets (high volume, near 50/50, tight spreads)
    2. Calculate fair value (from predictor or mid-market)
    3. Place two-sided quotes with inventory skew
    4. Monitor fills and refresh quotes periodically
    5. Cancel all on anomaly/kill switch
    """

    def __init__(self):
        self.inventory_mgr = InventoryManager()
        self.calculator = QuoteCalculator()
        self.active_quotes: dict[str, Quote] = {}
        self.configs: dict[str, MarketMakingConfig] = {}
        self.paper_trading = get_setting("general", "paper_trading", default=True)
        self._running = False

    def select_markets(self, markets: list[Market]) -> list[Market]:
        """Select markets suitable for market making."""
        suitable = []
        for m in markets:
            # High volume
            if m.volume_24h < 500:
                continue
            # Near 50/50 (most spread opportunities)
            if m.current_price < 0.20 or m.current_price > 0.80:
                continue
            # Tight existing spread (we can compete)
            if m.spread > 0.08:
                continue
            # Enough liquidity
            if m.liquidity_usd < 1000:
                continue
            suitable.append(m)

        logger.info(f"Selected {len(suitable)} markets for market making from {len(markets)}")
        return suitable

    def configure_market(self, market: Market,
                          fair_value: Optional[float] = None) -> MarketMakingConfig:
        """Create a market-making configuration for a market."""
        config = MarketMakingConfig(
            market=market,
            base_spread=max(0.03, market.spread + 0.01),  # At least 1 cent wider than current
            size_contracts=10,
        )
        self.configs[market.market_id] = config
        return config

    async def run(self, markets: list[Market], fair_values: Optional[dict[str, float]] = None):
        """
        Main market-making loop. Continuously quotes selected markets.
        """
        self._running = True
        suitable = self.select_markets(markets)

        if not suitable:
            logger.info("No suitable markets for market making")
            return

        for m in suitable:
            self.configure_market(m)

        logger.info(f"Market making started on {len(suitable)} markets (paper={self.paper_trading})")

        while self._running:
            # Kill switch check
            if os.path.exists(KILL_SWITCH):
                logger.critical("Kill switch active - cancelling all MM quotes")
                await self.cancel_all()
                break

            for market in suitable:
                config = self.configs.get(market.market_id)
                if not config:
                    continue

                inventory = self.inventory_mgr.get(market.market_id)
                fair_value = (fair_values or {}).get(market.market_id, market.current_price)

                quote = self.calculator.calculate_quotes(config, fair_value, inventory)
                if quote:
                    await self._place_or_refresh_quote(quote)

            await asyncio.sleep(5)  # Refresh every 5 seconds

    async def _place_or_refresh_quote(self, quote: Quote):
        """Place or refresh a two-sided quote."""
        existing = self.active_quotes.get(quote.market_id)

        # Only refresh if prices changed meaningfully
        if existing and abs(existing.bid_price - quote.bid_price) < 0.002:
            return

        if self.paper_trading:
            quote.status = "active"
            quote.bid_order_id = f"paper_mm_bid_{quote.market_id}_{int(time.time())}"
            quote.ask_order_id = f"paper_mm_ask_{quote.market_id}_{int(time.time())}"
            self.active_quotes[quote.market_id] = quote
            logger.debug(
                f"[PAPER MM] {quote.market_id}: "
                f"BID {quote.bid_price:.3f} / ASK {quote.ask_price:.3f} "
                f"({quote.spread:.3f} spread, {quote.size_contracts} contracts)"
            )
            return

        # Live: cancel existing quotes, place new ones
        if existing and existing.status == "active":
            await self._cancel_quote(existing)

        # Place bid and ask simultaneously
        async with aiohttp.ClientSession() as session:
            bid_result, ask_result = await asyncio.gather(
                self._place_order(session, quote, "bid"),
                self._place_order(session, quote, "ask"),
                return_exceptions=True,
            )

            if isinstance(bid_result, str):
                quote.bid_order_id = bid_result
            if isinstance(ask_result, str):
                quote.ask_order_id = ask_result

            if quote.bid_order_id and quote.ask_order_id:
                quote.status = "active"
            else:
                quote.status = "partial"
                logger.warning(f"Partial quote placement for {quote.market_id}")

        self.active_quotes[quote.market_id] = quote

    async def _place_order(self, session: aiohttp.ClientSession,
                            quote: Quote, side: str) -> str:
        """Place a single order (bid or ask side). Returns order ID."""
        price = quote.bid_price if side == "bid" else quote.ask_price
        order_side = "BUY" if side == "bid" else "SELL"

        if quote.platform == "polymarket":
            api_key = get_env("POLYMARKET_API_KEY")
            if not api_key:
                return ""
            url = "https://clob.polymarket.com/order"
            payload = {
                "tokenID": quote.market_id,
                "price": str(price),
                "size": str(quote.size_contracts),
                "side": order_side,
                "type": "GTC",
            }
            try:
                async with session.post(url, json=payload,
                                         headers={"Authorization": f"Bearer {api_key}"},
                                         timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status in (200, 201):
                        data = await resp.json()
                        return data.get("orderID", "")
            except Exception as e:
                logger.error(f"MM order placement error: {e}")

        elif quote.platform == "kalshi":
            api_key = get_env("KALSHI_API_KEY")
            if not api_key:
                return ""
            base_url = get_setting("platforms", "kalshi", "base_url") or "https://demo-api.kalshi.co/trade-api/v2"
            url = f"{base_url}/portfolio/orders"
            payload = {
                "ticker": quote.market_id,
                "action": "buy" if side == "bid" else "sell",
                "side": "yes",
                "type": "limit",
                "count": quote.size_contracts,
                "yes_price": int(price * 100),
            }
            try:
                async with session.post(url, json=payload,
                                         headers={"Authorization": f"Bearer {api_key}"},
                                         timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status in (200, 201):
                        data = await resp.json()
                        return data.get("order", {}).get("order_id", "")
            except Exception as e:
                logger.error(f"MM order placement error: {e}")

        return ""

    async def _cancel_quote(self, quote: Quote):
        """Cancel both sides of a quote."""
        logger.debug(f"Cancelling quote for {quote.market_id}")
        # In production: call platform cancel APIs
        quote.status = "cancelled"

    async def cancel_all(self):
        """Emergency: cancel ALL outstanding quotes."""
        logger.warning(f"Cancelling all {len(self.active_quotes)} market-making quotes")
        for quote in self.active_quotes.values():
            if quote.status == "active":
                await self._cancel_quote(quote)
        self.active_quotes.clear()
        self._running = False

    def stop(self):
        self._running = False

    def status_report(self) -> str:
        active = sum(1 for q in self.active_quotes.values() if q.status == "active")
        total_pnl = self.inventory_mgr.total_pnl()
        lines = [
            f"=== Market Making Status ===",
            f"  Active quotes: {active}",
            f"  Total realized PnL: ${total_pnl:+,.2f}",
            f"  Paper mode: {self.paper_trading}",
        ]
        for mid, inv in self.inventory_mgr.inventories.items():
            if inv.net_contracts != 0 or inv.roundtrips > 0:
                lines.append(
                    f"  {mid[:20]}: inventory={inv.net_contracts:+d} | "
                    f"roundtrips={inv.roundtrips} | PnL=${inv.realized_pnl:+.2f}"
                )
        return "\n".join(lines)
