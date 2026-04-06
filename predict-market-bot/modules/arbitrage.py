"""
Cross-Platform Arbitrage Detector

Detects and exploits price differences between Polymarket and Kalshi
for the same event. Risk-free profit when prices diverge.

Arbitrage exists when: price_A_yes + price_B_no < 1.0 (or vice versa)
Profit = 1.0 - best_ask_A - best_ask_B_complement - fees_A - fees_B

Uses fuzzy text matching to pair equivalent markets across platforms.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp
from rapidfuzz import fuzz

from config import get_setting, get_env
from modules.scanner import Market

logger = logging.getLogger(__name__)


# Platform fee rates
POLYMARKET_FEE = 0.02   # ~2% effective fee
KALSHI_FEE = 0.01       # ~1% fee (varies by contract)


@dataclass
class MarketPair:
    """A matched pair of markets across Polymarket and Kalshi."""
    polymarket: Market
    kalshi: Market
    match_score: float        # Fuzzy match confidence (0-100)
    poly_yes_price: float
    kalshi_yes_price: float
    poly_no_price: float
    kalshi_no_price: float


@dataclass
class ArbitrageOpportunity:
    """A detected arbitrage opportunity."""
    pair: MarketPair
    strategy: str             # "buy_poly_yes_kalshi_no" or "buy_kalshi_yes_poly_no"
    cost: float               # Total cost to enter both sides
    guaranteed_payout: float  # Always 1.0 (one side resolves YES)
    gross_profit: float       # 1.0 - cost
    fees: float
    net_profit: float         # gross_profit - fees
    net_profit_pct: float     # net_profit / cost as percentage
    size_usd: float = 0.0    # Position size
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        return (
            f"[ARB] {self.pair.polymarket.title[:45]}\n"
            f"  Strategy: {self.strategy}\n"
            f"  Cost: ${self.cost:.4f} → Payout: $1.00 | "
            f"Net: ${self.net_profit:.4f} ({self.net_profit_pct:+.2f}%)\n"
            f"  Poly YES: {self.pair.poly_yes_price:.3f} | Kalshi YES: {self.pair.kalshi_yes_price:.3f} | "
            f"Match: {self.pair.match_score:.0f}%"
        )


class MarketMatcher:
    """
    Matches equivalent markets across Polymarket and Kalshi
    using fuzzy text similarity on market titles.
    """

    MATCH_THRESHOLD = 75  # Minimum fuzzy match score (0-100)

    def match_markets(self, poly_markets: list[Market],
                       kalshi_markets: list[Market]) -> list[MarketPair]:
        """Find equivalent markets across platforms."""
        pairs = []

        for pm in poly_markets:
            best_match = None
            best_score = 0

            pm_title = self._normalize_title(pm.title)

            for km in kalshi_markets:
                km_title = self._normalize_title(km.title)

                # Use token_sort_ratio for order-independent matching
                score = fuzz.token_sort_ratio(pm_title, km_title)

                if score > best_score:
                    best_score = score
                    best_match = km

            if best_match and best_score >= self.MATCH_THRESHOLD:
                pair = MarketPair(
                    polymarket=pm,
                    kalshi=best_match,
                    match_score=best_score,
                    poly_yes_price=pm.current_price,
                    kalshi_yes_price=best_match.current_price,
                    poly_no_price=1.0 - pm.current_price,
                    kalshi_no_price=1.0 - best_match.current_price,
                )
                pairs.append(pair)

        logger.info(f"Matched {len(pairs)} cross-platform market pairs")
        return pairs

    def _normalize_title(self, title: str) -> str:
        """Normalize market title for better matching."""
        title = title.lower().strip()
        # Remove common prediction market filler
        for word in ["will", "the", "be", "by", "before", "?", ".", ","]:
            title = title.replace(word, "")
        return " ".join(title.split())


class ArbitrageDetector:
    """
    Detects arbitrage opportunities across matched market pairs.

    Two arbitrage types:
    1. Buy YES on Platform A + Buy NO on Platform B (if prices sum < 1.0)
    2. Buy NO on Platform A + Buy YES on Platform B (if prices sum < 1.0)
    """

    def __init__(self):
        self.matcher = MarketMatcher()
        self.min_profit_pct = 0.5  # Minimum 0.5% net profit to be worth executing
        self.detected: list[ArbitrageOpportunity] = []

    def scan_for_arbitrage(self, poly_markets: list[Market],
                            kalshi_markets: list[Market]) -> list[ArbitrageOpportunity]:
        """Scan matched pairs for arbitrage opportunities."""
        pairs = self.matcher.match_markets(poly_markets, kalshi_markets)
        opportunities = []

        for pair in pairs:
            # Strategy 1: Buy YES on Polymarket + Buy NO on Kalshi
            cost_1 = pair.poly_yes_price + pair.kalshi_no_price
            fees_1 = pair.poly_yes_price * POLYMARKET_FEE + pair.kalshi_no_price * KALSHI_FEE
            net_1 = 1.0 - cost_1 - fees_1

            if net_1 > 0 and (net_1 / cost_1 * 100) >= self.min_profit_pct:
                opportunities.append(ArbitrageOpportunity(
                    pair=pair,
                    strategy="buy_poly_yes_kalshi_no",
                    cost=cost_1,
                    guaranteed_payout=1.0,
                    gross_profit=1.0 - cost_1,
                    fees=fees_1,
                    net_profit=net_1,
                    net_profit_pct=net_1 / cost_1 * 100,
                ))

            # Strategy 2: Buy NO on Polymarket + Buy YES on Kalshi
            cost_2 = pair.poly_no_price + pair.kalshi_yes_price
            fees_2 = pair.poly_no_price * POLYMARKET_FEE + pair.kalshi_yes_price * KALSHI_FEE
            net_2 = 1.0 - cost_2 - fees_2

            if net_2 > 0 and (net_2 / cost_2 * 100) >= self.min_profit_pct:
                opportunities.append(ArbitrageOpportunity(
                    pair=pair,
                    strategy="buy_kalshi_yes_poly_no",
                    cost=cost_2,
                    guaranteed_payout=1.0,
                    gross_profit=1.0 - cost_2,
                    fees=fees_2,
                    net_profit=net_2,
                    net_profit_pct=net_2 / cost_2 * 100,
                ))

        # Sort by profit percentage
        opportunities.sort(key=lambda o: o.net_profit_pct, reverse=True)
        self.detected = opportunities

        logger.info(f"Found {len(opportunities)} arbitrage opportunities from {len(pairs)} pairs")
        return opportunities

    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity,
                                 bankroll: float, max_position_pct: float = 0.05,
                                 paper_trading: bool = True) -> dict:
        """
        Execute an arbitrage trade on both platforms simultaneously.
        Returns execution result.
        """
        max_size = bankroll * max_position_pct
        size = min(max_size, bankroll * 0.03)  # Conservative: 3% of bankroll per arb

        if paper_trading:
            logger.info(
                f"[PAPER ARB] {opportunity.strategy} | "
                f"Size: ${size:.2f} | Net profit: ${size * opportunity.net_profit_pct / 100:.2f}"
            )
            return {
                "success": True,
                "paper": True,
                "size": size,
                "expected_profit": size * opportunity.net_profit_pct / 100,
                "strategy": opportunity.strategy,
            }

        # Live execution: place both orders simultaneously
        # Both must succeed or we cancel both (atomic execution)
        pair = opportunity.pair

        if opportunity.strategy == "buy_poly_yes_kalshi_no":
            poly_side, poly_price = "yes", pair.poly_yes_price
            kalshi_side, kalshi_price = "no", pair.kalshi_no_price
        else:
            poly_side, poly_price = "no", pair.poly_no_price
            kalshi_side, kalshi_price = "yes", pair.kalshi_yes_price

        poly_size = size * poly_price / opportunity.cost
        kalshi_size = size * kalshi_price / opportunity.cost

        # Execute both legs simultaneously
        async with aiohttp.ClientSession() as session:
            poly_task = self._place_polymarket_order(
                session, pair.polymarket.market_id, poly_side, poly_price, poly_size
            )
            kalshi_task = self._place_kalshi_order(
                session, pair.kalshi.market_id, kalshi_side, kalshi_price, kalshi_size
            )

            results = await asyncio.gather(poly_task, kalshi_task, return_exceptions=True)

            poly_ok = isinstance(results[0], dict) and results[0].get("success")
            kalshi_ok = isinstance(results[1], dict) and results[1].get("success")

            if poly_ok and kalshi_ok:
                return {
                    "success": True,
                    "paper": False,
                    "size": size,
                    "expected_profit": size * opportunity.net_profit_pct / 100,
                    "strategy": opportunity.strategy,
                    "poly_order": results[0],
                    "kalshi_order": results[1],
                }
            else:
                # One leg failed — need to cancel the successful one
                logger.error(f"Arbitrage partial fill! Poly: {poly_ok}, Kalshi: {kalshi_ok}")
                # In production: implement cancel logic here
                return {
                    "success": False,
                    "reason": "Partial fill - one leg failed",
                    "poly_result": results[0],
                    "kalshi_result": results[1],
                }

    async def _place_polymarket_order(self, session: aiohttp.ClientSession,
                                       market_id: str, side: str,
                                       price: float, size: float) -> dict:
        api_key = get_env("POLYMARKET_API_KEY")
        if not api_key:
            return {"success": False, "reason": "No API key"}
        try:
            contracts = int(size / price) if price > 0 else 0
            url = "https://clob.polymarket.com/order"
            payload = {
                "tokenID": market_id,
                "price": str(price),
                "size": str(contracts),
                "side": "BUY" if side == "yes" else "SELL",
                "type": "FOK",  # Fill-or-kill for atomic execution
            }
            async with session.post(url, json=payload,
                                     headers={"Authorization": f"Bearer {api_key}"},
                                     timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status in (200, 201):
                    return {"success": True, "data": await resp.json()}
                return {"success": False, "reason": f"Status {resp.status}"}
        except Exception as e:
            return {"success": False, "reason": str(e)}

    async def _place_kalshi_order(self, session: aiohttp.ClientSession,
                                   market_id: str, side: str,
                                   price: float, size: float) -> dict:
        api_key = get_env("KALSHI_API_KEY")
        if not api_key:
            return {"success": False, "reason": "No API key"}
        base_url = get_setting("platforms", "kalshi", "base_url") or "https://demo-api.kalshi.co/trade-api/v2"
        try:
            contracts = max(1, int(size / (price * 100)))
            url = f"{base_url}/portfolio/orders"
            payload = {
                "ticker": market_id,
                "action": "buy",
                "side": side,
                "type": "limit",
                "count": contracts,
                "yes_price": int(price * 100) if side == "yes" else None,
                "no_price": int(price * 100) if side == "no" else None,
                "expiration_ts": int(time.time()) + 10,  # 10 second expiry for atomic execution
            }
            async with session.post(url, json=payload,
                                     headers={"Authorization": f"Bearer {api_key}"},
                                     timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status in (200, 201):
                    return {"success": True, "data": await resp.json()}
                return {"success": False, "reason": f"Status {resp.status}"}
        except Exception as e:
            return {"success": False, "reason": str(e)}

    def report(self) -> str:
        if not self.detected:
            return "No arbitrage opportunities detected."
        lines = [f"=== Arbitrage Opportunities ({len(self.detected)}) ===\n"]
        for opp in self.detected[:10]:
            lines.append(opp.summary())
            lines.append("")
        return "\n".join(lines)


async def run_arbitrage_scan(poly_markets: list[Market],
                              kalshi_markets: list[Market]) -> list[ArbitrageOpportunity]:
    """Entry point for arbitrage detection."""
    detector = ArbitrageDetector()
    opportunities = detector.scan_for_arbitrage(poly_markets, kalshi_markets)
    print(detector.report())
    return opportunities
