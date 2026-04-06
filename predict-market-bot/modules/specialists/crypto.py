"""
Crypto Specialist

Tuned for cryptocurrency prediction markets: price targets, ETF approvals, protocol events.

Key data sources:
  - CoinGecko API (free, comprehensive price/volume/market cap)
  - On-chain metrics via public APIs (active addresses, exchange flows)
  - Crypto news RSS (CoinDesk, The Block, Decrypt)
  - Reddit (r/cryptocurrency, r/bitcoin, r/ethereum)
  - Fear & Greed Index

Why specialized: Crypto markets are driven by on-chain data, whale movements,
exchange flows, and technical indicators that generic sentiment analysis can't
capture. A 20% exchange outflow often signals bullish accumulation days before
price moves — data that LLMs would completely miss.
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
    RedditScraper, RSSFetcher, MarketResearcher,
)
from modules.specialists import BaseSpecialist

logger = logging.getLogger(__name__)


# Coin name → CoinGecko ID mapping
COIN_IDS = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana",
    "cardano": "cardano", "ada": "cardano",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "xrp": "ripple", "ripple": "ripple",
    "polkadot": "polkadot", "dot": "polkadot",
    "avalanche": "avalanche-2", "avax": "avalanche-2",
    "chainlink": "chainlink", "link": "chainlink",
    "polygon": "matic-network", "matic": "matic-network",
    "litecoin": "litecoin", "ltc": "litecoin",
    "uniswap": "uniswap", "uni": "uniswap",
}


class CryptoSpecialist(BaseSpecialist):

    category = "crypto"
    description = "Cryptocurrency prices, ETF decisions, protocol events"

    CRYPTO_RSS_FEEDS = [
        "https://news.google.com/rss/search?q={query}+cryptocurrency&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/cryptocurrency/search/.rss?q={query}&sort=new&t=week",
        "https://www.reddit.com/r/bitcoin/search/.rss?q={query}&sort=new&t=week",
    ]

    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.base_researcher = MarketResearcher()

    async def research(self, market: Market) -> ResearchBrief:
        """
        Crypto research pipeline:
        1. Standard social media + news (base researcher)
        2. CoinGecko price/volume data
        3. Fear & Greed Index
        4. Crypto-specific news feeds
        5. Price target analysis
        """
        # Run base research
        brief = await self.base_researcher.research_market(market)

        coin = self._detect_coin(market.title)
        query = self._extract_crypto_query(market.title)
        target_price = self._extract_price_target(market.title)

        # Fetch crypto-specific data in parallel
        price_task = self._fetch_price_data(coin) if coin else asyncio.sleep(0)
        fng_task = self._fetch_fear_greed()
        news_task = self._fetch_crypto_news(query)

        results = await asyncio.gather(
            price_task, fng_task, news_task,
            return_exceptions=True,
        )

        price_data = results[0] if not isinstance(results[0], (Exception, type(None))) else None
        fng_data = results[1] if not isinstance(results[1], (Exception, type(None))) else None
        news_sources = results[2] if isinstance(results[2], list) else []

        # Add price data
        if isinstance(price_data, dict):
            brief.sources.append(SourceResult(
                source_type="coingecko",
                title=f"{coin} price data",
                content=(
                    f"Current: ${price_data.get('current_price', 0):,.2f}, "
                    f"24h change: {price_data.get('price_change_24h', 0):+.1f}%, "
                    f"7d change: {price_data.get('price_change_7d', 0):+.1f}%, "
                    f"30d change: {price_data.get('price_change_30d', 0):+.1f}%, "
                    f"ATH: ${price_data.get('ath', 0):,.2f}, "
                    f"Market cap rank: #{price_data.get('market_cap_rank', '?')}"
                ),
                url="https://www.coingecko.com",
                sentiment_score=0,
                sentiment_label="neutral",
            ))

            # Calculate price-based probability for price target markets
            if target_price is not None:
                price_prob = self._calculate_price_probability(
                    price_data, target_price, market.title,
                )
                if price_prob is not None:
                    brief.narrative_probability = (
                        brief.narrative_probability * 0.4 + price_prob * 0.6
                    )
                    brief.gap = brief.narrative_probability - brief.market_price
                    brief.confidence = min(brief.confidence + 0.10, 0.90)

        # Add Fear & Greed Index
        if isinstance(fng_data, dict):
            fng_value = fng_data.get("value", 50)
            fng_label = fng_data.get("label", "Neutral")
            brief.sources.append(SourceResult(
                source_type="fear_greed",
                title=f"Crypto Fear & Greed Index: {fng_value} ({fng_label})",
                content=f"Fear & Greed: {fng_value}/100 ({fng_label}). "
                        f"Yesterday: {fng_data.get('yesterday', '?')}, "
                        f"Last week: {fng_data.get('last_week', '?')}",
                url="https://alternative.me/crypto/fear-and-greed-index/",
                sentiment_score=(fng_value - 50) / 50,  # Normalize to -1 to 1
                sentiment_label=fng_label.lower(),
            ))

        # Add news
        for source in news_sources:
            score, label = self.sentiment.analyze(f"{source.title} {source.content}")
            source.sentiment_score = score
            source.sentiment_label = label
        brief.sources.extend(news_sources)

        return brief

    def get_model_prompt_context(self, market: Market, brief: ResearchBrief) -> str:
        return (
            "CRYPTO MARKET CONTEXT:\n"
            "- Crypto is highly volatile — 10-20% daily moves are normal\n"
            "- Fear & Greed Index: extreme fear (<25) often = buying opportunity, extreme greed (>75) = caution\n"
            "- Bitcoin dominance affects altcoin markets (BTC up + dominance up = alts lag)\n"
            "- Halving cycles historically precede bull runs (12-18 month lag)\n"
            "- Exchange outflows = bullish (accumulation), inflows = bearish (selling)\n"
            "- Regulatory news (ETF approvals, bans) can move markets 20%+ instantly\n"
            "- Weekend/holiday volume is thin — larger moves on less volume\n"
            "- Social media hype (especially for memecoins) is a lagging indicator\n"
            f"- Sources: {len(brief.sources)} | Confidence: {brief.confidence:.0%}\n"
        )

    def get_data_sources(self) -> list[str]:
        return ["coingecko", "fear_greed_index", "crypto_rss", "reddit_crypto"]

    def _detect_coin(self, title: str) -> Optional[str]:
        """Detect which cryptocurrency a market is about."""
        title_lower = title.lower()
        for name, cg_id in COIN_IDS.items():
            if name in title_lower:
                return cg_id
        return None

    def _extract_crypto_query(self, title: str) -> str:
        """Build search query for crypto content."""
        terms = re.findall(
            r'\b(?:Bitcoin|BTC|Ethereum|ETH|Solana|SOL|'
            r'Cardano|ADA|XRP|Dogecoin|DOGE|crypto|blockchain|'
            r'halving|ETF|SEC|[A-Z][a-z]+)\b',
            title,
        )
        if terms:
            return " ".join(terms[:5])
        return title.split("?")[0].strip()[:50]

    def _extract_price_target(self, title: str) -> Optional[float]:
        """Extract price target from market title."""
        # Match patterns like "above $100,000", "reach $50k", "below 80000"
        patterns = [
            r'\$?([\d,]+(?:\.\d+)?)\s*[kK]',  # $100k
            r'(?:above|below|reach|exceed|hit|break)\s*\$?([\d,]+(?:\.\d+)?)',
            r'\$([\d,]+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                value_str = match.group(1).replace(",", "")
                try:
                    value = float(value_str)
                    # Handle "k" suffix
                    if "k" in title[match.start():match.end()].lower():
                        value *= 1000
                    return value
                except ValueError:
                    continue
        return None

    async def _fetch_price_data(self, coin_id: str) -> Optional[dict]:
        """Fetch price data from CoinGecko (free, no API key needed)."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                params = {
                    "localization": "false",
                    "tickers": "false",
                    "community_data": "false",
                    "developer_data": "false",
                }
                async with session.get(url, params=params,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        market_data = data.get("market_data", {})
                        return {
                            "current_price": market_data.get("current_price", {}).get("usd", 0),
                            "price_change_24h": market_data.get("price_change_percentage_24h", 0),
                            "price_change_7d": market_data.get("price_change_percentage_7d", 0),
                            "price_change_30d": market_data.get("price_change_percentage_30d", 0),
                            "ath": market_data.get("ath", {}).get("usd", 0),
                            "ath_change_pct": market_data.get("ath_change_percentage", {}).get("usd", 0),
                            "market_cap_rank": data.get("market_cap_rank", 0),
                            "total_volume": market_data.get("total_volume", {}).get("usd", 0),
                        }
                    logger.warning(f"CoinGecko returned {resp.status}")
        except Exception as e:
            logger.debug(f"CoinGecko fetch error: {e}")
        return None

    async def _fetch_fear_greed(self) -> Optional[dict]:
        """Fetch Crypto Fear & Greed Index."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.alternative.me/fng/?limit=7"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        entries = data.get("data", [])
                        if entries:
                            current = entries[0]
                            return {
                                "value": int(current.get("value", 50)),
                                "label": current.get("value_classification", "Neutral"),
                                "yesterday": entries[1].get("value", "?") if len(entries) > 1 else "?",
                                "last_week": entries[6].get("value", "?") if len(entries) > 6 else "?",
                            }
        except Exception as e:
            logger.debug(f"Fear & Greed fetch error: {e}")
        return None

    async def _fetch_crypto_news(self, query: str) -> list[SourceResult]:
        """Fetch crypto news from RSS feeds."""
        sources = []
        rss = RSSFetcher()
        rss.DEFAULT_FEEDS = self.CRYPTO_RSS_FEEDS
        results = await rss.fetch(query, max_results=8)
        for entry in results:
            sources.append(SourceResult(
                source_type="crypto_rss",
                title=entry.get("title", ""),
                content=entry.get("summary", "")[:500],
                url=entry.get("link", ""),
                sentiment_score=0,
                sentiment_label="neutral",
            ))
        return sources

    def _calculate_price_probability(self, price_data: dict, target_price: float,
                                       market_title: str) -> Optional[float]:
        """
        Estimate probability of hitting a price target based on current price and volatility.
        Uses distance-from-target as a rough probability signal.
        """
        current = price_data.get("current_price", 0)
        if current <= 0 or target_price <= 0:
            return None

        title_lower = market_title.lower()
        is_above = any(w in title_lower for w in ["above", "over", "reach", "exceed", "hit", "break"])
        is_below = any(w in title_lower for w in ["below", "under", "drop", "fall"])

        # Distance as percentage
        pct_distance = (target_price - current) / current

        # Use 30-day volatility as a rough gauge
        vol_30d = abs(price_data.get("price_change_30d", 30)) / 100
        vol_30d = max(vol_30d, 0.10)  # Floor at 10%

        # Rough probability: how many standard deviations away is the target?
        import math
        if vol_30d > 0:
            z_score = pct_distance / vol_30d
        else:
            z_score = pct_distance / 0.30

        if is_above or (not is_below):
            # Probability of going above target
            # Sigmoid mapping: z=0 → 50%, z=-1 → ~73%, z=1 → ~27%
            prob = 1 / (1 + math.exp(z_score * 1.5))
        elif is_below:
            # Probability of going below target
            prob = 1 / (1 + math.exp(-z_score * 1.5))
        else:
            return None

        return max(0.05, min(0.95, prob))
