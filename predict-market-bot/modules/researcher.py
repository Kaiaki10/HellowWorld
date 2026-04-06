"""
Step 2: Research - Gather Intelligence

For each market flagged by the scanner, scrapes Twitter, Reddit, RSS, and news.
Runs NLP sentiment analysis and compares narrative signal against market odds.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import feedparser
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import get_setting, get_env
from modules.scanner import Market

logger = logging.getLogger(__name__)


@dataclass
class SourceResult:
    """A single piece of sourced intelligence."""
    source_type: str       # "twitter", "reddit", "rss", "news"
    title: str
    content: str
    url: str
    timestamp: Optional[datetime] = None
    sentiment_score: float = 0.0   # -1.0 (bearish) to +1.0 (bullish)
    sentiment_label: str = "neutral"
    relevance_score: float = 0.0


@dataclass
class ResearchBrief:
    """Aggregated research output for one market."""
    market: Market
    sources: list[SourceResult] = field(default_factory=list)
    overall_sentiment: float = 0.0
    sentiment_label: str = "neutral"
    narrative_consensus: str = ""
    market_price: float = 0.0
    narrative_probability: float = 0.0
    gap: float = 0.0              # narrative probability - market price
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        return (
            f"Market: {self.market.title[:60]}\n"
            f"  Market Price: {self.market_price:.2f} | "
            f"Narrative Est: {self.narrative_probability:.2f} | "
            f"Gap: {self.gap:+.2f}\n"
            f"  Sentiment: {self.sentiment_label} ({self.overall_sentiment:+.3f}) | "
            f"Sources: {len(self.sources)} | Confidence: {self.confidence:.2f}"
        )


class SentimentAnalyzer:
    """Dual-method sentiment analysis using VADER and TextBlob."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> tuple[float, str]:
        """
        Returns (score, label) where score is -1 to 1.
        Uses average of VADER compound and TextBlob polarity.
        """
        if not text.strip():
            return 0.0, "neutral"

        # Sanitize: treat content as data, never as instructions
        clean_text = self._sanitize(text)

        # VADER
        vader_scores = self.vader.polarity_scores(clean_text)
        vader_compound = vader_scores["compound"]

        # TextBlob
        blob = TextBlob(clean_text)
        textblob_polarity = blob.sentiment.polarity

        # Average both
        score = (vader_compound + textblob_polarity) / 2.0
        score = max(-1.0, min(1.0, score))

        if score > 0.1:
            label = "bullish"
        elif score < -0.1:
            label = "bearish"
        else:
            label = "neutral"

        return score, label

    def _sanitize(self, text: str) -> str:
        """
        Sanitize external content to prevent prompt injection.
        Strip anything that looks like system/prompt instructions.
        """
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove common injection patterns
        text = re.sub(r"(ignore previous|system prompt|you are now|act as)", "", text, flags=re.IGNORECASE)
        # Limit length
        return text[:2000]


class TwitterScraper:
    """Fetch recent tweets about a market topic."""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.bearer = get_env("TWITTER_BEARER_TOKEN")

    async def search(self, query: str, max_results: int = 20) -> list[dict]:
        if not self.bearer:
            logger.debug("Twitter bearer token not configured")
            return []

        try:
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {self.bearer}"}
            params = {
                "query": f"{query} -is:retweet lang:en",
                "max_results": min(max_results, 100),
                "tweet.fields": "created_at,public_metrics,text",
            }
            async with self.session.get(url, headers=headers, params=params,
                                         timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", [])
                logger.warning(f"Twitter API returned {resp.status}")
        except Exception as e:
            logger.error(f"Twitter search error: {e}")
        return []


class RedditScraper:
    """Fetch recent Reddit posts about a market topic."""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def search(self, query: str, max_results: int = 15) -> list[dict]:
        try:
            # Use Reddit's public JSON API (no auth needed for search)
            url = "https://www.reddit.com/search.json"
            headers = {"User-Agent": "PredictMarketBot/1.0"}
            params = {
                "q": query,
                "sort": "relevance",
                "t": "week",
                "limit": max_results,
            }
            async with self.session.get(url, headers=headers, params=params,
                                         timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    posts = data.get("data", {}).get("children", [])
                    return [p.get("data", {}) for p in posts]
                logger.warning(f"Reddit API returned {resp.status}")
        except Exception as e:
            logger.error(f"Reddit search error: {e}")
        return []


class RSSFetcher:
    """Fetch news from RSS feeds relevant to prediction markets."""

    DEFAULT_FEEDS = [
        "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/predictit/.rss",
        "https://www.reddit.com/r/polymarket/.rss",
    ]

    async def fetch(self, query: str, max_results: int = 10) -> list[dict]:
        results = []
        for feed_url_template in self.DEFAULT_FEEDS:
            try:
                feed_url = feed_url_template.format(query=query.replace(" ", "+"))
                feed = await asyncio.to_thread(feedparser.parse, feed_url)
                for entry in feed.entries[:max_results]:
                    results.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                    })
            except Exception as e:
                logger.debug(f"RSS fetch error for {feed_url_template}: {e}")
        return results[:max_results]


class ResearchCache:
    """
    Caches ResearchBriefs to avoid redundant scraping and sentiment analysis.
    If a market's price moved < 2% since last research, reuse the cached brief.
    """

    def __init__(self, cache_ttl_seconds: int = 900):  # 15 min default TTL
        self.cache: dict[str, dict] = {}  # market_id -> {brief, price, timestamp}
        self.cache_ttl = cache_ttl_seconds
        self.price_change_threshold = 0.02  # 2% price change invalidates cache
        self.hits = 0
        self.misses = 0

    def get(self, market: Market) -> Optional[ResearchBrief]:
        """Return cached brief if still valid, else None."""
        key = f"{market.platform}:{market.market_id}"
        entry = self.cache.get(key)
        if entry is None:
            self.misses += 1
            return None

        # Check TTL
        age = (datetime.now(timezone.utc) - entry["timestamp"]).total_seconds()
        if age > self.cache_ttl:
            self.misses += 1
            del self.cache[key]
            return None

        # Check price change
        cached_price = entry["price"]
        if cached_price > 0:
            price_change = abs(market.current_price - cached_price) / cached_price
            if price_change > self.price_change_threshold:
                self.misses += 1
                del self.cache[key]
                return None

        self.hits += 1
        brief = entry["brief"]
        # Update the market reference to current prices
        brief.market = market
        brief.market_price = market.current_price
        brief.gap = brief.narrative_probability - market.current_price
        logger.debug(f"Cache HIT: {market.title[:40]} (age {age:.0f}s)")
        return brief

    def put(self, market: Market, brief: ResearchBrief):
        key = f"{market.platform}:{market.market_id}"
        self.cache[key] = {
            "brief": brief,
            "price": market.current_price,
            "timestamp": datetime.now(timezone.utc),
        }

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total if total > 0 else 0
        return f"Research cache: {self.hits}/{total} hits ({rate:.0%}), {len(self.cache)} entries"


class MarketResearcher:
    """
    Orchestrates research across multiple sources for a given market.
    Produces a ResearchBrief with sentiment analysis and gap detection.
    Uses caching to avoid redundant API calls when prices haven't moved.
    """

    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.max_sources = get_setting("research", "max_sources_per_market") or 20
        self.cache = ResearchCache(cache_ttl_seconds=900)

    async def research_market(self, market: Market) -> ResearchBrief:
        """Research a single market, using cache when available."""
        # Check cache first
        cached = self.cache.get(market)
        if cached is not None:
            return cached

        logger.info(f"Researching: {market.title[:50]}...")

        # Build search query from market title
        query = self._build_query(market.title)

        async with aiohttp.ClientSession() as session:
            twitter = TwitterScraper(session)
            reddit = RedditScraper(session)
            rss = RSSFetcher()

            # Fetch all sources in parallel
            twitter_results, reddit_results, rss_results = await asyncio.gather(
                twitter.search(query),
                reddit.search(query),
                rss.fetch(query),
            )

            # Process results into SourceResults
            sources: list[SourceResult] = []
            sources.extend(self._process_twitter(twitter_results))
            sources.extend(self._process_reddit(reddit_results))
            sources.extend(self._process_rss(rss_results))

        # Limit sources
        sources = sources[:self.max_sources]

        # Run sentiment analysis on all sources
        for source in sources:
            text = f"{source.title} {source.content}"
            score, label = self.sentiment.analyze(text)
            source.sentiment_score = score
            source.sentiment_label = label

        # Build the research brief
        brief = self._build_brief(market, sources)

        # Cache it
        self.cache.put(market, brief)

        logger.info(f"Research complete: {brief.summary()}")
        return brief

    async def research_markets(self, markets: list[Market], max_concurrent: int = 5) -> list[ResearchBrief]:
        """Research multiple markets with concurrency control and caching."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_research(m: Market) -> ResearchBrief:
            async with semaphore:
                return await self.research_market(m)

        briefs = await asyncio.gather(*[_bounded_research(m) for m in markets])
        logger.info(self.cache.stats())
        return list(briefs)

    def _build_query(self, title: str) -> str:
        """Extract key terms from market title for search."""
        # Remove common filler words
        stopwords = {"will", "the", "be", "a", "an", "in", "on", "at", "to", "by", "of", "for", "is", "or", "and"}
        words = title.split()
        keywords = [w for w in words if w.lower().strip("?.,!") not in stopwords]
        return " ".join(keywords[:6])

    def _process_twitter(self, tweets: list[dict]) -> list[SourceResult]:
        results = []
        for tweet in tweets:
            results.append(SourceResult(
                source_type="twitter",
                title="",
                content=tweet.get("text", ""),
                url=f"https://twitter.com/i/status/{tweet.get('id', '')}",
            ))
        return results

    def _process_reddit(self, posts: list[dict]) -> list[SourceResult]:
        results = []
        for post in posts:
            results.append(SourceResult(
                source_type="reddit",
                title=post.get("title", ""),
                content=post.get("selftext", "")[:500],
                url=f"https://reddit.com{post.get('permalink', '')}",
            ))
        return results

    def _process_rss(self, entries: list[dict]) -> list[SourceResult]:
        results = []
        for entry in entries:
            results.append(SourceResult(
                source_type="rss",
                title=entry.get("title", ""),
                content=entry.get("summary", "")[:500],
                url=entry.get("link", ""),
            ))
        return results

    def _build_brief(self, market: Market, sources: list[SourceResult]) -> ResearchBrief:
        """Aggregate source sentiments into a research brief."""
        if not sources:
            return ResearchBrief(
                market=market,
                market_price=market.current_price,
                narrative_probability=market.current_price,
                gap=0.0,
                confidence=0.0,
            )

        scores = [s.sentiment_score for s in sources]
        avg_sentiment = sum(scores) / len(scores) if scores else 0.0

        # Convert sentiment (-1 to 1) into a probability estimate (0 to 1)
        # Sentiment of +1 -> ~0.85 probability, -1 -> ~0.15, 0 -> 0.50
        narrative_prob = 0.5 + (avg_sentiment * 0.35)
        narrative_prob = max(0.05, min(0.95, narrative_prob))

        gap = narrative_prob - market.current_price

        # Confidence based on source agreement and count
        if len(scores) >= 3:
            import numpy as np
            std = float(np.std(scores))
            agreement = max(0, 1.0 - std)  # High agreement = low std
            count_factor = min(len(scores) / 10, 1.0)
            confidence = agreement * 0.6 + count_factor * 0.4
        else:
            confidence = 0.2

        if avg_sentiment > 0.1:
            label = "bullish"
        elif avg_sentiment < -0.1:
            label = "bearish"
        else:
            label = "neutral"

        # Determine narrative consensus
        bullish_count = sum(1 for s in scores if s > 0.1)
        bearish_count = sum(1 for s in scores if s < -0.1)
        total = len(scores)
        if bullish_count > total * 0.6:
            consensus = f"Strong bullish consensus ({bullish_count}/{total} sources)"
        elif bearish_count > total * 0.6:
            consensus = f"Strong bearish consensus ({bearish_count}/{total} sources)"
        elif bullish_count > bearish_count:
            consensus = f"Lean bullish ({bullish_count} vs {bearish_count} bearish)"
        elif bearish_count > bullish_count:
            consensus = f"Lean bearish ({bearish_count} vs {bullish_count} bullish)"
        else:
            consensus = "Mixed/unclear signals"

        return ResearchBrief(
            market=market,
            sources=sources,
            overall_sentiment=avg_sentiment,
            sentiment_label=label,
            narrative_consensus=consensus,
            market_price=market.current_price,
            narrative_probability=narrative_prob,
            gap=gap,
            confidence=confidence,
        )


async def run_research(markets: list[Market]) -> list[ResearchBrief]:
    """Entry point for researching a list of markets."""
    researcher = MarketResearcher()
    briefs = await researcher.research_markets(markets)

    print(f"\n=== Research Results ({len(briefs)} markets) ===\n")
    for brief in briefs:
        print(brief.summary())
        print()

    return briefs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Standalone test requires markets from scanner
    from modules.scanner import run_scan
    markets = asyncio.run(run_scan())
    if markets:
        asyncio.run(run_research(markets[:5]))
