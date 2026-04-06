"""
Politics Specialist

Tuned for political prediction markets: elections, legislation, court cases.

Key data sources:
  - Polling aggregators (FiveThirtyEight, RealClearPolitics, 270toWin)
  - Congressional activity (congress.gov)
  - Court dockets (PACER/RECAP)
  - Political news RSS feeds
  - Standard social media (Twitter, Reddit r/politics)

Why specialized: Political markets are driven by polling data, legal proceedings,
and institutional actions. Generic sentiment analysis misses the structured data
(poll numbers, vote counts, court schedules) that actually moves these markets.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone

import aiohttp

from modules.scanner import Market
from modules.researcher import (
    ResearchBrief, SourceResult, SentimentAnalyzer,
    TwitterScraper, RedditScraper, RSSFetcher, MarketResearcher,
)
from modules.specialists import BaseSpecialist

logger = logging.getLogger(__name__)


class PoliticsSpecialist(BaseSpecialist):

    category = "politics"
    description = "Elections, legislation, court cases, and political events"

    POLITICAL_RSS_FEEDS = [
        "https://news.google.com/rss/search?q={query}+politics&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/politics/search/.rss?q={query}&sort=new&t=week",
        "https://www.reddit.com/r/PoliticalDiscussion/search/.rss?q={query}&sort=new&t=week",
    ]

    POLLING_SOURCES = [
        # FiveThirtyEight polling averages
        "https://projects.fivethirtyeight.com/polls/rss.xml",
    ]

    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.base_researcher = MarketResearcher()

    async def research(self, market: Market) -> ResearchBrief:
        """
        Political research pipeline:
        1. Standard social media + news (base researcher)
        2. Political-specific RSS feeds
        3. Polling data extraction
        4. Enhanced sentiment with political context
        """
        # Run base research for general sources
        brief = await self.base_researcher.research_market(market)

        # Add political-specific sources
        query = self._extract_political_query(market.title)
        political_sources = await self._fetch_political_sources(query)

        # Re-analyze with political context
        for source in political_sources:
            text = f"{source.title} {source.content}"
            score, label = self.sentiment.analyze(text)
            source.sentiment_score = score
            source.sentiment_label = label

        # Merge political sources into the brief
        brief.sources.extend(political_sources)

        # Check for polling data in sources
        poll_signal = self._extract_poll_signal(brief.sources, market.title)
        if poll_signal is not None:
            # Weight polling data heavily — it's the most predictive signal for politics
            brief.narrative_probability = (
                brief.narrative_probability * 0.4 + poll_signal * 0.6
            )
            brief.gap = brief.narrative_probability - brief.market_price

        return brief

    def get_model_prompt_context(self, market: Market, brief: ResearchBrief) -> str:
        """Inject political analysis context into LLM prompts."""
        return (
            "POLITICAL MARKET CONTEXT:\n"
            "- Weight polling data and aggregates heavily (most predictive for elections)\n"
            "- Consider the difference between national polls and state-level polls\n"
            "- For legislation: check committee status, whip counts, procedural hurdles\n"
            "- For court cases: examine precedent, judge history, and legal consensus\n"
            "- Political markets often overreact to single events and mean-revert\n"
            "- Be aware of partisan bias in social media sources\n"
            f"- Sources analyzed: {len(brief.sources)} "
            f"(sentiment: {brief.sentiment_label} {brief.overall_sentiment:+.2f})\n"
        )

    def get_data_sources(self) -> list[str]:
        return ["polls", "political_rss", "reddit_politics", "twitter", "news"]

    def _extract_political_query(self, title: str) -> str:
        """Build a search query optimized for political content."""
        # Extract proper nouns and political terms
        political_terms = re.findall(
            r'\b(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|'
            r'election|senate|congress|vote|bill|court|'
            r'democrat|republican|primary|impeach)\b',
            title, re.IGNORECASE
        )
        if political_terms:
            return " ".join(political_terms[:5])
        return title.split("?")[0].strip()[:50]

    async def _fetch_political_sources(self, query: str) -> list[SourceResult]:
        """Fetch from political-specific RSS feeds."""
        sources = []
        rss = RSSFetcher()
        rss.DEFAULT_FEEDS = self.POLITICAL_RSS_FEEDS

        results = await rss.fetch(query, max_results=10)
        for entry in results:
            sources.append(SourceResult(
                source_type="political_rss",
                title=entry.get("title", ""),
                content=entry.get("summary", "")[:500],
                url=entry.get("link", ""),
            ))

        return sources

    def _extract_poll_signal(self, sources: list[SourceResult], market_title: str) -> float | None:
        """
        Try to extract polling numbers from source content.
        Returns a probability based on polling data, or None if not found.
        """
        # Look for percentage patterns in content
        all_text = " ".join(f"{s.title} {s.content}" for s in sources)

        # Match patterns like "leads 52% to 48%", "polling at 65%", "47% support"
        pct_pattern = re.findall(r'(\d{1,2})(?:\.\d)?%', all_text)
        if not pct_pattern:
            return None

        # Convert to floats, filter reasonable political percentages
        percentages = [float(p) for p in pct_pattern if 20 <= float(p) <= 80]
        if not percentages:
            return None

        # Use the most common percentage range as the signal
        # This is a rough heuristic — a proper implementation would
        # parse specific poll formats more carefully
        avg_pct = sum(percentages) / len(percentages)
        return avg_pct / 100.0
