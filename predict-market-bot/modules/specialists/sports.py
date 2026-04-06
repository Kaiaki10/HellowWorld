"""
Sports Specialist

Tuned for sports prediction markets: game outcomes, player props, championships.

Key data sources:
  - ESPN API (free, scores/schedules/injuries)
  - Odds API (free tier: live odds from 50+ bookmakers)
  - Team/player stats from public APIs
  - Sports news RSS feeds
  - Reddit (r/sportsbook, r/nba, r/nfl, etc.)

Why specialized: Sports markets are driven by injuries, matchups, and
statistical models. A sports specialist can pull real-time injury reports,
compare odds across bookmakers, and use team stats — data that generic
sentiment analysis completely misses.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from modules.scanner import Market
from modules.researcher import (
    ResearchBrief, SourceResult, SentimentAnalyzer,
    RedditScraper, RSSFetcher, MarketResearcher,
)
from modules.specialists import BaseSpecialist

logger = logging.getLogger(__name__)


# Sport-specific subreddits for research
SPORT_SUBREDDITS = {
    "nba": ["nba", "sportsbook"],
    "nfl": ["nfl", "sportsbook"],
    "mlb": ["baseball", "sportsbook"],
    "nhl": ["hockey", "sportsbook"],
    "soccer": ["soccer", "sportsbook"],
    "mma": ["mma", "sportsbook"],
    "tennis": ["tennis", "sportsbook"],
    "golf": ["golf", "sportsbook"],
}

# Team name mappings for fuzzy matching
NFL_TEAMS = [
    "chiefs", "eagles", "bills", "49ers", "cowboys", "ravens", "lions",
    "dolphins", "packers", "bengals", "jets", "chargers", "rams", "seahawks",
    "bears", "steelers", "vikings", "jaguars", "broncos", "commanders",
    "texans", "buccaneers", "saints", "falcons", "colts", "raiders",
    "cardinals", "titans", "panthers", "browns", "giants", "patriots",
]

NBA_TEAMS = [
    "celtics", "nuggets", "bucks", "76ers", "suns", "lakers", "warriors",
    "clippers", "knicks", "heat", "cavaliers", "nets", "mavericks", "grizzlies",
    "kings", "timberwolves", "pelicans", "thunder", "hawks", "raptors",
    "bulls", "trail blazers", "jazz", "pacers", "rockets", "spurs",
    "pistons", "magic", "hornets", "wizards",
]


class SportsSpecialist(BaseSpecialist):

    category = "sports"
    description = "Game outcomes, player props, injuries, and championships"

    SPORTS_RSS_FEEDS = [
        "https://news.google.com/rss/search?q={query}+sports&hl=en-US&gl=US&ceid=US:en",
        "https://www.reddit.com/r/sportsbook/search/.rss?q={query}&sort=new&t=week",
    ]

    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.base_researcher = MarketResearcher()

    async def research(self, market: Market) -> ResearchBrief:
        """
        Sports research pipeline:
        1. Standard social media + news (base researcher)
        2. Injury report check
        3. Odds comparison from bookmakers
        4. Sport-specific Reddit communities
        5. Enhanced analysis with sports context
        """
        # Run base research
        brief = await self.base_researcher.research_market(market)

        sport = self._detect_sport(market.title)
        query = self._extract_sports_query(market.title)

        # Fetch sport-specific sources in parallel
        injury_task = self._fetch_injury_reports(query, sport)
        odds_task = self._fetch_odds(query, sport)
        rss_task = self._fetch_sports_news(query)

        injuries, odds, news_sources = await asyncio.gather(
            injury_task, odds_task, rss_task,
            return_exceptions=True,
        )

        # Add injury reports
        if isinstance(injuries, list):
            for source in injuries:
                score, label = self.sentiment.analyze(f"{source.title} {source.content}")
                source.sentiment_score = score
                source.sentiment_label = label
            brief.sources.extend(injuries)

        # Add odds data
        if isinstance(odds, list):
            brief.sources.extend(odds)
            # Extract odds-implied probability
            odds_prob = self._extract_odds_probability(odds, market.title)
            if odds_prob is not None:
                # Weight bookmaker odds heavily — they aggregate massive info
                brief.narrative_probability = (
                    brief.narrative_probability * 0.35 + odds_prob * 0.65
                )
                brief.gap = brief.narrative_probability - brief.market_price
                brief.confidence = min(brief.confidence + 0.15, 0.95)

        # Add news
        if isinstance(news_sources, list):
            for source in news_sources:
                score, label = self.sentiment.analyze(f"{source.title} {source.content}")
                source.sentiment_score = score
                source.sentiment_label = label
            brief.sources.extend(news_sources)

        return brief

    def get_model_prompt_context(self, market: Market, brief: ResearchBrief) -> str:
        return (
            "SPORTS MARKET CONTEXT:\n"
            "- Bookmaker odds are extremely efficient — weight them heavily (60%+)\n"
            "- Injury reports can swing probabilities 10-20% for key players\n"
            "- Home/away advantage matters: ~57% home win rate in NFL, ~60% in NBA\n"
            "- Recent form (last 5-10 games) is more predictive than season averages\n"
            "- Playoff/tournament context changes team behavior significantly\n"
            "- Weather affects outdoor sports (NFL, MLB, soccer)\n"
            "- Back-to-back games in NBA cause significant fatigue effects\n"
            f"- Sources: {len(brief.sources)} | Confidence: {brief.confidence:.0%}\n"
        )

    def get_data_sources(self) -> list[str]:
        return ["odds_api", "espn_injuries", "sports_rss", "reddit_sportsbook"]

    def _detect_sport(self, title: str) -> str:
        """Detect which sport a market is about."""
        title_lower = title.lower()

        sport_keywords = {
            "nba": ["nba", "basketball"] + NBA_TEAMS,
            "nfl": ["nfl", "football", "super bowl"] + NFL_TEAMS,
            "mlb": ["mlb", "baseball", "world series"],
            "nhl": ["nhl", "hockey", "stanley cup"],
            "soccer": ["soccer", "premier league", "champions league", "world cup", "mls", "fifa"],
            "mma": ["ufc", "mma", "boxing", "fight"],
            "tennis": ["tennis", "wimbledon", "us open", "roland garros", "australian open"],
            "golf": ["golf", "pga", "masters", "open championship"],
            "f1": ["f1", "formula 1", "grand prix", "racing"],
        }

        best_sport = "general"
        best_count = 0
        for sport, keywords in sport_keywords.items():
            count = sum(1 for kw in keywords if kw in title_lower)
            if count > best_count:
                best_count = count
                best_sport = sport

        return best_sport

    def _extract_sports_query(self, title: str) -> str:
        """Build search query for sports content."""
        # Extract team names, player names (capitalized words), and sport terms
        terms = re.findall(
            r'\b(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|'
            r'NBA|NFL|MLB|NHL|UFC|F1|PGA|MLS)\b',
            title,
        )
        if terms:
            return " ".join(terms[:5])
        return title.split("?")[0].strip()[:50]

    async def _fetch_injury_reports(self, query: str, sport: str) -> list[SourceResult]:
        """Fetch injury reports from ESPN API."""
        sources = []
        try:
            async with aiohttp.ClientSession() as session:
                # ESPN public API for news/injuries
                sport_map = {
                    "nba": "basketball/nba",
                    "nfl": "football/nfl",
                    "mlb": "baseball/mlb",
                    "nhl": "hockey/nhl",
                    "soccer": "soccer/eng.1",
                }
                espn_sport = sport_map.get(sport, "")
                if not espn_sport:
                    return sources

                url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_sport}/news"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        articles = data.get("articles", [])
                        for article in articles[:5]:
                            headline = article.get("headline", "")
                            description = article.get("description", "")
                            # Check if relevant to our query
                            query_words = set(query.lower().split())
                            article_text = f"{headline} {description}".lower()
                            if any(w in article_text for w in query_words):
                                sources.append(SourceResult(
                                    source_type="espn_news",
                                    title=headline,
                                    content=description[:500],
                                    url=article.get("links", {}).get("web", {}).get("href", ""),
                                    sentiment_score=0,
                                    sentiment_label="neutral",
                                ))
        except Exception as e:
            logger.debug(f"ESPN fetch error: {e}")
        return sources

    async def _fetch_odds(self, query: str, sport: str) -> list[SourceResult]:
        """Fetch odds from The Odds API (free tier: 500 requests/month)."""
        sources = []
        try:
            from config import get_env
            api_key = get_env("ODDS_API_KEY", "")
            if not api_key:
                return sources

            sport_map = {
                "nba": "basketball_nba",
                "nfl": "americanfootball_nfl",
                "mlb": "baseball_mlb",
                "nhl": "icehockey_nhl",
                "soccer": "soccer_epl",
                "mma": "mma_mixed_martial_arts",
            }
            odds_sport = sport_map.get(sport, "")
            if not odds_sport:
                return sources

            async with aiohttp.ClientSession() as session:
                url = f"https://api.the-odds-api.com/v4/sports/{odds_sport}/odds"
                params = {
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                }
                async with session.get(url, params=params,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for event in data[:3]:
                            home = event.get("home_team", "")
                            away = event.get("away_team", "")
                            bookmakers = event.get("bookmakers", [])
                            if bookmakers:
                                # Average odds across bookmakers
                                all_odds = []
                                for bm in bookmakers[:5]:
                                    for market in bm.get("markets", []):
                                        if market.get("key") == "h2h":
                                            for outcome in market.get("outcomes", []):
                                                all_odds.append({
                                                    "team": outcome["name"],
                                                    "odds": outcome["price"],
                                                    "bookmaker": bm["title"],
                                                })
                                sources.append(SourceResult(
                                    source_type="odds_api",
                                    title=f"{home} vs {away} odds",
                                    content=f"Odds data: {all_odds[:10]}",
                                    url="https://the-odds-api.com",
                                    sentiment_score=0,
                                    sentiment_label="neutral",
                                ))
        except Exception as e:
            logger.debug(f"Odds API fetch error: {e}")
        return sources

    async def _fetch_sports_news(self, query: str) -> list[SourceResult]:
        """Fetch sports news from RSS feeds."""
        sources = []
        rss = RSSFetcher()
        rss.DEFAULT_FEEDS = self.SPORTS_RSS_FEEDS
        results = await rss.fetch(query, max_results=8)
        for entry in results:
            sources.append(SourceResult(
                source_type="sports_rss",
                title=entry.get("title", ""),
                content=entry.get("summary", "")[:500],
                url=entry.get("link", ""),
                sentiment_score=0,
                sentiment_label="neutral",
            ))
        return sources

    def _extract_odds_probability(self, odds_sources: list[SourceResult],
                                    market_title: str) -> Optional[float]:
        """
        Extract implied probability from odds data.
        Decimal odds → implied prob = 1 / odds
        """
        for source in odds_sources:
            if source.source_type != "odds_api":
                continue
            # Parse odds from content
            odds_match = re.findall(r"'odds':\s*([\d.]+)", source.content)
            if len(odds_match) >= 2:
                try:
                    odds_values = [float(o) for o in odds_match[:2]]
                    # Implied probabilities (with overround)
                    probs = [1.0 / o for o in odds_values if o > 0]
                    if probs:
                        # Normalize to remove overround
                        total = sum(probs)
                        normalized = [p / total for p in probs]
                        # Return the first team's probability
                        # (rough heuristic — production would match team names)
                        return normalized[0]
                except (ValueError, ZeroDivisionError):
                    continue
        return None
