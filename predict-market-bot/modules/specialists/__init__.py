"""
Specialist Bot Framework

Auto-routes markets to specialized research + prediction pipelines
based on market category. Each specialist has tuned data sources,
models, and prediction strategies for its vertical.

Architecture:
    Market → Classifier → Specialist Router
                              ├── PoliticsSpecialist  (polls, legislation, LLM reasoning)
                              ├── WeatherSpecialist   (NOAA, GFS/ECMWF models)
                              ├── SportsSpecialist    (injury reports, stats, odds)
                              ├── CryptoSpecialist    (on-chain data, exchange flows)
                              └── GeneralSpecialist   (fallback: standard pipeline)
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from modules.scanner import Market
from modules.researcher import ResearchBrief

logger = logging.getLogger(__name__)


# ─── Market Categories ───

CATEGORIES = {
    "politics": {
        "keywords": [
            "president", "election", "senate", "congress", "vote", "poll",
            "democrat", "republican", "trump", "biden", "governor", "mayor",
            "legislation", "impeach", "supreme court", "nominee", "primary",
            "electoral", "ballot", "caucus", "midterm", "cabinet", "veto",
            "party", "liberal", "conservative", "indictment", "trial",
        ],
        "weight": 1.0,
    },
    "weather": {
        "keywords": [
            "temperature", "rain", "snow", "hurricane", "tornado", "storm",
            "weather", "forecast", "celsius", "fahrenheit", "wind", "flood",
            "drought", "heat wave", "cold", "precipitation", "climate",
            "noaa", "nws", "typhoon", "blizzard", "wildfire",
        ],
        "weight": 1.0,
    },
    "sports": {
        "keywords": [
            "game", "match", "score", "win", "championship", "playoff",
            "nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball",
            "baseball", "tennis", "golf", "ufc", "boxing", "f1", "racing",
            "super bowl", "world series", "finals", "mvp", "draft",
            "team", "player", "coach", "injury", "season", "league",
        ],
        "weight": 1.0,
    },
    "crypto": {
        "keywords": [
            "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
            "token", "defi", "nft", "solana", "sol", "altcoin", "mining",
            "halving", "whale", "exchange", "binance", "coinbase",
            "stablecoin", "usdt", "usdc", "market cap", "memecoin",
        ],
        "weight": 1.0,
    },
    "economics": {
        "keywords": [
            "gdp", "inflation", "interest rate", "fed", "federal reserve",
            "unemployment", "cpi", "jobs report", "recession", "tariff",
            "trade", "stock", "s&p", "nasdaq", "dow", "treasury", "yield",
            "housing", "oil", "commodity", "central bank",
        ],
        "weight": 0.8,
    },
}


@dataclass
class ClassificationResult:
    """Result of market category classification."""
    market_id: str
    market_title: str
    category: str          # "politics", "weather", "sports", "crypto", "economics", "general"
    confidence: float      # 0-1
    keyword_matches: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"[{self.category.upper():10s}] {self.confidence:.0%} | {self.market_title[:50]}"


class MarketClassifier:
    """
    Classifies markets into categories based on keyword matching.
    Used to route markets to the appropriate specialist.
    """

    def __init__(self):
        self.categories = CATEGORIES
        # Pre-compile patterns for faster matching
        self._patterns: dict[str, list[re.Pattern]] = {}
        for cat, config in self.categories.items():
            self._patterns[cat] = [
                re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
                for kw in config["keywords"]
            ]

    def classify(self, market: Market) -> ClassificationResult:
        """Classify a market into a category."""
        title = market.title.lower()
        scores: dict[str, float] = {}
        matches: dict[str, list[str]] = {}

        for cat, patterns in self._patterns.items():
            cat_matches = []
            for pattern in patterns:
                if pattern.search(title):
                    cat_matches.append(pattern.pattern.strip(r'\b'))
            weight = self.categories[cat]["weight"]
            scores[cat] = len(cat_matches) * weight
            matches[cat] = cat_matches

        if not any(scores.values()):
            return ClassificationResult(
                market_id=market.market_id,
                market_title=market.title,
                category="general",
                confidence=0.5,
            )

        best_cat = max(scores, key=scores.get)
        best_score = scores[best_cat]

        # Confidence based on number of keyword matches
        confidence = min(best_score / 3, 1.0)  # 3+ matches = 100% confidence

        return ClassificationResult(
            market_id=market.market_id,
            market_title=market.title,
            category=best_cat,
            confidence=confidence,
            keyword_matches=matches[best_cat],
        )

    def classify_batch(self, markets: list[Market]) -> dict[str, list[Market]]:
        """Classify multiple markets and group by category."""
        groups: dict[str, list[Market]] = {
            cat: [] for cat in list(self.categories.keys()) + ["general"]
        }

        for market in markets:
            result = self.classify(market)
            groups[result.category].append(market)
            logger.debug(f"Classified: {result.summary()}")

        # Log summary
        for cat, mkts in groups.items():
            if mkts:
                logger.info(f"  {cat}: {len(mkts)} markets")

        return groups


class BaseSpecialist(ABC):
    """
    Base class for market vertical specialists.
    Each specialist implements custom research and prediction logic
    tuned for its market category.
    """

    category: str = "general"
    description: str = ""

    @abstractmethod
    async def research(self, market: Market) -> ResearchBrief:
        """Run specialized research for this market category."""
        pass

    @abstractmethod
    def get_model_prompt_context(self, market: Market, brief: ResearchBrief) -> str:
        """
        Return additional context to inject into LLM prediction prompts.
        This is category-specific knowledge that helps the LLM make better predictions.
        """
        pass

    @abstractmethod
    def get_data_sources(self) -> list[str]:
        """Return the list of data sources this specialist uses."""
        pass


class SpecialistRegistry:
    """
    Registry of all available specialists.
    Routes markets to the correct specialist based on classification.
    """

    def __init__(self):
        self.specialists: dict[str, BaseSpecialist] = {}
        self.classifier = MarketClassifier()

    def register(self, specialist: BaseSpecialist):
        self.specialists[specialist.category] = specialist
        logger.info(f"Registered specialist: {specialist.category} - {specialist.description}")

    def get(self, category: str) -> Optional[BaseSpecialist]:
        return self.specialists.get(category)

    def route(self, market: Market) -> tuple[BaseSpecialist | None, ClassificationResult]:
        """Classify a market and return the appropriate specialist."""
        result = self.classifier.classify(market)
        specialist = self.specialists.get(result.category)
        return specialist, result

    def list_specialists(self) -> str:
        lines = ["Registered Specialists:"]
        for cat, spec in self.specialists.items():
            sources = ", ".join(spec.get_data_sources())
            lines.append(f"  [{cat}] {spec.description} | Sources: {sources}")
        return "\n".join(lines)
