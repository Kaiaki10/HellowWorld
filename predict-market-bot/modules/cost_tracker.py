"""
API Cost Tracker

Tracks actual API spending from response headers/token counts.
Enforces the daily budget cap and provides cost reporting.
"""

import json
import logging
from datetime import datetime, timezone, date
from pathlib import Path

from config import get_setting

logger = logging.getLogger(__name__)

DATA_DIR = Path(get_setting("general", "data_dir") or "data")

# Cost per 1M tokens (approximate, as of early 2026)
TOKEN_COSTS = {
    # Anthropic
    "claude-opus-4-20250514":      {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514":    {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001":   {"input": 0.80,  "output": 4.00},
    # OpenAI
    "gpt-4o":                      {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":                 {"input": 0.15,  "output": 0.60},
    # Google
    "gemini-pro":                  {"input": 0.50,  "output": 1.50},
}


class CostTracker:
    """Tracks API costs across all LLM providers."""

    def __init__(self):
        self.daily_cost = 0.0
        self.daily_limit = get_setting("risk", "max_daily_api_cost_usd") or 50.0
        self.today = date.today()
        self.log: list[dict] = []
        self.cost_file = DATA_DIR / "api_costs.json"
        self._load()

    def record_call(self, model: str, input_tokens: int, output_tokens: int,
                     source: str = "") -> float:
        """
        Record an API call and return its cost.

        Args:
            model: Model ID (e.g., "claude-opus-4-20250514")
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            source: Which module made the call (e.g., "predictor", "researcher")

        Returns:
            Cost of this call in USD
        """
        self._check_day_rollover()

        costs = TOKEN_COSTS.get(model, {"input": 5.0, "output": 15.0})
        cost = (input_tokens / 1_000_000 * costs["input"] +
                output_tokens / 1_000_000 * costs["output"])

        self.daily_cost += cost

        entry = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": round(cost, 6),
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.log.append(entry)
        self._save()

        if self.daily_cost >= self.daily_limit * 0.8:
            remaining = self.daily_limit - self.daily_cost
            logger.warning(f"API budget alert: ${self.daily_cost:.2f}/${self.daily_limit:.2f} "
                          f"(${remaining:.2f} remaining)")

        return cost

    def estimate_call_cost(self, model: str, input_tokens: int = 1000,
                            output_tokens: int = 300) -> float:
        """Estimate cost of a call before making it."""
        costs = TOKEN_COSTS.get(model, {"input": 5.0, "output": 15.0})
        return (input_tokens / 1_000_000 * costs["input"] +
                output_tokens / 1_000_000 * costs["output"])

    def can_afford(self, estimated_cost: float = 0.05) -> bool:
        """Check if we're within daily budget."""
        self._check_day_rollover()
        return (self.daily_cost + estimated_cost) < self.daily_limit

    def budget_remaining(self) -> float:
        self._check_day_rollover()
        return max(0, self.daily_limit - self.daily_cost)

    def _check_day_rollover(self):
        """Reset counters at midnight."""
        today = date.today()
        if today != self.today:
            logger.info(f"Day rollover: yesterday's API cost was ${self.daily_cost:.2f}")
            self.daily_cost = 0.0
            self.today = today
            self.log = []

    def report(self) -> str:
        self._check_day_rollover()
        by_model: dict[str, float] = {}
        by_source: dict[str, float] = {}
        for entry in self.log:
            model = entry.get("model", "unknown")
            source = entry.get("source", "unknown")
            cost = entry.get("cost", 0)
            by_model[model] = by_model.get(model, 0) + cost
            by_source[source] = by_source.get(source, 0) + cost

        lines = [
            f"=== API Cost Report (today) ===",
            f"  Total: ${self.daily_cost:.4f} / ${self.daily_limit:.2f} limit",
            f"  Calls: {len(self.log)}",
            f"  Remaining: ${self.budget_remaining():.2f}",
        ]
        if by_model:
            lines.append("  By model:")
            for model, cost in sorted(by_model.items(), key=lambda x: -x[1]):
                lines.append(f"    {model}: ${cost:.4f}")
        if by_source:
            lines.append("  By source:")
            for source, cost in sorted(by_source.items(), key=lambda x: -x[1]):
                lines.append(f"    {source}: ${cost:.4f}")
        return "\n".join(lines)

    def _save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "daily_cost": self.daily_cost,
            "daily_limit": self.daily_limit,
            "date": str(self.today),
            "calls": len(self.log),
        }
        self.cost_file.write_text(json.dumps(state, indent=2))

    def _load(self):
        if self.cost_file.exists():
            try:
                state = json.loads(self.cost_file.read_text())
                saved_date = state.get("date", "")
                if saved_date == str(date.today()):
                    self.daily_cost = state.get("daily_cost", 0.0)
            except (json.JSONDecodeError, KeyError):
                pass


# Global singleton
_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker
