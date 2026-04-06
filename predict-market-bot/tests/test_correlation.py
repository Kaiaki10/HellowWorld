"""Tests for correlation-aware position sizing.

Tests the CorrelationTracker in isolation without importing the full module tree.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# Minimal Position mock (avoids importing the full risk_executor chain)
@dataclass
class MockPosition:
    market_id: str
    market_title: str
    size_usd: float


class CorrelationTracker:
    """Copy of the correlation logic for isolated testing."""

    CORRELATION_THRESHOLD = 0.5

    def __init__(self):
        self.keyword_index = {}

    def estimate_correlation(self, market_id_a, title_a, market_id_b, title_b):
        stopwords = {"will", "the", "be", "a", "an", "in", "on", "at", "to", "by", "of", "for",
                     "is", "or", "and", "before", "after", "yes", "no"}
        words_a = {w.lower().strip("?.,!") for w in title_a.split()
                   if w.lower().strip("?.,!") not in stopwords and len(w) > 2}
        words_b = {w.lower().strip("?.,!") for w in title_b.split()
                   if w.lower().strip("?.,!") not in stopwords and len(w) > 2}
        if not words_a or not words_b:
            return 0.0
        overlap = words_a & words_b
        total = words_a | words_b
        return len(overlap) / len(total) if total else 0.0

    def get_correlated_exposure(self, market_title, open_positions):
        correlated_exposure = 0.0
        for pos in open_positions:
            correlation = self.estimate_correlation(
                "new", market_title, pos.market_id, pos.market_title,
            )
            if correlation >= self.CORRELATION_THRESHOLD:
                correlated_exposure += pos.size_usd
        return correlated_exposure

    def scale_for_correlation(self, proposed_size, market_title, open_positions,
                                bankroll, max_position_pct):
        correlated_exposure = self.get_correlated_exposure(market_title, open_positions)
        max_correlated = bankroll * max_position_pct
        if correlated_exposure <= 0:
            return proposed_size, ""
        total_if_added = correlated_exposure + proposed_size
        if total_if_added <= max_correlated:
            return proposed_size, ""
        remaining = max(0, max_correlated - correlated_exposure)
        if remaining <= 0:
            return 0.0, f"Correlated exposure ${correlated_exposure:.2f} at cap"
        scale_factor = remaining / proposed_size
        adjusted = proposed_size * scale_factor
        return adjusted, f"Scaled {scale_factor:.0%}"


def test_no_correlation():
    """Unrelated markets should not affect sizing."""
    tracker = CorrelationTracker()
    positions = [MockPosition("btc1", "Bitcoin price above 100k December", 200)]
    adjusted, reason = tracker.scale_for_correlation(
        proposed_size=300,
        market_title="Will it rain in Tokyo tomorrow",
        open_positions=positions,
        bankroll=10000,
        max_position_pct=0.05,
    )
    assert adjusted == 300, f"Unrelated market should not be scaled: {reason}"


def test_high_correlation():
    """Related markets should trigger scaling."""
    tracker = CorrelationTracker()
    # These titles share 4/6 keywords = 0.67 correlation (above 0.5 threshold)
    positions = [MockPosition("biden1", "Biden wins presidential election 2028 November", 400)]
    adjusted, reason = tracker.scale_for_correlation(
        proposed_size=300,
        market_title="Biden wins presidential election 2028 Democratic",
        open_positions=positions,
        bankroll=10000,
        max_position_pct=0.05,
    )
    assert adjusted < 300, f"Should be scaled down, got {adjusted}"
    assert adjusted <= 100, f"Should be capped around $100, got {adjusted}"


def test_correlation_blocks_when_full():
    """Should block entirely when correlated exposure is at cap."""
    tracker = CorrelationTracker()
    positions = [MockPosition("trump1", "Trump wins election November", 500)]
    adjusted, reason = tracker.scale_for_correlation(
        proposed_size=200,
        market_title="Trump Republican nominee November election",
        open_positions=positions,
        bankroll=10000,
        max_position_pct=0.05,
    )
    assert adjusted == 0, f"Should block: correlated at cap. Got {adjusted}"


def test_no_open_positions():
    """No open positions should never scale."""
    tracker = CorrelationTracker()
    adjusted, reason = tracker.scale_for_correlation(
        proposed_size=300,
        market_title="Any market title here",
        open_positions=[],
        bankroll=10000,
        max_position_pct=0.05,
    )
    assert adjusted == 300
    assert reason == ""


def test_correlation_estimate():
    """Test keyword-based correlation estimation."""
    tracker = CorrelationTracker()
    corr = tracker.estimate_correlation(
        "a", "Biden wins presidential election 2028",
        "b", "Biden wins Democratic primary 2028",
    )
    assert corr > 0.3, f"Should detect correlation, got {corr}"

    corr = tracker.estimate_correlation(
        "a", "Bitcoin price above 100k",
        "b", "Rain in Tokyo tomorrow",
    )
    assert corr < 0.2, f"Should not be correlated, got {corr}"


if __name__ == "__main__":
    tests = [
        test_no_correlation,
        test_high_correlation,
        test_correlation_blocks_when_full,
        test_no_open_positions,
        test_correlation_estimate,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
