"""Tests for Kelly Criterion position sizing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.kelly_size import kelly_criterion


def test_basic_kelly():
    """70% win prob, 2:1 odds, $10k bankroll."""
    result = kelly_criterion(0.70, 0.333, 10000, kelly_fraction=1.0)
    assert result.full_kelly_fraction > 0, "Should recommend a bet"
    assert result.position_size_usd > 0
    assert result.position_size_usd <= 10000


def test_quarter_kelly_smaller():
    """Quarter-Kelly should be ~25% of full Kelly."""
    # Use high max_position_pct to avoid capping
    full = kelly_criterion(0.70, 0.333, 10000, kelly_fraction=1.0, max_position_pct=0.50)
    quarter = kelly_criterion(0.70, 0.333, 10000, kelly_fraction=0.25, max_position_pct=0.50)
    assert quarter.position_size_usd < full.position_size_usd
    ratio = quarter.position_size_usd / full.position_size_usd
    assert 0.20 <= ratio <= 0.30, f"Expected ~0.25 ratio, got {ratio}"


def test_no_edge_no_bet():
    """When win prob equals market price, Kelly says don't bet."""
    result = kelly_criterion(0.50, 0.50, 10000)
    assert result.position_size_usd == 0, "No edge should mean no bet"


def test_negative_edge():
    """When model is worse than market, Kelly is negative (no bet)."""
    result = kelly_criterion(0.30, 0.50, 10000)
    assert result.full_kelly_fraction < 0
    assert result.position_size_usd == 0


def test_position_cap():
    """Position should be capped at max_position_pct of bankroll."""
    result = kelly_criterion(0.95, 0.10, 10000, kelly_fraction=1.0, max_position_pct=0.05)
    assert result.position_size_usd <= 500, f"Should cap at $500, got {result.position_size_usd}"
    assert result.capped


def test_zero_bankroll():
    """Zero bankroll should produce zero position."""
    result = kelly_criterion(0.70, 0.333, 0)
    assert result.position_size_usd == 0


def test_extreme_probability():
    """Edge cases with extreme probabilities."""
    # 99% sure, very cheap market
    result = kelly_criterion(0.99, 0.10, 10000, kelly_fraction=0.25)
    assert result.position_size_usd > 0

    # Invalid probabilities
    result = kelly_criterion(0.0, 0.50, 10000)
    assert result.position_size_usd == 0

    result = kelly_criterion(1.0, 0.50, 10000)
    assert result.position_size_usd == 0


def test_invalid_market_price():
    """Invalid market prices should return zero."""
    result = kelly_criterion(0.70, 0.0, 10000)
    assert result.position_size_usd == 0

    result = kelly_criterion(0.70, 1.0, 10000)
    assert result.position_size_usd == 0


if __name__ == "__main__":
    tests = [
        test_basic_kelly,
        test_quarter_kelly_smaller,
        test_no_edge_no_bet,
        test_negative_edge,
        test_position_cap,
        test_zero_bankroll,
        test_extreme_probability,
        test_invalid_market_price,
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
