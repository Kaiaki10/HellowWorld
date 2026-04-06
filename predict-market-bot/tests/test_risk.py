"""Tests for risk validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_risk import validate_risk


def test_all_checks_pass():
    """Normal trade should pass all checks."""
    result = validate_risk(
        p_model=0.65,
        p_market=0.49,
        position_size_usd=300,
        bankroll=10000,
        current_exposure_usd=1000,
        daily_pnl_usd=-50,
        current_drawdown_pct=0.02,
        open_positions_count=3,
        daily_api_cost_usd=10,
    )
    assert result.all_passed, f"Should pass all checks: {result.summary()}"


def test_edge_too_small():
    """Edge below 4% should fail."""
    result = validate_risk(
        p_model=0.52,
        p_market=0.50,
        position_size_usd=300,
        bankroll=10000,
        current_exposure_usd=0,
        daily_pnl_usd=0,
        current_drawdown_pct=0,
        open_positions_count=0,
        daily_api_cost_usd=0,
    )
    assert not result.all_passed
    failed = [c.check_name for c in result.checks if not c.passed]
    assert "Edge Check" in failed


def test_position_too_large():
    """Position > 5% of bankroll should fail."""
    result = validate_risk(
        p_model=0.70,
        p_market=0.50,
        position_size_usd=600,  # 6% of $10k
        bankroll=10000,
        current_exposure_usd=0,
        daily_pnl_usd=0,
        current_drawdown_pct=0,
        open_positions_count=0,
        daily_api_cost_usd=0,
    )
    assert not result.all_passed
    failed = [c.check_name for c in result.checks if not c.passed]
    assert "Position Size" in failed


def test_max_drawdown_blocks():
    """Drawdown > 8% should block ALL trades."""
    result = validate_risk(
        p_model=0.80,
        p_market=0.50,
        position_size_usd=100,
        bankroll=10000,
        current_exposure_usd=0,
        daily_pnl_usd=0,
        current_drawdown_pct=0.09,  # 9% > 8% limit
        open_positions_count=0,
        daily_api_cost_usd=0,
    )
    assert not result.all_passed
    failed = [c.check_name for c in result.checks if not c.passed]
    assert "Max Drawdown" in failed


def test_daily_loss_limit():
    """Daily loss exceeding limit should block."""
    result = validate_risk(
        p_model=0.70,
        p_market=0.50,
        position_size_usd=100,
        bankroll=10000,
        current_exposure_usd=0,
        daily_pnl_usd=-1600,  # -$1600 > 15% of $10k = $1500
        current_drawdown_pct=0.02,
        open_positions_count=0,
        daily_api_cost_usd=0,
    )
    assert not result.all_passed
    failed = [c.check_name for c in result.checks if not c.passed]
    assert "Daily Loss Limit" in failed


def test_too_many_positions():
    """Exceeding max concurrent positions should block."""
    result = validate_risk(
        p_model=0.70,
        p_market=0.50,
        position_size_usd=100,
        bankroll=10000,
        current_exposure_usd=0,
        daily_pnl_usd=0,
        current_drawdown_pct=0,
        open_positions_count=15,  # At the max
        daily_api_cost_usd=0,
    )
    assert not result.all_passed
    failed = [c.check_name for c in result.checks if not c.passed]
    assert "Concurrent Positions" in failed


def test_api_cost_exceeded():
    """API cost over limit should block."""
    result = validate_risk(
        p_model=0.70,
        p_market=0.50,
        position_size_usd=100,
        bankroll=10000,
        current_exposure_usd=0,
        daily_pnl_usd=0,
        current_drawdown_pct=0,
        open_positions_count=0,
        daily_api_cost_usd=55,  # > $50 limit
    )
    assert not result.all_passed
    failed = [c.check_name for c in result.checks if not c.passed]
    assert "Daily API Cost" in failed


def test_exposure_check():
    """Total exposure exceeding limit should block."""
    result = validate_risk(
        p_model=0.70,
        p_market=0.50,
        position_size_usd=400,
        bankroll=10000,
        current_exposure_usd=4700,  # 4700 + 400 = 5100 > 50% of 10k = 5000
        daily_pnl_usd=0,
        current_drawdown_pct=0,
        open_positions_count=3,
        daily_api_cost_usd=0,
    )
    assert not result.all_passed
    failed = [c.check_name for c in result.checks if not c.passed]
    assert "Exposure Check" in failed


def test_multiple_failures():
    """Multiple failures should all be reported."""
    result = validate_risk(
        p_model=0.51,       # Edge too small
        p_market=0.50,
        position_size_usd=600,  # Position too large
        bankroll=10000,
        current_exposure_usd=0,
        daily_pnl_usd=0,
        current_drawdown_pct=0.09,  # Drawdown too high
        open_positions_count=0,
        daily_api_cost_usd=0,
    )
    assert not result.all_passed
    failed = [c.check_name for c in result.checks if not c.passed]
    assert len(failed) >= 3, f"Expected 3+ failures, got {failed}"


if __name__ == "__main__":
    tests = [
        test_all_checks_pass,
        test_edge_too_small,
        test_position_too_large,
        test_max_drawdown_blocks,
        test_daily_loss_limit,
        test_too_many_positions,
        test_api_cost_exceeded,
        test_exposure_check,
        test_multiple_failures,
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
