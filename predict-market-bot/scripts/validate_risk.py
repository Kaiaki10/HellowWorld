"""
Deterministic Risk Validation Script

All risk checks must pass before any trade is executed.
This is a Python script (not LLM instructions) to ensure deterministic behavior.
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class RiskCheckResult:
    passed: bool
    check_name: str
    details: str
    value: Optional[float] = None
    limit: Optional[float] = None


@dataclass
class RiskValidation:
    all_passed: bool
    checks: list[RiskCheckResult]
    timestamp: str

    def summary(self) -> str:
        status = "PASSED" if self.all_passed else "BLOCKED"
        lines = [f"Risk Validation: {status}"]
        for c in self.checks:
            icon = "[OK]" if c.passed else "[FAIL]"
            lines.append(f"  {icon} {c.check_name}: {c.details}")
        return "\n".join(lines)


def validate_risk(
    p_model: float,
    p_market: float,
    position_size_usd: float,
    bankroll: float,
    current_exposure_usd: float,
    daily_pnl_usd: float,
    current_drawdown_pct: float,
    open_positions_count: int,
    daily_api_cost_usd: float,
    # Configurable limits
    min_edge: float = 0.04,
    max_position_pct: float = 0.05,
    max_total_exposure_pct: float = 0.50,
    var_limit_usd: Optional[float] = None,
    max_drawdown_pct: float = 0.08,
    max_daily_loss_pct: float = 0.15,
    max_concurrent_positions: int = 15,
    max_daily_api_cost: float = 50.0,
) -> RiskValidation:
    """
    Run all risk checks. ALL must pass for trade execution.

    Returns RiskValidation with detailed results.
    """
    checks: list[RiskCheckResult] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Edge check: p_model - p_market must be > min_edge
    edge = abs(p_model - p_market)
    checks.append(RiskCheckResult(
        passed=edge > min_edge,
        check_name="Edge Check",
        details=f"Edge {edge:.4f} {'>' if edge > min_edge else '<='} min {min_edge}",
        value=edge,
        limit=min_edge,
    ))

    # 2. Position size check: must not exceed max % of bankroll
    max_position = bankroll * max_position_pct
    checks.append(RiskCheckResult(
        passed=position_size_usd <= max_position,
        check_name="Position Size",
        details=f"${position_size_usd:.2f} {'<=' if position_size_usd <= max_position else '>'} "
                f"max ${max_position:.2f} ({max_position_pct:.0%} of bankroll)",
        value=position_size_usd,
        limit=max_position,
    ))

    # 3. Exposure check: new bet + existing must not exceed max total exposure
    max_exposure = bankroll * max_total_exposure_pct
    new_total_exposure = current_exposure_usd + position_size_usd
    checks.append(RiskCheckResult(
        passed=new_total_exposure <= max_exposure,
        check_name="Exposure Check",
        details=f"Total exposure ${new_total_exposure:.2f} {'<=' if new_total_exposure <= max_exposure else '>'} "
                f"max ${max_exposure:.2f}",
        value=new_total_exposure,
        limit=max_exposure,
    ))

    # 4. VaR check (simplified): 95% VaR should be within daily limit
    # Simplified VaR: assume worst case is losing entire position with 5% probability
    if var_limit_usd is not None:
        simple_var = position_size_usd * 0.5  # Assume 50% loss potential at 95% confidence
        checks.append(RiskCheckResult(
            passed=simple_var <= var_limit_usd,
            check_name="VaR Check (95%)",
            details=f"VaR ${simple_var:.2f} {'<=' if simple_var <= var_limit_usd else '>'} "
                    f"limit ${var_limit_usd:.2f}",
            value=simple_var,
            limit=var_limit_usd,
        ))

    # 5. Max drawdown check: if drawdown exceeds limit, block ALL trades
    checks.append(RiskCheckResult(
        passed=current_drawdown_pct < max_drawdown_pct,
        check_name="Max Drawdown",
        details=f"Drawdown {current_drawdown_pct:.1%} {'<' if current_drawdown_pct < max_drawdown_pct else '>='} "
                f"limit {max_drawdown_pct:.0%}",
        value=current_drawdown_pct,
        limit=max_drawdown_pct,
    ))

    # 6. Daily loss limit
    daily_loss_limit = bankroll * max_daily_loss_pct
    daily_loss = abs(min(daily_pnl_usd, 0))
    checks.append(RiskCheckResult(
        passed=daily_loss < daily_loss_limit,
        check_name="Daily Loss Limit",
        details=f"Daily loss ${daily_loss:.2f} {'<' if daily_loss < daily_loss_limit else '>='} "
                f"limit ${daily_loss_limit:.2f}",
        value=daily_loss,
        limit=daily_loss_limit,
    ))

    # 7. Concurrent positions check
    checks.append(RiskCheckResult(
        passed=open_positions_count < max_concurrent_positions,
        check_name="Concurrent Positions",
        details=f"{open_positions_count} open {'<' if open_positions_count < max_concurrent_positions else '>='} "
                f"max {max_concurrent_positions}",
        value=float(open_positions_count),
        limit=float(max_concurrent_positions),
    ))

    # 8. API cost check
    checks.append(RiskCheckResult(
        passed=daily_api_cost_usd < max_daily_api_cost,
        check_name="Daily API Cost",
        details=f"${daily_api_cost_usd:.2f} {'<' if daily_api_cost_usd < max_daily_api_cost else '>='} "
                f"limit ${max_daily_api_cost:.2f}",
        value=daily_api_cost_usd,
        limit=max_daily_api_cost,
    ))

    all_passed = all(c.passed for c in checks)

    return RiskValidation(
        all_passed=all_passed,
        checks=checks,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    # Example validation
    result = validate_risk(
        p_model=0.65,
        p_market=0.49,
        position_size_usd=300,
        bankroll=10000,
        current_exposure_usd=1200,
        daily_pnl_usd=-50,
        current_drawdown_pct=0.02,
        open_positions_count=5,
        daily_api_cost_usd=12.50,
    )
    print(result.summary())

    # Also support JSON input from stdin for pipeline integration
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        data = json.loads(sys.stdin.read())
        result = validate_risk(**data)
        print(json.dumps({
            "all_passed": result.all_passed,
            "checks": [
                {
                    "passed": c.passed,
                    "check_name": c.check_name,
                    "details": c.details,
                    "value": c.value,
                    "limit": c.limit,
                }
                for c in result.checks
            ],
            "timestamp": result.timestamp,
        }, indent=2))
