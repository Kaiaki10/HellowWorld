"""
Kelly Criterion Position Sizing Calculator

f* = (p * b - q) / b
Where:
    p = win probability (your model estimate)
    q = 1 - p (loss probability)
    b = net odds (decimal odds minus 1)

Uses Fractional Kelly (0.25-0.50) to reduce variance.
"""

import sys
from dataclasses import dataclass


@dataclass
class KellyResult:
    full_kelly_fraction: float
    fractional_kelly: float
    position_size_usd: float
    bankroll: float
    kelly_multiplier: float
    win_prob: float
    net_odds: float
    capped: bool = False
    cap_reason: str = ""


def kelly_criterion(
    win_prob: float,
    market_price: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.05,
) -> KellyResult:
    """
    Calculate optimal position size using Kelly Criterion.

    Args:
        win_prob: Model's estimated probability of YES outcome (0-1)
        market_price: Current market price / implied probability (0-1)
        bankroll: Total available capital in USD
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter-Kelly)
        max_position_pct: Maximum position as % of bankroll

    Returns:
        KellyResult with sizing details
    """
    if win_prob <= 0 or win_prob >= 1:
        return KellyResult(0, 0, 0, bankroll, kelly_fraction, win_prob, 0)

    if market_price <= 0 or market_price >= 1:
        return KellyResult(0, 0, 0, bankroll, kelly_fraction, win_prob, 0)

    # Net odds: what you win per dollar risked
    # If market price is 0.40, you pay $0.40 to win $1.00, net odds = (1/0.40) - 1 = 1.5
    b = (1.0 / market_price) - 1.0
    p = win_prob
    q = 1.0 - p

    # Full Kelly: f* = (p * b - q) / b
    full_kelly = (p * b - q) / b

    # If full Kelly is negative, don't bet (no edge)
    if full_kelly <= 0:
        return KellyResult(
            full_kelly_fraction=full_kelly,
            fractional_kelly=0,
            position_size_usd=0,
            bankroll=bankroll,
            kelly_multiplier=kelly_fraction,
            win_prob=p,
            net_odds=b,
        )

    # Apply fractional Kelly
    frac_kelly = full_kelly * kelly_fraction

    # Calculate position in USD
    position_usd = bankroll * frac_kelly

    # Apply maximum position cap
    max_usd = bankroll * max_position_pct
    capped = False
    cap_reason = ""
    if position_usd > max_usd:
        position_usd = max_usd
        capped = True
        cap_reason = f"Capped at {max_position_pct*100:.0f}% of bankroll (${max_usd:.2f})"

    return KellyResult(
        full_kelly_fraction=full_kelly,
        fractional_kelly=frac_kelly,
        position_size_usd=round(position_usd, 2),
        bankroll=bankroll,
        kelly_multiplier=kelly_fraction,
        win_prob=p,
        net_odds=b,
        capped=capped,
        cap_reason=cap_reason,
    )


def print_kelly(result: KellyResult):
    """Pretty-print Kelly Criterion results."""
    print(f"Kelly Criterion Position Sizing")
    print(f"{'─' * 40}")
    print(f"  Win Probability:     {result.win_prob:.1%}")
    print(f"  Net Odds:            {result.net_odds:.2f}:1")
    print(f"  Full Kelly:          {result.full_kelly_fraction:.4f} ({result.full_kelly_fraction:.1%})")
    print(f"  Fractional Kelly:    {result.fractional_kelly:.4f} ({result.fractional_kelly:.1%}) "
          f"[{result.kelly_multiplier:.0%} Kelly]")
    print(f"  Bankroll:            ${result.bankroll:,.2f}")
    print(f"  Position Size:       ${result.position_size_usd:,.2f}")
    if result.capped:
        print(f"  ⚠ {result.cap_reason}")
    print()


if __name__ == "__main__":
    # Example from the doc: $10k bankroll, 70% win prob, 2:1 odds
    print("Example: 70% win prob, 2:1 odds, $10,000 bankroll\n")

    # Full Kelly
    full = kelly_criterion(0.70, 0.333, 10000, kelly_fraction=1.0)
    print_kelly(full)

    # Quarter Kelly
    quarter = kelly_criterion(0.70, 0.333, 10000, kelly_fraction=0.25)
    print_kelly(quarter)

    # Test with command line args
    if len(sys.argv) == 4:
        p = float(sys.argv[1])
        price = float(sys.argv[2])
        bank = float(sys.argv[3])
        result = kelly_criterion(p, price, bank)
        print_kelly(result)
