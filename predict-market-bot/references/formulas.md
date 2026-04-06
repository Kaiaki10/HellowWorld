# Core Formulas Reference

## Market Edge
```
edge = p_model - p_market
```
Only trade when `|edge| > 0.04` (4%)

## Expected Value
```
EV = p * b - (1 - p)
```
Where:
- `p` = your model probability
- `b` = decimal odds minus 1 = `(1 / p_market) - 1`

## Kelly Criterion (Position Sizing)
```
f* = (p * b - q) / b
```
Where:
- `p` = win probability
- `q` = 1 - p
- `b` = net odds

**Use Fractional Kelly** (multiply by 0.25 to 0.50):
- Full Kelly is mathematically optimal but extremely volatile
- Quarter-Kelly (0.25) recommended for real trading

### Example
- Bankroll: $10,000
- Win probability: 70%
- Reward/risk: 2:1
- Full Kelly: 12% ($1,200)
- Quarter-Kelly: 3% ($300)

## Mispricing Z-Score
```
delta = (p_model - p_market) / standard_deviation
```
Higher is better. Measures model-vs-market divergence in standard deviations.

## Brier Score (Calibration)
```
BS = (1/n) * Σ(predicted - outcome)²
```
- Lower is better
- Well-calibrated model: BS < 0.25
- Perfect score: 0.0

## Sharpe Ratio
```
Sharpe = mean(returns) / std(returns) * sqrt(252)
```
Target: above 2.0

## Profit Factor
```
PF = gross_profit / gross_loss
```
Target: above 1.5
