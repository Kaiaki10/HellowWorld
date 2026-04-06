# Prediction Market Trading Bot

Automated trading bot for [Polymarket](https://polymarket.com) and [Kalshi](https://kalshi.com) prediction markets. Uses a tiered LLM ensemble (Claude Opus, GPT-4o, Gemini) with domain-specialist routing for high-quality probability estimation.

## Architecture

**5-step pipeline:** Scan → Research → Predict → Risk/Execute → Compound

```
Markets (Polymarket + Kalshi)
    │
    ▼
┌─────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│  Scan   │───▶│  Research    │───▶│  Predict  │───▶│ Risk &   │───▶│ Compound │
│ Markets │    │ (Specialist) │    │ (Ensemble)│    │ Execute  │    │ (Learn)  │
└─────────┘    └──────────────┘    └───────────┘    └──────────┘    └──────────┘
                     │                   │
              ┌──────┴──────┐     ┌──────┴──────┐
              │ Auto-route  │     │ Tiered LLM  │
              │ by vertical │     │  routing    │
              └─────────────┘     └─────────────┘
```

## Features

- **Real-time monitoring** — WebSocket feeds for sub-5s reaction to price anomalies
- **Tiered model routing** — Skip LLMs when edge < 2%, Haiku for 2-4%, full ensemble for 4%+ (saves ~70% LLM costs)
- **Calibration-weighted ensemble** — Model weights auto-adjust based on per-model Brier Score track record
- **Cross-platform arbitrage** — Detects price discrepancies between Polymarket and Kalshi
- **Market making** — Two-sided quoting with inventory skew
- **Correlation-aware sizing** — Scales down when correlated positions build up
- **Research caching** — Reuses briefs when price moved < 2%
- **Paper trading** — Enabled by default, no real money at risk

## Market Specialists

Markets are auto-routed to domain specialists for higher quality predictions:

| Specialist | Data Sources | Edge |
|---|---|---|
| **Economics** | FRED API, treasury yields, leading indicators | Hard government data weighted 70% — it's the resolution source |
| **Weather** | Open-Meteo forecasts, historical baselines | Numerical weather models at 80% confidence beat any LLM |
| **Sports** | ESPN injuries, The Odds API, sport-specific Reddit | Bookmaker odds weighted 65% — they aggregate massive information |
| **Crypto** | CoinGecko prices, Fear & Greed Index | Volatility-based price targets + on-chain sentiment |
| **Politics** | Polling aggregators, political RSS | Poll numbers weighted 60% when detected |

Markets that don't match any specialist fall back to the general pipeline.

## Quick Start

### 1. Install dependencies

```bash
cd predict-market-bot
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp config/.env.example config/.env
# Edit config/.env with your keys
```

**Required:**
- `ANTHROPIC_API_KEY` — Claude Opus for predictions
- `POLYMARKET_API_KEY` + `POLYMARKET_WALLET_PRIVATE_KEY` — Polymarket trading
- `KALSHI_API_KEY` + `KALSHI_API_SECRET` — Kalshi trading

**Recommended (free):**
- `FRED_API_KEY` — Federal Reserve economic data ([get key](https://fred.stlouisfed.org/docs/api/api_key.html))
- `OPENAI_API_KEY` — GPT-4o for ensemble
- `GOOGLE_AI_API_KEY` — Gemini for ensemble

**Optional:**
- `ODDS_API_KEY` — Sports odds (free tier: 500 req/month)
- `TWITTER_BEARER_TOKEN` — Twitter sentiment
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` — Reddit research

### 3. Run

```bash
# Recommended: real-time WebSocket + scheduled discovery
python main.py --realtime

# Single pipeline run
python main.py

# Scan only (no trading)
python main.py --step scan

# Cross-platform arbitrage
python main.py --arbitrage

# Market making
python main.py --market-make

# Portfolio status
python main.py --status

# API cost report
python main.py --costs

# Nightly performance review
python main.py --review
```

## Configuration

All settings in `config/settings.yaml`:

| Setting | Default | Description |
|---|---|---|
| `general.paper_trading` | `true` | Paper trading mode (no real money) |
| `general.initial_bankroll` | `$500` | Starting bankroll |
| `risk.kelly_fraction` | `0.25` | Quarter-Kelly (conservative) |
| `risk.max_position_pct` | `5%` | Max per-position size |
| `risk.max_daily_api_cost_usd` | `$50` | Daily LLM cost cap |
| `prediction.min_edge` | `4%` | Minimum edge to trade |
| `scanner.schedule_interval_minutes` | `30` | Discovery scan interval |
| `specialists.enabled` | `true` | Enable specialist routing |

## Kill Switch

Create a file named `STOP` in the project root to immediately halt all trading:

```bash
touch STOP    # halt trading
rm STOP       # resume trading
```

## Project Structure

```
predict-market-bot/
├── main.py                      # Orchestrator (all modes)
├── config/
│   ├── settings.yaml            # All configuration
│   ├── .env.example             # API key template
│   └── __init__.py              # Config loader
├── modules/
│   ├── scanner.py               # Step 1: Market discovery
│   ├── researcher.py            # Step 2: Research + sentiment
│   ├── predictor.py             # Step 3: Tiered ensemble prediction
│   ├── risk_executor.py         # Step 4: Kelly sizing + execution
│   ├── compounder.py            # Step 5: Learning + review
│   ├── realtime_monitor.py      # WebSocket price monitoring
│   ├── arbitrage.py             # Cross-platform arbitrage
│   ├── market_maker.py          # Spread-earning strategy
│   ├── polymarket_signer.py     # EIP-712 order signing
│   ├── cost_tracker.py          # API cost tracking
│   ├── startup.py               # Pre-flight validation
│   └── specialists/
│       ├── __init__.py           # Classifier + registry
│       ├── politics.py           # Polls, political RSS
│       ├── weather.py            # Open-Meteo forecasts
│       ├── sports.py             # ESPN, bookmaker odds
│       ├── crypto.py             # CoinGecko, Fear & Greed
│       └── economics.py          # FRED, yield curve, leading indicators
├── scripts/
│   ├── kelly_size.py            # Kelly Criterion calculator
│   └── validate_risk.py         # Risk check validator
├── tests/
│   ├── test_kelly.py            # Kelly sizing tests
│   ├── test_risk.py             # Risk validation tests
│   └── test_correlation.py      # Correlation tracking tests
└── requirements.txt
```

## Estimated Costs

| Component | Cost | Notes |
|---|---|---|
| LLM ensemble (full tier) | ~$0.04/market | Only fires when edge > 4% |
| LLM Haiku (cheap tier) | ~$0.002/market | For 2-4% edge |
| Statistical model | Free | Handles < 2% edge |
| FRED API | Free | 120 req/min |
| Open-Meteo | Free | No key needed |
| CoinGecko | Free | No key needed |
| ESPN | Free | No key needed |
| The Odds API | Free tier | 500 req/month |
| **Typical daily cost** | **$5-15** | With tiered routing at 30-min intervals |

## Tests

```bash
cd predict-market-bot
python -m pytest tests/ -v
```
