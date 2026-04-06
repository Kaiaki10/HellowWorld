"""
Prediction Market Trading Bot - Main Orchestrator

Runs the 5-step pipeline: Scan → Research → Predict → Risk/Execute → Compound
Plus: Arbitrage detection, Market making, Real-time monitoring

Modes:
    python main.py                  # Run full pipeline once
    python main.py --step scan      # Run only the scanner
    python main.py --step research  # Run scanner + research
    python main.py --step predict   # Run scanner + research + predict
    python main.py --review         # Run nightly review
    python main.py --schedule       # Run on a schedule (every 30 min)
    python main.py --realtime       # RECOMMENDED: WebSocket + scheduled hybrid
    python main.py --arbitrage      # Scan for cross-platform arbitrage
    python main.py --market-make    # Run market-making strategy
    python main.py --status         # Show portfolio status
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_setting
from modules.scanner import MarketScanner, Market, run_scan
from modules.researcher import MarketResearcher, run_research
from modules.predictor import EnsemblePredictor, run_predictions
from modules.risk_executor import RiskManager, run_execution
from modules.compounder import Compounder
from modules.realtime_monitor import RealtimeMonitor, AnomalyEvent
from modules.arbitrage import ArbitrageDetector, run_arbitrage_scan
from modules.market_maker import MarketMaker

logger = logging.getLogger("predict-market-bot")

KILL_SWITCH = get_setting("general", "kill_switch_file") or "STOP"


def setup_logging():
    """Configure logging."""
    log_level = get_setting("general", "log_level") or "INFO"
    log_dir = Path(get_setting("general", "log_dir") or "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"bot_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def check_kill_switch() -> bool:
    """Check if trading should be halted."""
    if os.path.exists(KILL_SWITCH):
        logger.critical(f"KILL SWITCH ACTIVE ({KILL_SWITCH} file found) - All trading halted")
        return True
    return False


async def run_pipeline(step: str = "all", top_markets: int = 10) -> list[Market]:
    """
    Run the trading pipeline. Returns discovered markets for WebSocket monitoring.

    Args:
        step: Which step to run up to ("scan", "research", "predict", "execute", "all")
        top_markets: How many top markets to process
    """
    if check_kill_switch():
        print("Kill switch is active. Remove the STOP file to resume trading.")
        return []

    compounder = Compounder()

    # ─── Step 1: Scan ───
    print("\n" + "=" * 60)
    print("STEP 1: SCANNING MARKETS")
    print("=" * 60)

    scanner = MarketScanner()
    markets = await scanner.scan()
    print(scanner.format_results(markets, top_n=top_markets))

    if not markets:
        print("No tradeable markets found. Exiting.")
        return []

    if step == "scan":
        return markets

    # Check knowledge base for warnings on these markets
    for market in markets[:top_markets]:
        warnings = compounder.get_market_warnings(market.title)
        if warnings:
            print(f"\n  [!] Warnings for '{market.title[:40]}':")
            for w in warnings:
                print(f"      {w}")

    # ─── Step 2: Research ───
    print("\n" + "=" * 60)
    print("STEP 2: RESEARCHING TOP MARKETS")
    print("=" * 60)

    researcher = MarketResearcher()
    briefs = await researcher.research_markets(markets[:top_markets])

    for brief in briefs:
        print(brief.summary())
        print()

    if step == "research":
        return markets

    # ─── Step 3: Predict ───
    print("\n" + "=" * 60)
    print("STEP 3: GENERATING PREDICTIONS")
    print("=" * 60)

    predictor = EnsemblePredictor()
    signals = await predictor.predict_batch(briefs)

    tradeable = [s for s in signals if s.signal != "no_trade"]
    print(f"\nTrade signals: {len(tradeable)} actionable from {len(signals)} analyzed\n")
    for s in signals:
        print(s.summary())
        print()

    if step == "predict":
        return markets

    # ─── Step 4: Risk & Execute ───
    print("\n" + "=" * 60)
    print("STEP 4: RISK CHECK & EXECUTION")
    print("=" * 60)

    if check_kill_switch():
        print("Kill switch activated during pipeline. Halting.")
        return markets

    risk_manager = RiskManager()
    results = await risk_manager.process_signals(signals)

    executed = [r for r in results if r.success]
    blocked = [r for r in results if not r.success]

    print(f"\nExecuted: {len(executed)} | Blocked: {len(blocked)}")
    for r in results:
        status = "EXECUTED" if r.success else "BLOCKED"
        market_title = r.position.market_title[:40] if r.position else "N/A"
        size = f"${r.kelly_result.position_size_usd:.2f}" if r.kelly_result else "N/A"
        print(f"  [{status}] {market_title} | Size: {size} | {r.reason}")

    print(f"\n{risk_manager.status_report()}")

    # ─── Step 5: Compound (for any closed trades) ───
    metrics = compounder.current_metrics()
    if metrics.total_trades > 0:
        print(f"\n{metrics.summary()}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return markets


async def run_pipeline_for_market(market: Market):
    """Run research → predict → execute for a single market (triggered by WebSocket anomaly)."""
    if check_kill_switch():
        return

    compounder = Compounder()
    warnings = compounder.get_market_warnings(market.title)
    if warnings:
        logger.info(f"Knowledge base warnings for '{market.title[:40]}': {len(warnings)}")

    # Research
    researcher = MarketResearcher()
    brief = await researcher.research_market(market)
    logger.info(f"Research: {brief.summary()}")

    # Predict
    predictor = EnsemblePredictor()
    signal = await predictor.predict(brief)
    logger.info(f"Prediction: {signal.summary()}")

    if signal.signal == "no_trade":
        logger.info(f"No trade signal for {market.title[:40]}")
        return

    # Execute
    risk_manager = RiskManager()
    result = await risk_manager.process_signal(signal)

    if result.success:
        logger.warning(
            f"TRADE EXECUTED: {market.title[:40]} | "
            f"Side: {result.position.side} | Size: ${result.kelly_result.position_size_usd:.2f}"
        )
    else:
        logger.info(f"Trade blocked: {result.reason}")


async def run_realtime(top_markets: int = 10, discovery_interval_min: int = 30):
    """
    RECOMMENDED MODE: Hybrid WebSocket + scheduled discovery.

    - WebSocket feeds monitor known markets in real-time (sub-5s reaction)
    - Scheduled scan discovers NEW markets every 30-60 min
    - Pipeline only runs when an anomaly is detected (saves LLM costs)
    """
    print("=" * 60)
    print("REAL-TIME MODE (WebSocket + Scheduled Discovery)")
    print("=" * 60)
    print(f"  Discovery interval: {discovery_interval_min} min")
    print(f"  Top markets to watch: {top_markets}")
    print(f"  Paper trading: {get_setting('general', 'paper_trading', default=True)}")
    print(f"  Kill switch: {KILL_SWITCH}")
    print("  Press Ctrl+C to stop.\n")

    # Queue for anomaly events (thread-safe bridge from WebSocket callbacks)
    anomaly_queue: asyncio.Queue[tuple[AnomalyEvent, Market]] = asyncio.Queue()

    def on_anomaly(event: AnomalyEvent, market: Market):
        """Called by WebSocket monitor when anomaly detected."""
        logger.warning(f"ANOMALY: {event.summary()}")
        try:
            anomaly_queue.put_nowait((event, market))
        except asyncio.QueueFull:
            logger.warning("Anomaly queue full, dropping event")

    monitor = RealtimeMonitor(on_anomaly=on_anomaly)

    # Initial scan to discover markets
    print("Running initial market scan...")
    markets = await run_pipeline(step="scan", top_markets=top_markets)
    if markets:
        monitor.add_markets(markets[:top_markets])

    async def anomaly_processor():
        """Process anomaly events from the queue — triggers pipeline for each."""
        while True:
            event, market = await anomaly_queue.get()
            if check_kill_switch():
                continue
            try:
                logger.info(f"Processing anomaly: {event.summary()}")
                await run_pipeline_for_market(market)
            except Exception as e:
                logger.error(f"Error processing anomaly for {market.title[:40]}: {e}", exc_info=True)
            finally:
                anomaly_queue.task_done()

    async def discovery_loop():
        """Periodically scan for new markets and update WebSocket subscriptions."""
        while True:
            await asyncio.sleep(discovery_interval_min * 60)
            if check_kill_switch():
                continue
            try:
                logger.info("Running scheduled market discovery...")
                new_markets = await run_pipeline(step="scan", top_markets=top_markets)
                if new_markets:
                    monitor.add_markets(new_markets[:top_markets])
                    logger.info(f"Updated watch list: {len(monitor.watched_markets)} markets")
            except Exception as e:
                logger.error(f"Discovery scan error: {e}", exc_info=True)

    async def stats_reporter():
        """Log monitor stats periodically."""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            logger.info(monitor.stats())

    # Run everything concurrently
    await asyncio.gather(
        monitor.start(),           # WebSocket connections
        anomaly_processor(),       # Process anomaly events
        discovery_loop(),          # Discover new markets periodically
        stats_reporter(),          # Log stats
        return_exceptions=True,
    )


async def run_arbitrage(top_markets: int = 20):
    """
    Scan for cross-platform arbitrage opportunities.
    Looks for price discrepancies between Polymarket and Kalshi on the same events.
    """
    print("=" * 60)
    print("CROSS-PLATFORM ARBITRAGE SCAN")
    print("=" * 60)

    scanner = MarketScanner()
    markets = await scanner.scan()

    poly_markets = [m for m in markets if m.platform == "polymarket"]
    kalshi_markets = [m for m in markets if m.platform == "kalshi"]

    print(f"Polymarket: {len(poly_markets)} | Kalshi: {len(kalshi_markets)}")

    if not poly_markets or not kalshi_markets:
        print("Need markets from both platforms to detect arbitrage.")
        return

    detector = ArbitrageDetector()
    opportunities = detector.scan_for_arbitrage(poly_markets, kalshi_markets)
    print(detector.report())

    if opportunities:
        paper_trading = get_setting("general", "paper_trading", default=True)
        bankroll = get_setting("general", "initial_bankroll") or 500.0
        for opp in opportunities[:3]:  # Execute top 3
            result = await detector.execute_arbitrage(opp, bankroll, paper_trading=paper_trading)
            status = "EXECUTED" if result.get("success") else "FAILED"
            print(f"  [{status}] {opp.strategy} | ${result.get('expected_profit', 0):.2f} profit")


async def run_market_making(top_markets: int = 10):
    """
    Run market-making strategy: provide liquidity and earn spreads.
    """
    print("=" * 60)
    print("MARKET MAKING MODE")
    print("=" * 60)
    print(f"  Paper trading: {get_setting('general', 'paper_trading', default=True)}")
    print(f"  Kill switch: {KILL_SWITCH}")
    print("  Press Ctrl+C to stop.\n")

    scanner = MarketScanner()
    markets = await scanner.scan()

    maker = MarketMaker()
    suitable = maker.select_markets(markets[:top_markets * 2])

    if not suitable:
        print("No markets suitable for market making right now.")
        return

    print(f"Making markets on {len(suitable)} markets:\n")
    for m in suitable:
        print(f"  {m.title[:50]} | Price: {m.current_price:.2f} | Vol: {m.volume_24h:.0f}")
    print()

    await maker.run(suitable)


async def run_scheduled():
    """Run the pipeline on a schedule (polling mode, no WebSocket)."""
    interval = get_setting("scanner", "schedule_interval_minutes") or 30
    print(f"Starting scheduled pipeline (every {interval} minutes)")
    print(f"Paper trading: {get_setting('general', 'paper_trading', default=True)}")
    print(f"Kill switch file: {KILL_SWITCH}")
    print("Press Ctrl+C to stop.\n")

    while True:
        if check_kill_switch():
            print("Kill switch active. Waiting 60s before rechecking...")
            await asyncio.sleep(60)
            continue

        try:
            await run_pipeline()
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)

        print(f"\nNext run in {interval} minutes...")
        await asyncio.sleep(interval * 60)


def main():
    parser = argparse.ArgumentParser(description="Prediction Market Trading Bot")
    parser.add_argument("--step", choices=["scan", "research", "predict", "execute", "all"],
                       default="all", help="Pipeline step to run up to")
    parser.add_argument("--review", action="store_true", help="Run nightly review")
    parser.add_argument("--schedule", action="store_true", help="Run on polling schedule")
    parser.add_argument("--realtime", action="store_true",
                       help="RECOMMENDED: WebSocket real-time + scheduled discovery")
    parser.add_argument("--arbitrage", action="store_true",
                       help="Scan for cross-platform arbitrage opportunities")
    parser.add_argument("--market-make", action="store_true",
                       help="Run market-making strategy (earn spreads)")
    parser.add_argument("--status", action="store_true", help="Show portfolio status")
    parser.add_argument("--top", type=int, default=10, help="Number of top markets to process")
    parser.add_argument("--discovery-interval", type=int, default=30,
                       help="Minutes between market discovery scans (realtime mode)")
    args = parser.parse_args()

    setup_logging()

    if args.review:
        compounder = Compounder()
        print(compounder.nightly_review())
        return

    if args.status:
        manager = RiskManager()
        print(manager.status_report())
        metrics = Compounder().current_metrics()
        if metrics.total_trades > 0:
            print(f"\n{metrics.summary()}")
        return

    if args.arbitrage:
        asyncio.run(run_arbitrage(top_markets=args.top))
    elif args.market_make:
        asyncio.run(run_market_making(top_markets=args.top))
    elif args.realtime:
        asyncio.run(run_realtime(
            top_markets=args.top,
            discovery_interval_min=args.discovery_interval,
        ))
    elif args.schedule:
        asyncio.run(run_scheduled())
    else:
        asyncio.run(run_pipeline(step=args.step, top_markets=args.top))


if __name__ == "__main__":
    main()
