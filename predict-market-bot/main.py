"""
Prediction Market Trading Bot - Main Orchestrator

Runs the 5-step pipeline: Scan → Research → Predict → Risk/Execute → Compound

Usage:
    python main.py                  # Run full pipeline once
    python main.py --step scan      # Run only the scanner
    python main.py --step research  # Run scanner + research
    python main.py --step predict   # Run scanner + research + predict
    python main.py --review         # Run nightly review
    python main.py --schedule       # Run on a schedule (every 15 min)
    python main.py --status         # Show portfolio status
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_setting
from modules.scanner import MarketScanner, run_scan
from modules.researcher import MarketResearcher, run_research
from modules.predictor import EnsemblePredictor, run_predictions
from modules.risk_executor import RiskManager, run_execution
from modules.compounder import Compounder

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


async def run_pipeline(step: str = "all", top_markets: int = 10):
    """
    Run the trading pipeline.

    Args:
        step: Which step to run up to ("scan", "research", "predict", "execute", "all")
        top_markets: How many top markets to process
    """
    if check_kill_switch():
        print("Kill switch is active. Remove the STOP file to resume trading.")
        return

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
        return

    if step == "scan":
        return

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
        return

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
        return

    # ─── Step 4: Risk & Execute ───
    print("\n" + "=" * 60)
    print("STEP 4: RISK CHECK & EXECUTION")
    print("=" * 60)

    if check_kill_switch():
        print("Kill switch activated during pipeline. Halting.")
        return

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
    # Note: In live trading, this runs when positions close.
    # Here we log the current state for review.
    metrics = compounder.current_metrics()
    if metrics.total_trades > 0:
        print(f"\n{metrics.summary()}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


async def run_scheduled():
    """Run the pipeline on a schedule."""
    interval = get_setting("scanner", "schedule_interval_minutes") or 15
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
    parser.add_argument("--schedule", action="store_true", help="Run on schedule")
    parser.add_argument("--status", action="store_true", help="Show portfolio status")
    parser.add_argument("--top", type=int, default=10, help="Number of top markets to process")
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

    if args.schedule:
        asyncio.run(run_scheduled())
    else:
        asyncio.run(run_pipeline(step=args.step, top_markets=args.top))


if __name__ == "__main__":
    main()
