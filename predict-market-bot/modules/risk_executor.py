"""
Step 4: Risk Management & Execution

Validates all risk checks, calculates position sizes via Kelly Criterion,
executes trades via platform APIs, and monitors for hedging/exit conditions.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

from config import get_setting, get_env
from modules.scanner import Market
from modules.predictor import TradeSignal
from scripts.kelly_size import kelly_criterion, KellyResult
from scripts.validate_risk import validate_risk, RiskValidation

logger = logging.getLogger(__name__)

DATA_DIR = Path(get_setting("general", "data_dir") or "data")
KILL_SWITCH = get_setting("general", "kill_switch_file") or "STOP"


@dataclass
class Position:
    """Represents an open position."""
    position_id: str
    platform: str
    market_id: str
    market_title: str
    side: str                  # "yes" or "no"
    entry_price: float
    size_usd: float
    contracts: int
    predicted_prob: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    status: str = "open"      # "open", "closed", "hedged"


@dataclass
class ExecutionResult:
    """Result of a trade execution attempt."""
    success: bool
    position: Optional[Position] = None
    risk_validation: Optional[RiskValidation] = None
    kelly_result: Optional[KellyResult] = None
    reason: str = ""
    order_id: str = ""


class PortfolioTracker:
    """Tracks all open positions and portfolio state."""

    def __init__(self):
        self.positions: list[Position] = []
        self.trade_history: list[Position] = []
        self.bankroll = get_setting("general", "initial_bankroll") or 500.0
        self.peak_bankroll = self.bankroll
        self.daily_pnl = 0.0
        self.daily_api_cost = 0.0
        self.state_file = DATA_DIR / "portfolio_state.json"
        self._load_state()

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status == "open"]

    @property
    def current_exposure(self) -> float:
        return sum(p.size_usd for p in self.open_positions)

    @property
    def current_drawdown(self) -> float:
        if self.peak_bankroll <= 0:
            return 0.0
        return (self.peak_bankroll - self.bankroll) / self.peak_bankroll

    def add_position(self, position: Position):
        self.positions.append(position)
        self._save_state()

    def close_position(self, position_id: str, exit_price: float):
        for p in self.positions:
            if p.position_id == position_id and p.status == "open":
                p.exit_price = exit_price
                p.exit_time = datetime.now(timezone.utc)
                p.status = "closed"

                # Calculate PnL
                if p.side == "yes":
                    p.pnl = (exit_price - p.entry_price) * p.contracts
                else:
                    p.pnl = (p.entry_price - exit_price) * p.contracts

                self.bankroll += p.pnl
                self.daily_pnl += p.pnl
                self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
                self.trade_history.append(p)
                self._save_state()
                logger.info(f"Closed position {position_id}: PnL ${p.pnl:+.2f}")
                return
        logger.warning(f"Position {position_id} not found or already closed")

    def reset_daily_counters(self):
        self.daily_pnl = 0.0
        self.daily_api_cost = 0.0
        self._save_state()

    def _save_state(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "bankroll": self.bankroll,
            "peak_bankroll": self.peak_bankroll,
            "daily_pnl": self.daily_pnl,
            "daily_api_cost": self.daily_api_cost,
            "open_positions": len(self.open_positions),
            "total_trades": len(self.trade_history),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        self.state_file.write_text(json.dumps(state, indent=2))

    def _load_state(self):
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text())
                self.bankroll = state.get("bankroll", self.bankroll)
                self.peak_bankroll = state.get("peak_bankroll", self.peak_bankroll)
                self.daily_pnl = state.get("daily_pnl", 0.0)
                self.daily_api_cost = state.get("daily_api_cost", 0.0)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to load portfolio state, using defaults")


class TradeExecutor:
    """Handles order execution on Polymarket and Kalshi."""

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading or get_setting("general", "paper_trading", default=True)

    async def execute_order(
        self, platform: str, market_id: str, side: str,
        price: float, size_usd: float
    ) -> tuple[bool, str, str]:
        """
        Place an order. Returns (success, order_id, reason).
        Uses limit orders to control slippage.
        """
        if self.paper_trading:
            order_id = f"paper_{platform}_{market_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            logger.info(f"[PAPER] Order placed: {side} on {platform}/{market_id} "
                       f"@ {price:.3f} for ${size_usd:.2f}")
            return True, order_id, "Paper trade executed"

        if platform == "polymarket":
            return await self._execute_polymarket(market_id, side, price, size_usd)
        elif platform == "kalshi":
            return await self._execute_kalshi(market_id, side, price, size_usd)
        else:
            return False, "", f"Unknown platform: {platform}"

    async def _execute_polymarket(self, market_id: str, side: str,
                                   price: float, size_usd: float) -> tuple[bool, str, str]:
        """Execute on Polymarket CLOB."""
        api_key = get_env("POLYMARKET_API_KEY")
        if not api_key:
            return False, "", "Polymarket API key not configured"

        try:
            async with aiohttp.ClientSession() as session:
                # Check current price for slippage
                url = f"https://clob.polymarket.com/book"
                params = {"token_id": market_id}
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        book = await resp.json()
                        best_ask = float(book.get("asks", [{}])[0].get("price", 0)) if book.get("asks") else 0
                        if best_ask > 0:
                            slippage = abs(best_ask - price) / price
                            slippage_limit = get_setting("risk", "slippage_abort_pct", default=2.0) / 100
                            if slippage > slippage_limit:
                                return False, "", f"Slippage {slippage:.1%} exceeds {slippage_limit:.1%} limit"

                # Place limit order via CLOB API
                order_url = "https://clob.polymarket.com/order"
                contracts = int(size_usd / price)
                order_payload = {
                    "tokenID": market_id,
                    "price": str(price),
                    "size": str(contracts),
                    "side": "BUY" if side == "yes" else "SELL",
                    "type": "GTC",  # Good till cancelled
                }

                async with session.post(order_url, json=order_payload,
                                         headers={"Authorization": f"Bearer {api_key}"}) as resp:
                    if resp.status in (200, 201):
                        data = await resp.json()
                        return True, data.get("orderID", ""), "Order placed"
                    else:
                        text = await resp.text()
                        return False, "", f"Polymarket order failed: {resp.status} - {text}"
        except Exception as e:
            return False, "", f"Polymarket execution error: {e}"

    async def _execute_kalshi(self, market_id: str, side: str,
                               price: float, size_usd: float) -> tuple[bool, str, str]:
        """Execute on Kalshi."""
        api_key = get_env("KALSHI_API_KEY")
        if not api_key:
            return False, "", "Kalshi API key not configured"

        base_url = get_setting("platforms", "kalshi", "base_url") or "https://demo-api.kalshi.co/trade-api/v2"
        try:
            async with aiohttp.ClientSession() as session:
                contracts = max(1, int(size_usd / (price * 100)))  # Kalshi prices in cents
                order_url = f"{base_url}/portfolio/orders"
                order_payload = {
                    "ticker": market_id,
                    "action": "buy",
                    "side": side,
                    "type": "limit",
                    "count": contracts,
                    "yes_price": int(price * 100) if side == "yes" else None,
                    "no_price": int((1 - price) * 100) if side == "no" else None,
                }

                async with session.post(order_url, json=order_payload,
                                         headers={"Authorization": f"Bearer {api_key}"}) as resp:
                    if resp.status in (200, 201):
                        data = await resp.json()
                        return True, data.get("order", {}).get("order_id", ""), "Order placed"
                    else:
                        text = await resp.text()
                        return False, "", f"Kalshi order failed: {resp.status} - {text}"
        except Exception as e:
            return False, "", f"Kalshi execution error: {e}"


class CorrelationTracker:
    """
    Tracks correlations between markets to prevent hidden concentration risk.

    If you bet on "Biden wins" AND "Democrats win Senate", those are ~80% correlated.
    Independent Kelly sizing would over-allocate to effectively the same bet.

    Groups correlated markets and applies a shared position cap.
    """

    CORRELATION_THRESHOLD = 0.5  # Markets above this are considered correlated

    def __init__(self):
        self.keyword_index: dict[str, list[str]] = {}  # keyword -> [market_ids]
        self.correlation_groups: dict[str, set[str]] = {}  # group_id -> {market_ids}

    def register_position(self, market_id: str, market_title: str):
        """Index a market's keywords for correlation detection."""
        stopwords = {"will", "the", "be", "a", "an", "in", "on", "at", "to", "by", "of", "for",
                     "is", "or", "and", "before", "after", "yes", "no"}
        keywords = [
            w.lower().strip("?.,!") for w in market_title.split()
            if w.lower().strip("?.,!") not in stopwords and len(w) > 2
        ]
        for kw in keywords:
            if kw not in self.keyword_index:
                self.keyword_index[kw] = []
            if market_id not in self.keyword_index[kw]:
                self.keyword_index[kw].append(market_id)

    def estimate_correlation(self, market_id_a: str, title_a: str,
                              market_id_b: str, title_b: str) -> float:
        """
        Estimate correlation between two markets based on keyword overlap.
        Returns 0-1 score. Higher = more correlated.
        """
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

    def get_correlated_exposure(self, market_title: str,
                                  open_positions: list[Position]) -> float:
        """
        Calculate total exposure in positions correlated with this market.
        Returns total USD exposure in correlated positions.
        """
        correlated_exposure = 0.0

        for pos in open_positions:
            correlation = self.estimate_correlation(
                "new", market_title,
                pos.market_id, pos.market_title,
            )
            if correlation >= self.CORRELATION_THRESHOLD:
                correlated_exposure += pos.size_usd

        return correlated_exposure

    def scale_for_correlation(self, proposed_size: float, market_title: str,
                                open_positions: list[Position],
                                bankroll: float, max_position_pct: float) -> tuple[float, str]:
        """
        Scale down a position if correlated positions already exist.
        Returns (adjusted_size, reason).
        """
        correlated_exposure = self.get_correlated_exposure(market_title, open_positions)
        max_correlated = bankroll * max_position_pct

        if correlated_exposure <= 0:
            return proposed_size, ""

        total_if_added = correlated_exposure + proposed_size
        if total_if_added <= max_correlated:
            return proposed_size, ""

        # Scale down to fit within the correlated cap
        remaining = max(0, max_correlated - correlated_exposure)
        if remaining <= 0:
            return 0.0, (
                f"Correlated exposure ${correlated_exposure:.2f} already at cap "
                f"${max_correlated:.2f} — blocking trade"
            )

        scale_factor = remaining / proposed_size
        adjusted = proposed_size * scale_factor
        reason = (
            f"Scaled {scale_factor:.0%}: correlated exposure ${correlated_exposure:.2f} + "
            f"${adjusted:.2f} = ${correlated_exposure + adjusted:.2f} "
            f"(cap ${max_correlated:.2f})"
        )
        return adjusted, reason


class RiskManager:
    """
    Orchestrates risk checks, position sizing, and trade execution.
    The gatekeeper between signals and real money.
    Now includes correlation-aware position sizing.
    """

    def __init__(self):
        self.portfolio = PortfolioTracker()
        self.executor = TradeExecutor(
            paper_trading=get_setting("general", "paper_trading", default=True)
        )
        self.correlation = CorrelationTracker()
        self.kelly_fraction = get_setting("risk", "kelly_fraction") or 0.25
        self.max_position_pct = get_setting("risk", "max_position_pct", default=5.0) / 100
        self.max_drawdown = get_setting("risk", "max_drawdown_pct", default=8.0) / 100
        self.max_daily_loss = get_setting("risk", "max_daily_loss_pct", default=15.0) / 100
        self.max_positions = get_setting("risk", "max_concurrent_positions") or 15
        self.max_api_cost = get_setting("risk", "max_daily_api_cost_usd") or 50.0

        # Register existing positions for correlation tracking
        for pos in self.portfolio.open_positions:
            self.correlation.register_position(pos.market_id, pos.market_title)

    def check_kill_switch(self) -> bool:
        """Check if the kill switch file exists."""
        if os.path.exists(KILL_SWITCH):
            logger.critical("KILL SWITCH ACTIVE - All trading halted")
            return True
        return False

    async def process_signal(self, signal: TradeSignal) -> ExecutionResult:
        """Process a trade signal through full risk pipeline with correlation checks."""
        market = signal.market

        # Kill switch check
        if self.check_kill_switch():
            return ExecutionResult(success=False, reason="Kill switch active")

        # Only process actionable signals
        if signal.signal == "no_trade":
            return ExecutionResult(success=False, reason="No trade signal")

        # Calculate position size via Kelly
        kelly = kelly_criterion(
            win_prob=signal.predicted_probability,
            market_price=signal.market_price,
            bankroll=self.portfolio.bankroll,
            kelly_fraction=self.kelly_fraction,
            max_position_pct=self.max_position_pct,
        )

        if kelly.position_size_usd <= 0:
            return ExecutionResult(
                success=False,
                kelly_result=kelly,
                reason="Kelly says no bet (no edge)",
            )

        # ─── Correlation-aware sizing ───
        adjusted_size, corr_reason = self.correlation.scale_for_correlation(
            proposed_size=kelly.position_size_usd,
            market_title=market.title,
            open_positions=self.portfolio.open_positions,
            bankroll=self.portfolio.bankroll,
            max_position_pct=self.max_position_pct,
        )

        if adjusted_size <= 0:
            return ExecutionResult(
                success=False,
                kelly_result=kelly,
                reason=f"Correlation block: {corr_reason}",
            )

        if corr_reason:
            logger.info(f"Correlation adjustment: {corr_reason}")

        # Use adjusted size for risk validation
        position_size = adjusted_size

        # Run risk validation
        risk_result = validate_risk(
            p_model=signal.predicted_probability,
            p_market=signal.market_price,
            position_size_usd=position_size,
            bankroll=self.portfolio.bankroll,
            current_exposure_usd=self.portfolio.current_exposure,
            daily_pnl_usd=self.portfolio.daily_pnl,
            current_drawdown_pct=self.portfolio.current_drawdown,
            open_positions_count=len(self.portfolio.open_positions),
            daily_api_cost_usd=self.portfolio.daily_api_cost,
            min_edge=0.04,
            max_position_pct=self.max_position_pct,
            max_drawdown_pct=self.max_drawdown,
            max_daily_loss_pct=self.max_daily_loss,
            max_concurrent_positions=self.max_positions,
            max_daily_api_cost=self.max_api_cost,
        )

        logger.info(f"Risk validation:\n{risk_result.summary()}")

        if not risk_result.all_passed:
            failed = [c.check_name for c in risk_result.checks if not c.passed]
            return ExecutionResult(
                success=False,
                risk_validation=risk_result,
                kelly_result=kelly,
                reason=f"Risk checks failed: {', '.join(failed)}",
            )

        # Execute the trade
        side = "yes" if signal.signal == "buy_yes" else "no"
        price = market.current_price if side == "yes" else (1 - market.current_price)

        success, order_id, reason = await self.executor.execute_order(
            platform=market.platform,
            market_id=market.market_id,
            side=side,
            price=price,
            size_usd=position_size,
        )

        if success:
            contracts = int(position_size / price) if price > 0 else 0
            position = Position(
                position_id=order_id,
                platform=market.platform,
                market_id=market.market_id,
                market_title=market.title,
                side=side,
                entry_price=price,
                size_usd=position_size,
                contracts=contracts,
                predicted_prob=signal.predicted_probability,
            )
            self.portfolio.add_position(position)
            self.correlation.register_position(market.market_id, market.title)

            reason_full = reason
            if corr_reason:
                reason_full += f" | {corr_reason}"

            return ExecutionResult(
                success=True,
                position=position,
                risk_validation=risk_result,
                kelly_result=kelly,
                order_id=order_id,
                reason=reason_full,
            )
        else:
            return ExecutionResult(
                success=False,
                risk_validation=risk_result,
                kelly_result=kelly,
                reason=reason,
            )

    async def process_signals(self, signals: list[TradeSignal]) -> list[ExecutionResult]:
        """Process multiple trade signals sequentially (risk requires sequential checks)."""
        results = []
        for signal in signals:
            if signal.signal != "no_trade":
                result = await self.process_signal(signal)
                results.append(result)
                logger.info(
                    f"{'EXECUTED' if result.success else 'BLOCKED'}: "
                    f"{signal.market.title[:40]} - {result.reason}"
                )
        return results

    async def monitor_positions(self):
        """Monitor open positions for hedging/exit conditions."""
        for position in self.portfolio.open_positions:
            logger.debug(f"Monitoring: {position.market_title[:40]} - {position.status}")

    def status_report(self) -> str:
        """Generate current portfolio status."""
        return (
            f"=== Portfolio Status ===\n"
            f"  Bankroll:     ${self.portfolio.bankroll:,.2f}\n"
            f"  Exposure:     ${self.portfolio.current_exposure:,.2f}\n"
            f"  Open Pos:     {len(self.portfolio.open_positions)}\n"
            f"  Daily P&L:    ${self.portfolio.daily_pnl:+,.2f}\n"
            f"  Drawdown:     {self.portfolio.current_drawdown:.1%}\n"
            f"  API Cost:     ${self.portfolio.daily_api_cost:.2f}\n"
            f"  Paper Mode:   {self.executor.paper_trading}\n"
            f"  Kill Switch:  {'ACTIVE' if self.check_kill_switch() else 'off'}"
        )


async def run_execution(signals: list[TradeSignal]) -> list[ExecutionResult]:
    """Entry point for risk management and execution."""
    manager = RiskManager()

    tradeable = [s for s in signals if s.signal != "no_trade"]
    print(f"\n=== Risk & Execution ({len(tradeable)} signals to process) ===\n")

    results = await manager.process_signals(signals)

    executed = [r for r in results if r.success]
    blocked = [r for r in results if not r.success]

    print(f"\nExecuted: {len(executed)} | Blocked: {len(blocked)}")
    print(manager.status_report())

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Risk executor module - run via main.py pipeline")
