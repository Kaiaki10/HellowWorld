"""
Step 5: Compound - Learn From Every Trade

Post-mortem analysis on every trade, especially losses.
Classifies failures, updates knowledge base, tracks performance metrics.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from config import get_setting
from modules.risk_executor import Position

logger = logging.getLogger(__name__)

DATA_DIR = Path(get_setting("general", "data_dir") or "data")
FAILURE_LOG = Path("references/failure_log.md")


@dataclass
class TradeRecord:
    """Complete record of a single trade for post-analysis."""
    position_id: str
    platform: str
    market_id: str
    market_title: str
    side: str
    entry_price: float
    exit_price: float
    predicted_probability: float
    actual_outcome: Optional[float] = None  # 1.0 if YES resolved, 0.0 if NO
    pnl: float = 0.0
    size_usd: float = 0.0
    contracts: int = 0
    entry_time: str = ""
    exit_time: str = ""
    hold_duration_hours: float = 0.0
    failure_category: str = ""   # "bad_prediction", "bad_timing", "bad_execution", "external_shock"
    lesson: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    brier_score: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    def summary(self) -> str:
        return (
            f"=== Performance Metrics ({self.total_trades} trades) ===\n"
            f"  Win Rate:      {self.win_rate:.1%} (target 60%+)\n"
            f"  Total P&L:     ${self.total_pnl:+,.2f}\n"
            f"  Profit Factor: {self.profit_factor:.2f} (target 1.5+)\n"
            f"  Sharpe Ratio:  {self.sharpe_ratio:.2f} (target 2.0+)\n"
            f"  Max Drawdown:  {self.max_drawdown:.1%} (limit 8%)\n"
            f"  Brier Score:   {self.brier_score:.4f} (target < 0.25)\n"
            f"  Avg Win:       ${self.avg_win:+,.2f}\n"
            f"  Avg Loss:      ${self.avg_loss:+,.2f}\n"
            f"  Best Trade:    ${self.best_trade:+,.2f}\n"
            f"  Worst Trade:   ${self.worst_trade:+,.2f}"
        )


class TradeLogger:
    """Logs every trade for post-analysis."""

    def __init__(self):
        self.log_file = DATA_DIR / "trade_log.json"
        self.records: list[TradeRecord] = []
        self._load()

    def log_trade(self, position: Position, actual_outcome: Optional[float] = None) -> TradeRecord:
        """Log a completed trade."""
        hold_hours = 0.0
        if position.entry_time and position.exit_time:
            delta = position.exit_time - position.entry_time
            hold_hours = delta.total_seconds() / 3600

        record = TradeRecord(
            position_id=position.position_id,
            platform=position.platform,
            market_id=position.market_id,
            market_title=position.market_title,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=position.exit_price or 0.0,
            predicted_probability=position.predicted_prob,
            actual_outcome=actual_outcome,
            pnl=position.pnl,
            size_usd=position.size_usd,
            contracts=position.contracts,
            entry_time=position.entry_time.isoformat() if position.entry_time else "",
            exit_time=position.exit_time.isoformat() if position.exit_time else "",
            hold_duration_hours=hold_hours,
        )

        # Auto-classify failure if it's a loss
        if record.pnl < 0:
            record.failure_category = self._classify_failure(record)
            record.lesson = self._generate_lesson(record)

        self.records.append(record)
        self._save()
        logger.info(f"Logged trade: {record.position_id} PnL: ${record.pnl:+.2f}")
        return record

    def _classify_failure(self, record: TradeRecord) -> str:
        """Classify why a trade lost money."""
        if record.actual_outcome is not None:
            # Check if prediction was fundamentally wrong
            pred_correct = (record.predicted_probability > 0.5) == (record.actual_outcome > 0.5)
            if not pred_correct:
                return "bad_prediction"

        # Short hold time suggests timing issue
        if record.hold_duration_hours < 1.0:
            return "bad_timing"

        # Large slippage between entry and exit
        if record.entry_price > 0:
            slippage = abs(record.exit_price - record.entry_price) / record.entry_price
            if slippage > 0.1:
                return "bad_execution"

        return "external_shock"

    def _generate_lesson(self, record: TradeRecord) -> str:
        """Generate a lesson from a losing trade."""
        lessons = {
            "bad_prediction": (
                f"Model predicted {record.predicted_probability:.0%} but outcome was "
                f"{record.actual_outcome:.0%}. Review research sources for "
                f"'{record.market_title[:40]}'. Consider source reliability."
            ),
            "bad_timing": (
                f"Position held only {record.hold_duration_hours:.1f}h before closing at a loss. "
                f"Review entry timing and patience thresholds."
            ),
            "bad_execution": (
                f"Entry {record.entry_price:.3f} vs exit {record.exit_price:.3f} shows "
                f"execution slippage. Review orderbook depth checks."
            ),
            "external_shock": (
                f"External event caused loss on '{record.market_title[:40]}'. "
                f"Monitor for similar event patterns in future."
            ),
        }
        return lessons.get(record.failure_category, "Unclassified loss - manual review needed.")

    def _save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        records_data = [asdict(r) for r in self.records]
        self.log_file.write_text(json.dumps(records_data, indent=2))

    def _load(self):
        if self.log_file.exists():
            try:
                data = json.loads(self.log_file.read_text())
                self.records = [TradeRecord(**r) for r in data]
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to load trade log, starting fresh")


class PerformanceAnalyzer:
    """Calculates performance metrics from trade history."""

    def analyze(self, records: list[TradeRecord]) -> PerformanceMetrics:
        if not records:
            return PerformanceMetrics()

        pnls = [r.pnl for r in records]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total = len(pnls)
        win_count = len(wins)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        # Sharpe Ratio: mean(returns) / std(returns) * sqrt(252)
        if len(pnls) >= 2:
            mean_return = np.mean(pnls)
            std_return = np.std(pnls)
            sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        else:
            sharpe = 0.0

        # Max Drawdown
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative)
        initial = records[0].size_usd if records else 1.0
        max_dd = float(np.max(drawdowns)) / max(initial, 1.0) if len(drawdowns) > 0 else 0.0

        # Brier Score (for trades with known outcomes)
        brier_records = [r for r in records if r.actual_outcome is not None]
        if brier_records:
            brier = sum(
                (r.predicted_probability - r.actual_outcome) ** 2
                for r in brier_records
            ) / len(brier_records)
        else:
            brier = 1.0

        return PerformanceMetrics(
            total_trades=total,
            winning_trades=win_count,
            losing_trades=len(losses),
            win_rate=win_count / total if total > 0 else 0.0,
            total_pnl=sum(pnls),
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            sharpe_ratio=float(sharpe),
            max_drawdown=max_dd,
            avg_win=np.mean(wins) if wins else 0.0,
            avg_loss=np.mean(losses) if losses else 0.0,
            brier_score=brier,
            best_trade=max(pnls) if pnls else 0.0,
            worst_trade=min(pnls) if pnls else 0.0,
        )


class KnowledgeBase:
    """
    Maintains a knowledge base of lessons learned.
    Read by scan and research agents before processing new markets.
    """

    def __init__(self):
        self.failure_log_path = FAILURE_LOG
        self.knowledge_file = DATA_DIR / "knowledge_base.json"
        self.entries: list[dict] = []
        self._load()

    def add_lesson(self, record: TradeRecord):
        """Add a lesson from a trade to the knowledge base."""
        entry = {
            "market_title": record.market_title,
            "failure_category": record.failure_category,
            "lesson": record.lesson,
            "pnl": record.pnl,
            "predicted": record.predicted_probability,
            "actual": record.actual_outcome,
            "timestamp": record.timestamp,
        }
        self.entries.append(entry)
        self._save()
        self._update_failure_log(record)

    def get_lessons_for_market(self, market_title: str) -> list[dict]:
        """Retrieve lessons relevant to a similar market."""
        keywords = set(market_title.lower().split())
        relevant = []
        for entry in self.entries:
            entry_keywords = set(entry["market_title"].lower().split())
            overlap = keywords & entry_keywords
            if len(overlap) >= 2:  # At least 2 keyword matches
                relevant.append(entry)
        return relevant

    def get_all_lessons(self) -> list[dict]:
        return self.entries

    def _update_failure_log(self, record: TradeRecord):
        """Append to the human-readable failure log."""
        self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = (
            f"\n## {record.timestamp[:10]} - {record.failure_category}\n"
            f"**Market:** {record.market_title}\n"
            f"**PnL:** ${record.pnl:+.2f} | "
            f"**Predicted:** {record.predicted_probability:.0%} | "
            f"**Actual:** {record.actual_outcome:.0%}\n"
            f"**Lesson:** {record.lesson}\n"
        )

        if self.failure_log_path.exists():
            content = self.failure_log_path.read_text()
        else:
            content = "# Failure Log\n\nLessons learned from losing trades.\n"
        content += entry
        self.failure_log_path.write_text(content)

    def _save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.knowledge_file.write_text(json.dumps(self.entries, indent=2))

    def _load(self):
        if self.knowledge_file.exists():
            try:
                self.entries = json.loads(self.knowledge_file.read_text())
            except json.JSONDecodeError:
                self.entries = []


class Compounder:
    """
    Orchestrates post-trade analysis, learning, and nightly reviews.
    Makes the system smarter over time.
    """

    def __init__(self):
        self.trade_logger = TradeLogger()
        self.analyzer = PerformanceAnalyzer()
        self.knowledge = KnowledgeBase()

    def process_closed_trade(self, position: Position, actual_outcome: Optional[float] = None):
        """Process a single closed trade: log, analyze, learn."""
        record = self.trade_logger.log_trade(position, actual_outcome)

        # If it was a loss, add to knowledge base
        if record.pnl < 0:
            self.knowledge.add_lesson(record)
            logger.info(f"Lesson learned: [{record.failure_category}] {record.lesson[:80]}")

        return record

    def nightly_review(self) -> str:
        """Run nightly consolidation - review all trades and update system."""
        records = self.trade_logger.records
        metrics = self.analyzer.analyze(records)

        report = [
            "=" * 60,
            f"NIGHTLY REVIEW - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "=" * 60,
            "",
            metrics.summary(),
            "",
        ]

        # Check if performance targets are met
        targets = get_setting("compound", "performance_targets") or {}
        alerts = []
        if metrics.win_rate < targets.get("win_rate", 0.60):
            alerts.append(f"Win rate {metrics.win_rate:.1%} below target {targets.get('win_rate', 0.60):.0%}")
        if metrics.sharpe_ratio < targets.get("sharpe_ratio", 2.0):
            alerts.append(f"Sharpe {metrics.sharpe_ratio:.2f} below target {targets.get('sharpe_ratio', 2.0)}")
        if metrics.profit_factor < targets.get("profit_factor", 1.5):
            alerts.append(f"Profit factor {metrics.profit_factor:.2f} below target {targets.get('profit_factor', 1.5)}")
        if metrics.max_drawdown > 0.08:
            alerts.append(f"Max drawdown {metrics.max_drawdown:.1%} exceeds 8% limit!")

        if alerts:
            report.append("ALERTS:")
            for a in alerts:
                report.append(f"  [!] {a}")
        else:
            report.append("All performance targets met.")

        # Recent failure analysis
        recent_losses = [r for r in records[-20:] if r.pnl < 0]
        if recent_losses:
            report.append(f"\nRecent losses ({len(recent_losses)} in last 20 trades):")
            categories = {}
            for r in recent_losses:
                cat = r.failure_category or "unclassified"
                categories[cat] = categories.get(cat, 0) + 1
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                report.append(f"  {cat}: {count}")

        review_text = "\n".join(report)
        logger.info(review_text)

        # Save review
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        review_file = DATA_DIR / f"review_{datetime.now(timezone.utc).strftime('%Y%m%d')}.txt"
        review_file.write_text(review_text)

        return review_text

    def get_market_warnings(self, market_title: str) -> list[str]:
        """Check knowledge base for warnings about a similar market."""
        lessons = self.knowledge.get_lessons_for_market(market_title)
        return [f"[{l['failure_category']}] {l['lesson']}" for l in lessons]

    def current_metrics(self) -> PerformanceMetrics:
        return self.analyzer.analyze(self.trade_logger.records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compounder = Compounder()
    print(compounder.nightly_review())
