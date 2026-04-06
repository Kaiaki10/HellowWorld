"""
Step 3: Predict - Estimate True Probability

Uses tiered model routing + calibration-weighted ensemble:
  1. Statistical model (free) pre-filters markets
  2. Only markets with edge > 2% get cheap LLM confirmation (Haiku)
  3. Only markets with edge > 4% get full ensemble (Opus + GPT-4 + Gemini)
  4. Model weights auto-adjust based on per-model Brier Score track record
"""

import json
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import aiohttp

from config import get_setting, get_env
from modules.scanner import Market
from modules.researcher import ResearchBrief
from modules.cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)

DATA_DIR = Path(get_setting("general", "data_dir") or "data")


# ─── Tier thresholds ───
TIER_SKIP = 0.02        # Edge < 2%: skip LLMs entirely
TIER_CHEAP = 0.04       # Edge 2-4%: cheap model only (Haiku)
# Edge >= 4%: full ensemble


@dataclass
class ModelPrediction:
    """Prediction from a single model in the ensemble."""
    model_name: str
    probability: float   # 0-1
    confidence: float    # 0-1
    reasoning: str = ""
    weight: float = 0.0
    tier: str = ""       # "free", "cheap", "full"


@dataclass
class TradeSignal:
    """Output signal from the prediction step."""
    market: Market
    research: ResearchBrief
    predicted_probability: float
    market_price: float
    edge: float                      # predicted - market
    expected_value: float
    mispricing_z_score: float
    confidence: float
    model_predictions: list[ModelPrediction] = field(default_factory=list)
    signal: str = "no_trade"         # "buy_yes", "buy_no", "no_trade"
    tier_used: str = ""              # "skip", "cheap", "full"
    estimated_cost: float = 0.0      # Estimated API cost for this prediction
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        return (
            f"[{self.signal.upper():8s}] {self.market.title[:55]}\n"
            f"  Model: {self.predicted_probability:.3f} vs Market: {self.market_price:.3f} | "
            f"Edge: {self.edge:+.3f} | EV: {self.expected_value:+.3f} | "
            f"Z: {self.mispricing_z_score:.2f} | Conf: {self.confidence:.2f} | "
            f"Tier: {self.tier_used} (~${self.estimated_cost:.3f})"
        )


class PerModelCalibration:
    """
    Tracks per-model Brier Scores over time.
    Dynamically adjusts ensemble weights: weight_i = (1/brier_i) / sum(1/brier_j).
    Models that predict well get more influence automatically.
    """

    def __init__(self):
        self.history_file = DATA_DIR / "model_calibration.json"
        self.records: dict[str, list[dict]] = {}  # model_name -> [{predicted, outcome}]
        self._load()

    def record(self, model_name: str, predicted: float, outcome: float):
        """Record a prediction and its outcome for a specific model."""
        if model_name not in self.records:
            self.records[model_name] = []
        self.records[model_name].append({
            "predicted": predicted,
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()

    def brier_score(self, model_name: str) -> float:
        """Brier score for a specific model. Lower is better."""
        records = self.records.get(model_name, [])
        if len(records) < 5:
            return 0.25  # Default until we have enough data
        n = len(records)
        total = sum((r["predicted"] - r["outcome"]) ** 2 for r in records)
        return total / n

    def get_calibrated_weights(self, model_names: list[str]) -> dict[str, float]:
        """
        Return calibration-adjusted weights.
        weight_i = (1 / brier_i) / sum(1 / brier_j)
        Models with lower (better) Brier scores get higher weight.
        """
        brier_scores = {}
        for name in model_names:
            bs = self.brier_score(name)
            brier_scores[name] = max(bs, 0.01)  # Prevent division by zero

        # Inverse Brier weighting
        inv_sum = sum(1.0 / bs for bs in brier_scores.values())
        if inv_sum == 0:
            # Equal weights fallback
            equal = 1.0 / len(model_names)
            return {name: equal for name in model_names}

        weights = {name: (1.0 / bs) / inv_sum for name, bs in brier_scores.items()}
        return weights

    def report(self) -> str:
        lines = ["Per-Model Calibration:"]
        for name, records in self.records.items():
            bs = self.brier_score(name)
            lines.append(f"  {name}: Brier={bs:.4f} ({len(records)} predictions)")
        return "\n".join(lines)

    def _save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.history_file.write_text(json.dumps(self.records, indent=2))

    def _load(self):
        if self.history_file.exists():
            try:
                self.records = json.loads(self.history_file.read_text())
            except json.JSONDecodeError:
                self.records = {}


class StatisticalModel:
    """Free statistical prediction based on research data."""

    def predict(self, brief: ResearchBrief) -> ModelPrediction:
        base_prob = brief.narrative_probability
        source_count = len(brief.sources)
        if source_count < 3:
            prob = base_prob * 0.4 + brief.market_price * 0.6
        else:
            prob = base_prob * 0.7 + brief.market_price * 0.3

        prob = max(0.05, min(0.95, prob))

        return ModelPrediction(
            model_name="statistical",
            probability=prob,
            confidence=brief.confidence * 0.8,
            reasoning=f"Based on {source_count} sources, sentiment {brief.overall_sentiment:+.2f}",
            weight=0.15,
            tier="free",
        )


class LLMPredictor:
    """Uses LLM APIs for probability estimation with tiered routing."""

    # Approximate costs per call (input + output tokens)
    COST_PER_CALL = {
        "haiku": 0.002,
        "claude": 0.024,
        "gpt4": 0.015,
        "gemini": 0.002,
    }

    async def predict_with_haiku(self, session: aiohttp.ClientSession,
                                  market_title: str, research_summary: str) -> Optional[ModelPrediction]:
        """Cheap confirmation via Claude Haiku."""
        api_key = get_env("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        tracker = get_cost_tracker()
        if not tracker.can_afford(tracker.estimate_call_cost("claude-haiku-4-5-20251001")):
            logger.warning("Daily API budget exceeded, skipping Haiku call")
            return None

        prompt = self._build_prompt(market_title, research_summary)
        try:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data["content"][0]["text"]
                    usage = data.get("usage", {})
                    tracker.record_call(
                        "claude-haiku-4-5-20251001",
                        usage.get("input_tokens", 500),
                        usage.get("output_tokens", 100),
                        source="predictor",
                    )
                    return self._parse_llm_response("haiku", text, 0.20, "cheap")
                logger.warning(f"Haiku API returned {resp.status}")
        except Exception as e:
            logger.error(f"Haiku prediction error: {e}")
        return None

    async def predict_with_claude(self, session: aiohttp.ClientSession,
                                   market_title: str, research_summary: str) -> Optional[ModelPrediction]:
        """Full prediction via Claude Opus."""
        api_key = get_env("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        tracker = get_cost_tracker()
        if not tracker.can_afford(tracker.estimate_call_cost("claude-opus-4-20250514")):
            logger.warning("Daily API budget exceeded, skipping Opus call")
            return None

        prompt = self._build_prompt(market_title, research_summary)
        try:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-opus-4-20250514",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data["content"][0]["text"]
                    usage = data.get("usage", {})
                    tracker.record_call(
                        "claude-opus-4-20250514",
                        usage.get("input_tokens", 800),
                        usage.get("output_tokens", 200),
                        source="predictor",
                    )
                    return self._parse_llm_response("claude", text, 0.20, "full")
                logger.warning(f"Claude API returned {resp.status}")
        except Exception as e:
            logger.error(f"Claude prediction error: {e}")
        return None

    async def predict_with_gpt4(self, session: aiohttp.ClientSession,
                                 market_title: str, research_summary: str) -> Optional[ModelPrediction]:
        """Full prediction via GPT-4."""
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            return None

        tracker = get_cost_tracker()
        if not tracker.can_afford(tracker.estimate_call_cost("gpt-4o")):
            logger.warning("Daily API budget exceeded, skipping GPT-4 call")
            return None

        prompt = self._build_prompt(market_title, research_summary)
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    tracker.record_call(
                        "gpt-4o",
                        usage.get("prompt_tokens", 800),
                        usage.get("completion_tokens", 200),
                        source="predictor",
                    )
                    return self._parse_llm_response("gpt4", text, 0.20, "full")
                logger.warning(f"GPT-4 API returned {resp.status}")
        except Exception as e:
            logger.error(f"GPT-4 prediction error: {e}")
        return None

    async def predict_with_gemini(self, session: aiohttp.ClientSession,
                                   market_title: str, research_summary: str) -> Optional[ModelPrediction]:
        """Full prediction via Gemini."""
        api_key = get_env("GOOGLE_AI_API_KEY")
        if not api_key:
            return None

        tracker = get_cost_tracker()
        if not tracker.can_afford(tracker.estimate_call_cost("gemini-pro")):
            logger.warning("Daily API budget exceeded, skipping Gemini call")
            return None

        prompt = self._build_prompt(market_title, research_summary)
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
            async with session.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    usage = data.get("usageMetadata", {})
                    tracker.record_call(
                        "gemini-pro",
                        usage.get("promptTokenCount", 800),
                        usage.get("candidatesTokenCount", 200),
                        source="predictor",
                    )
                    return self._parse_llm_response("gemini", text, 0.15, "full")
                logger.warning(f"Gemini API returned {resp.status}")
        except Exception as e:
            logger.error(f"Gemini prediction error: {e}")
        return None

    def _build_prompt(self, market_title: str, research_summary: str) -> str:
        return (
            "You are a prediction market analyst. Estimate the probability of the "
            "following event occurring. Respond with ONLY a JSON object:\n"
            '{"probability": 0.XX, "confidence": 0.XX, "reasoning": "brief reason"}\n\n'
            f"Event: {market_title}\n\n"
            f"Research Summary:\n{research_summary}\n\n"
            "Probability (0.0 to 1.0):"
        )

    def _parse_llm_response(self, model_name: str, text: str, weight: float,
                             tier: str) -> Optional[ModelPrediction]:
        try:
            text = text.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                prob = float(data.get("probability", 0.5))
                conf = float(data.get("confidence", 0.5))
                reasoning = data.get("reasoning", "")

                prob = max(0.05, min(0.95, prob))
                conf = max(0.0, min(1.0, conf))

                return ModelPrediction(
                    model_name=model_name,
                    probability=prob,
                    confidence=conf,
                    reasoning=reasoning,
                    weight=weight,
                    tier=tier,
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse {model_name} response: {e}")
        return None


class EnsemblePredictor:
    """
    Tiered ensemble predictor with calibration-weighted model voting.

    Routing tiers:
      - Edge < 2% (from stat model): SKIP LLMs entirely → save ~$0.04/market
      - Edge 2-4%: Haiku only (cheap confirmation) → ~$0.002/market
      - Edge >= 4%: Full ensemble (Opus + GPT-4 + Gemini) → ~$0.041/market

    Calibration weighting:
      - Each model's weight auto-adjusts based on its Brier Score track record
      - Models that predict well get more influence; poor models get down-weighted
    """

    def __init__(self):
        self.min_edge = get_setting("prediction", "min_edge") or 0.04
        self.confidence_threshold = get_setting("prediction", "confidence_threshold") or 0.65
        self.stat_model = StatisticalModel()
        self.llm_predictor = LLMPredictor()
        self.calibration = PerModelCalibration()

    async def predict(self, brief: ResearchBrief) -> TradeSignal:
        """Generate a trade signal with tiered model routing."""
        market = brief.market
        predictions: list[ModelPrediction] = []
        estimated_cost = 0.0

        # ─── Always run statistical model (free) ───
        stat_pred = self.stat_model.predict(brief)
        predictions.append(stat_pred)

        # ─── Determine tier based on statistical model's edge ───
        stat_edge = abs(stat_pred.probability - market.current_price)

        research_summary = (
            f"Sentiment: {brief.sentiment_label} ({brief.overall_sentiment:+.3f})\n"
            f"Consensus: {brief.narrative_consensus}\n"
            f"Sources: {len(brief.sources)} analyzed\n"
            f"Current market price: {brief.market_price:.3f}"
        )

        if stat_edge < TIER_SKIP:
            # ─── SKIP TIER: No edge worth investigating ───
            tier_used = "skip"
            logger.debug(f"SKIP: {market.title[:40]} (stat edge {stat_edge:.3f} < {TIER_SKIP})")

        elif stat_edge < TIER_CHEAP:
            # ─── CHEAP TIER: Confirm with Haiku only ───
            tier_used = "cheap"
            async with aiohttp.ClientSession() as session:
                haiku_pred = await self.llm_predictor.predict_with_haiku(
                    session, market.title, research_summary
                )
                if haiku_pred:
                    predictions.append(haiku_pred)
                    estimated_cost += LLMPredictor.COST_PER_CALL["haiku"]
            logger.info(f"CHEAP: {market.title[:40]} (stat edge {stat_edge:.3f})")

        else:
            # ─── FULL TIER: Deploy all models ───
            tier_used = "full"
            async with aiohttp.ClientSession() as session:
                llm_results = await asyncio.gather(
                    self.llm_predictor.predict_with_claude(session, market.title, research_summary),
                    self.llm_predictor.predict_with_gpt4(session, market.title, research_summary),
                    self.llm_predictor.predict_with_gemini(session, market.title, research_summary),
                    return_exceptions=True,
                )

                for result in llm_results:
                    if isinstance(result, ModelPrediction):
                        predictions.append(result)
                        estimated_cost += LLMPredictor.COST_PER_CALL.get(result.model_name, 0.01)

            logger.info(f"FULL: {market.title[:40]} (stat edge {stat_edge:.3f}, {len(predictions)} models)")

        # ─── Apply calibration-weighted ensemble ───
        model_names = [p.model_name for p in predictions]
        calibrated_weights = self.calibration.get_calibrated_weights(model_names)

        # Override weights with calibration-adjusted ones
        for pred in predictions:
            if pred.model_name in calibrated_weights:
                pred.weight = calibrated_weights[pred.model_name]

        ensemble_prob = self._weighted_average(predictions)
        confidence = self._calculate_confidence(predictions)

        # Calculate edge and expected value
        edge = ensemble_prob - market.current_price
        ev = self._expected_value(ensemble_prob, market.current_price)
        z_score = self._mispricing_z_score(predictions, market.current_price)

        # Determine signal
        signal = "no_trade"
        if abs(edge) >= self.min_edge and confidence >= self.confidence_threshold:
            if edge > 0:
                signal = "buy_yes"
            elif edge < -self.min_edge:
                signal = "buy_no"

        trade_signal = TradeSignal(
            market=market,
            research=brief,
            predicted_probability=ensemble_prob,
            market_price=market.current_price,
            edge=edge,
            expected_value=ev,
            mispricing_z_score=z_score,
            confidence=confidence,
            model_predictions=predictions,
            signal=signal,
            tier_used=tier_used,
            estimated_cost=estimated_cost,
        )

        logger.info(f"Prediction: {trade_signal.summary()}")
        return trade_signal

    async def predict_batch(self, briefs: list[ResearchBrief]) -> list[TradeSignal]:
        """Generate trade signals for multiple markets."""
        signals = []
        total_cost = 0.0
        tier_counts = {"skip": 0, "cheap": 0, "full": 0}

        for brief in briefs:
            signal = await self.predict(brief)
            signals.append(signal)
            total_cost += signal.estimated_cost
            tier_counts[signal.tier_used] = tier_counts.get(signal.tier_used, 0) + 1

        logger.info(
            f"Batch complete: {len(signals)} markets | "
            f"Tiers: {tier_counts} | Est. cost: ${total_cost:.3f}"
        )
        return signals

    def _weighted_average(self, predictions: list[ModelPrediction]) -> float:
        if not predictions:
            return 0.5
        total_weight = sum(p.weight for p in predictions)
        if total_weight == 0:
            return sum(p.probability for p in predictions) / len(predictions)
        weighted_sum = sum(p.probability * p.weight for p in predictions)
        return weighted_sum / total_weight

    def _calculate_confidence(self, predictions: list[ModelPrediction]) -> float:
        if len(predictions) < 2:
            return predictions[0].confidence if predictions else 0.0

        probs = [p.probability for p in predictions]
        std = float(np.std(probs))
        agreement = max(0, 1.0 - std * 3)

        avg_confidence = sum(p.confidence * p.weight for p in predictions) / max(
            sum(p.weight for p in predictions), 0.01
        )

        model_count_factor = min(len(predictions) / 4, 1.0)
        return agreement * 0.4 + avg_confidence * 0.4 + model_count_factor * 0.2

    def _expected_value(self, p_model: float, p_market: float) -> float:
        if p_market <= 0 or p_market >= 1:
            return 0.0
        b = (1.0 / p_market) - 1.0
        return p_model * b - (1.0 - p_model)

    def _mispricing_z_score(self, predictions: list[ModelPrediction], market_price: float) -> float:
        if len(predictions) < 2:
            return 0.0
        probs = [p.probability for p in predictions]
        mean = float(np.mean(probs))
        std = float(np.std(probs))
        if std < 0.01:
            std = 0.01
        return (mean - market_price) / std


class CalibrationTracker:
    """Tracks overall Brier Score and calibration over time."""

    def __init__(self, history_file: str = "data/calibration_history.json"):
        self.history_file = history_file
        self.predictions: list[dict] = []

    def record(self, predicted: float, outcome: float):
        self.predictions.append({
            "predicted": predicted,
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def brier_score(self) -> float:
        if not self.predictions:
            return 1.0
        n = len(self.predictions)
        total = sum((p["predicted"] - p["outcome"]) ** 2 for p in self.predictions)
        return total / n

    def calibration_report(self) -> str:
        n = len(self.predictions)
        if n == 0:
            return "No predictions recorded yet."
        bs = self.brier_score()
        wins = sum(1 for p in self.predictions if p["outcome"] == 1)
        return (
            f"Calibration Report ({n} predictions)\n"
            f"  Brier Score: {bs:.4f} (target < 0.25)\n"
            f"  Outcomes: {wins} yes / {n - wins} no"
        )


async def run_predictions(briefs: list[ResearchBrief]) -> list[TradeSignal]:
    """Entry point for the prediction step."""
    predictor = EnsemblePredictor()
    signals = await predictor.predict_batch(briefs)

    tradeable = [s for s in signals if s.signal != "no_trade"]
    print(f"\n=== Prediction Results ({len(tradeable)} trade signals from {len(signals)} markets) ===\n")
    for s in signals:
        print(s.summary())
        print()

    return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Predictor module - run via main.py pipeline")
