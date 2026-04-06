"""
Step 3: Predict - Estimate True Probability

Uses ensemble of statistical models and LLM reasoning to calibrate
true probability vs market price. Generates trade signals when edge > threshold.
"""

import json
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import aiohttp

from config import get_setting, get_env
from modules.scanner import Market
from modules.researcher import ResearchBrief

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Prediction from a single model in the ensemble."""
    model_name: str
    probability: float   # 0-1
    confidence: float    # 0-1
    reasoning: str = ""
    weight: float = 0.0


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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        return (
            f"[{self.signal.upper():8s}] {self.market.title[:55]}\n"
            f"  Model: {self.predicted_probability:.3f} vs Market: {self.market_price:.3f} | "
            f"Edge: {self.edge:+.3f} | EV: {self.expected_value:+.3f} | "
            f"Z: {self.mispricing_z_score:.2f} | Conf: {self.confidence:.2f}"
        )


class StatisticalModel:
    """Simple statistical prediction based on research data."""

    def predict(self, brief: ResearchBrief) -> ModelPrediction:
        """Use sentiment and source agreement as a statistical signal."""
        # Base from research narrative probability
        base_prob = brief.narrative_probability

        # Adjust based on confidence and source count
        source_count = len(brief.sources)
        if source_count < 3:
            # Low confidence with few sources, regress toward market price
            prob = base_prob * 0.4 + brief.market_price * 0.6
        else:
            prob = base_prob * 0.7 + brief.market_price * 0.3

        prob = max(0.05, min(0.95, prob))

        return ModelPrediction(
            model_name="statistical",
            probability=prob,
            confidence=brief.confidence * 0.8,
            reasoning=f"Based on {source_count} sources, sentiment {brief.overall_sentiment:+.2f}",
            weight=get_setting("prediction", "ensemble_models", default=[])[-1].get("weight", 0.15)
            if get_setting("prediction", "ensemble_models") else 0.15,
        )


class LLMPredictor:
    """Uses LLM APIs for probability estimation."""

    def __init__(self):
        self.model_configs = {
            "claude": {
                "url": "https://api.anthropic.com/v1/messages",
                "key_env": "ANTHROPIC_API_KEY",
                "weight": 0.20,
            },
            "gpt4": {
                "url": "https://api.openai.com/v1/chat/completions",
                "key_env": "OPENAI_API_KEY",
                "weight": 0.20,
            },
            "gemini": {
                "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                "key_env": "GOOGLE_AI_API_KEY",
                "weight": 0.15,
            },
        }

    async def predict_with_claude(self, session: aiohttp.ClientSession,
                                   market_title: str, research_summary: str) -> Optional[ModelPrediction]:
        """Get probability estimate from Claude."""
        api_key = get_env("ANTHROPIC_API_KEY")
        if not api_key:
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
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data["content"][0]["text"]
                    return self._parse_llm_response("claude", text, 0.20)
                logger.warning(f"Claude API returned {resp.status}")
        except Exception as e:
            logger.error(f"Claude prediction error: {e}")
        return None

    async def predict_with_gpt4(self, session: aiohttp.ClientSession,
                                 market_title: str, research_summary: str) -> Optional[ModelPrediction]:
        """Get probability estimate from GPT-4."""
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
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
                    return self._parse_llm_response("gpt4", text, 0.20)
                logger.warning(f"GPT-4 API returned {resp.status}")
        except Exception as e:
            logger.error(f"GPT-4 prediction error: {e}")
        return None

    async def predict_with_gemini(self, session: aiohttp.ClientSession,
                                   market_title: str, research_summary: str) -> Optional[ModelPrediction]:
        """Get probability estimate from Gemini."""
        api_key = get_env("GOOGLE_AI_API_KEY")
        if not api_key:
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
                    return self._parse_llm_response("gemini", text, 0.15)
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

    def _parse_llm_response(self, model_name: str, text: str, weight: float) -> Optional[ModelPrediction]:
        """Parse JSON probability from LLM response."""
        try:
            # Try to extract JSON from response
            text = text.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            # Find JSON object in text
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
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse {model_name} response: {e}")
        return None


class EnsemblePredictor:
    """
    Combines multiple model predictions into a single probability estimate.
    Generates trade signals when edge exceeds threshold.
    """

    def __init__(self):
        self.min_edge = get_setting("prediction", "min_edge") or 0.04
        self.confidence_threshold = get_setting("prediction", "confidence_threshold") or 0.65
        self.stat_model = StatisticalModel()
        self.llm_predictor = LLMPredictor()

    async def predict(self, brief: ResearchBrief) -> TradeSignal:
        """Generate a trade signal for a single market."""
        market = brief.market
        predictions: list[ModelPrediction] = []

        # Statistical model (always available)
        stat_pred = self.stat_model.predict(brief)
        predictions.append(stat_pred)

        # LLM predictions in parallel
        research_summary = (
            f"Sentiment: {brief.sentiment_label} ({brief.overall_sentiment:+.3f})\n"
            f"Consensus: {brief.narrative_consensus}\n"
            f"Sources: {len(brief.sources)} analyzed\n"
            f"Current market price: {brief.market_price:.3f}"
        )

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

        # Ensemble: weighted average
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
        )

        logger.info(f"Prediction: {trade_signal.summary()}")
        return trade_signal

    async def predict_batch(self, briefs: list[ResearchBrief]) -> list[TradeSignal]:
        """Generate trade signals for multiple markets."""
        signals = []
        for brief in briefs:
            signal = await self.predict(brief)
            signals.append(signal)
        return signals

    def _weighted_average(self, predictions: list[ModelPrediction]) -> float:
        """Compute weighted average probability from all models."""
        if not predictions:
            return 0.5

        total_weight = sum(p.weight for p in predictions)
        if total_weight == 0:
            return sum(p.probability for p in predictions) / len(predictions)

        weighted_sum = sum(p.probability * p.weight for p in predictions)
        return weighted_sum / total_weight

    def _calculate_confidence(self, predictions: list[ModelPrediction]) -> float:
        """Calculate overall confidence based on model agreement."""
        if len(predictions) < 2:
            return predictions[0].confidence if predictions else 0.0

        probs = [p.probability for p in predictions]
        std = float(np.std(probs))
        agreement = max(0, 1.0 - std * 3)  # High agreement = low std

        avg_confidence = sum(p.confidence * p.weight for p in predictions) / max(
            sum(p.weight for p in predictions), 0.01
        )

        # More models = higher base confidence
        model_count_factor = min(len(predictions) / 4, 1.0)

        return agreement * 0.4 + avg_confidence * 0.4 + model_count_factor * 0.2

    def _expected_value(self, p_model: float, p_market: float) -> float:
        """EV = p * b - (1 - p) where b = (1/p_market) - 1 (decimal odds minus 1)."""
        if p_market <= 0 or p_market >= 1:
            return 0.0
        b = (1.0 / p_market) - 1.0
        return p_model * b - (1.0 - p_model)

    def _mispricing_z_score(self, predictions: list[ModelPrediction], market_price: float) -> float:
        """Z-score of model consensus vs market. delta = (p_model - p_market) / std."""
        if len(predictions) < 2:
            return 0.0
        probs = [p.probability for p in predictions]
        mean = float(np.mean(probs))
        std = float(np.std(probs))
        if std < 0.01:
            std = 0.01  # Avoid division by zero
        return (mean - market_price) / std


class CalibrationTracker:
    """Tracks Brier Score and calibration over time."""

    def __init__(self, history_file: str = "data/calibration_history.json"):
        self.history_file = history_file
        self.predictions: list[dict] = []

    def record(self, predicted: float, outcome: float):
        """Record a prediction and its actual outcome (0 or 1)."""
        self.predictions.append({
            "predicted": predicted,
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def brier_score(self) -> float:
        """BS = (1/n) * sum((predicted - outcome)^2). Lower is better."""
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
