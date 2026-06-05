from dataclasses import dataclass
from itertools import combinations
import json
from math import factorial
from pathlib import Path
from typing import Literal


PredictionPeriod = Literal["1d", "1w", "1m", "3m", "6m", "1y"]
PredictionSignal = Literal["worse", "neutral", "better"]


SUPPORTED_PERIODS: tuple[PredictionPeriod, ...] = ("1d", "1w", "1m", "3m", "6m", "1y")
SENTIMENT_PERIODS: set[PredictionPeriod] = {"1d", "1w"}
DEFAULT_WEIGHTS_PATH = Path("saved_models/weight_adjustor/weights.json")

RECENT_MARKET_PATTERN = "recent_market_pattern"
TIME_SERIES_FORECAST = "time_series_forecast"
RECENT_NEWS_SENTIMENT = "recent_news_sentiment"
PUBLIC_SIGNAL_GROUPS = (
    RECENT_MARKET_PATTERN,
    TIME_SERIES_FORECAST,
    RECENT_NEWS_SENTIMENT,
)

SIGNAL_SCORES: dict[PredictionSignal, float] = {
    "worse": -1.0,
    "neutral": 0.0,
    "better": 1.0,
}


@dataclass(frozen=True)
class AdjustedWeights:
    lstm: float
    timesfm: float
    sentiment: float

    def as_dict(self) -> dict[str, float]:
        return {
            "lstm": self.lstm,
            "timesfm": self.timesfm,
            "sentiment": self.sentiment,
        }


@dataclass(frozen=True)
class EnsemblePrediction:
    signal: PredictionSignal
    score: float
    weights: AdjustedWeights

    def as_dict(self) -> dict[str, object]:
        return {
            "signal": self.signal,
            "score": self.score,
            "weights": self.weights.as_dict(),
        }


class WeightAdjustor:
    """Adjusts ensemble weights for LSTM, TimesFM, and recent news sentiment."""

    BASE_WEIGHTS: dict[PredictionPeriod, AdjustedWeights] = {
        "1d": AdjustedWeights(lstm=0.45, timesfm=0.35, sentiment=0.20),
        "1w": AdjustedWeights(lstm=0.40, timesfm=0.40, sentiment=0.20),
        "1m": AdjustedWeights(lstm=0.45, timesfm=0.55, sentiment=0.00),
        "3m": AdjustedWeights(lstm=0.35, timesfm=0.65, sentiment=0.00),
        "6m": AdjustedWeights(lstm=0.30, timesfm=0.70, sentiment=0.00),
        "1y": AdjustedWeights(lstm=0.65, timesfm=0.35, sentiment=0.00),
    }

    def __init__(
        self,
        neutral_threshold: float = 0.20,
        weights_path: str | Path | None = DEFAULT_WEIGHTS_PATH,
    ):
        if neutral_threshold < 0:
            raise ValueError("neutral_threshold must be non-negative")
        self.neutral_threshold = neutral_threshold
        self.base_weights = self._load_base_weights(weights_path)

    def adjust_weights(
        self,
        period: PredictionPeriod,
        *,
        lstm_confidence: float | None = None,
        timesfm_confidence: float | None = None,
        sentiment_score: float | None = None,
        sentiment_confidence: float | None = None,
        sentiment_relevance: float | None = None,
        news_count: int = 0,
    ) -> AdjustedWeights:
        self._validate_period(period)

        base = self.base_weights[period]
        lstm_weight = base.lstm * self._confidence_multiplier(lstm_confidence)
        timesfm_weight = base.timesfm * self._confidence_multiplier(timesfm_confidence)

        sentiment_weight = 0.0
        if period in SENTIMENT_PERIODS and sentiment_score is not None:
            sentiment_strength = self._sentiment_strength(
                sentiment_score=sentiment_score,
                sentiment_confidence=sentiment_confidence,
                sentiment_relevance=sentiment_relevance,
                news_count=news_count,
            )
            sentiment_weight = base.sentiment * sentiment_strength

        return self._normalize(lstm_weight, timesfm_weight, sentiment_weight)

    def combine_predictions(
        self,
        period: PredictionPeriod,
        *,
        lstm_signal: PredictionSignal,
        timesfm_signal: PredictionSignal,
        sentiment_score: float | None = None,
        lstm_confidence: float | None = None,
        timesfm_confidence: float | None = None,
        sentiment_confidence: float | None = None,
        sentiment_relevance: float | None = None,
        news_count: int = 0,
    ) -> EnsemblePrediction:
        self._validate_signal(lstm_signal)
        self._validate_signal(timesfm_signal)

        score, weights = self._score_prediction(
            period,
            lstm_signal=lstm_signal,
            timesfm_signal=timesfm_signal,
            sentiment_score=sentiment_score,
            lstm_confidence=lstm_confidence,
            timesfm_confidence=timesfm_confidence,
            sentiment_confidence=sentiment_confidence,
            sentiment_relevance=sentiment_relevance,
            news_count=news_count,
        )

        return EnsemblePrediction(
            signal=self._score_to_signal(score),
            score=round(score, 4),
            weights=weights,
        )

    def explain_prediction(
        self,
        period: PredictionPeriod,
        *,
        lstm_signal: PredictionSignal,
        timesfm_signal: PredictionSignal,
        sentiment_score: float | None = None,
        lstm_confidence: float | None = None,
        timesfm_confidence: float | None = None,
        sentiment_confidence: float | None = None,
        sentiment_relevance: float | None = None,
        news_count: int = 0,
    ) -> dict[str, object]:
        self._validate_signal(lstm_signal)
        self._validate_signal(timesfm_signal)

        final_score, weights = self._score_prediction(
            period,
            lstm_signal=lstm_signal,
            timesfm_signal=timesfm_signal,
            sentiment_score=sentiment_score,
            lstm_confidence=lstm_confidence,
            timesfm_confidence=timesfm_confidence,
            sentiment_confidence=sentiment_confidence,
            sentiment_relevance=sentiment_relevance,
            news_count=news_count,
        )
        baseline_score = self._coalition_score(period, set(), {
            RECENT_MARKET_PATTERN: {
                "signal": lstm_signal,
                "confidence": lstm_confidence,
            },
            TIME_SERIES_FORECAST: {
                "signal": timesfm_signal,
                "confidence": timesfm_confidence,
            },
            RECENT_NEWS_SENTIMENT: {
                "score": sentiment_score,
                "confidence": sentiment_confidence,
                "relevance": sentiment_relevance,
                "news_count": news_count,
            },
        })
        contributions = self._shapley_contributions(
            period,
            lstm_signal=lstm_signal,
            timesfm_signal=timesfm_signal,
            sentiment_score=sentiment_score,
            lstm_confidence=lstm_confidence,
            timesfm_confidence=timesfm_confidence,
            sentiment_confidence=sentiment_confidence,
            sentiment_relevance=sentiment_relevance,
            news_count=news_count,
        )

        public_weights = {
            RECENT_MARKET_PATTERN: weights.lstm,
            TIME_SERIES_FORECAST: weights.timesfm,
            RECENT_NEWS_SENTIMENT: weights.sentiment,
        }
        public_signals = self._public_signals(
            lstm_signal=lstm_signal,
            timesfm_signal=timesfm_signal,
            sentiment_score=sentiment_score,
            lstm_confidence=lstm_confidence,
            timesfm_confidence=timesfm_confidence,
            sentiment_confidence=sentiment_confidence,
            sentiment_relevance=sentiment_relevance,
            news_count=news_count,
            sentiment_used=period in SENTIMENT_PERIODS,
        )

        final_score = round(final_score, 4)
        baseline_score = round(baseline_score, 4)
        prediction_confidence = self._prediction_confidence(
            final_score=final_score,
            weights=public_weights,
            public_signals=public_signals,
        )

        return {
            "prediction": {
                "signal": self._score_to_signal(final_score),
                "score": final_score,
                "confidence": prediction_confidence,
            },
            "weights": public_weights,
            "signals": public_signals,
            "explanation": {
                "summary": self._build_summary(
                    self._score_to_signal(final_score),
                    period,
                    contributions,
                    public_signals,
                ),
                "baseline_score": baseline_score,
                "final_score": final_score,
                "contributions": contributions,
                "decision_details": self._decision_details(
                    period,
                    contributions,
                    public_signals,
                ),
            },
        }

    def _score_prediction(
        self,
        period: PredictionPeriod,
        *,
        lstm_signal: PredictionSignal,
        timesfm_signal: PredictionSignal,
        sentiment_score: float | None,
        lstm_confidence: float | None,
        timesfm_confidence: float | None,
        sentiment_confidence: float | None,
        sentiment_relevance: float | None,
        news_count: int,
    ) -> tuple[float, AdjustedWeights]:
        self._validate_period(period)

        weights = self.adjust_weights(
            period,
            lstm_confidence=lstm_confidence,
            timesfm_confidence=timesfm_confidence,
            sentiment_score=sentiment_score,
            sentiment_confidence=sentiment_confidence,
            sentiment_relevance=sentiment_relevance,
            news_count=news_count,
        )

        score = (
            SIGNAL_SCORES[lstm_signal] * weights.lstm
            + SIGNAL_SCORES[timesfm_signal] * weights.timesfm
            + self._clip(sentiment_score or 0.0, -1.0, 1.0) * weights.sentiment
        )

        return score, weights

    def _shapley_contributions(
        self,
        period: PredictionPeriod,
        *,
        lstm_signal: PredictionSignal,
        timesfm_signal: PredictionSignal,
        sentiment_score: float | None,
        lstm_confidence: float | None,
        timesfm_confidence: float | None,
        sentiment_confidence: float | None,
        sentiment_relevance: float | None,
        news_count: int,
    ) -> dict[str, float]:
        actual_values = {
            RECENT_MARKET_PATTERN: {
                "signal": lstm_signal,
                "confidence": lstm_confidence,
            },
            TIME_SERIES_FORECAST: {
                "signal": timesfm_signal,
                "confidence": timesfm_confidence,
            },
            RECENT_NEWS_SENTIMENT: {
                "score": sentiment_score,
                "confidence": sentiment_confidence,
                "relevance": sentiment_relevance,
                "news_count": news_count,
            },
        }
        n_groups = len(PUBLIC_SIGNAL_GROUPS)
        contributions = {}

        for group in PUBLIC_SIGNAL_GROUPS:
            other_groups = [candidate for candidate in PUBLIC_SIGNAL_GROUPS if candidate != group]
            contribution = 0.0

            for subset_size in range(n_groups):
                for subset in combinations(other_groups, subset_size):
                    subset = set(subset)
                    weight = (
                        factorial(subset_size)
                        * factorial(n_groups - subset_size - 1)
                        / factorial(n_groups)
                    )
                    with_group = subset | {group}
                    contribution += weight * (
                        self._coalition_score(period, with_group, actual_values)
                        - self._coalition_score(period, subset, actual_values)
                    )

            contributions[group] = round(contribution, 4)

        return contributions

    def _coalition_score(
        self,
        period: PredictionPeriod,
        included_groups: set[str],
        actual_values: dict[str, dict[str, object]],
    ) -> float:
        market = actual_values[RECENT_MARKET_PATTERN]
        forecast = actual_values[TIME_SERIES_FORECAST]
        sentiment = actual_values[RECENT_NEWS_SENTIMENT]

        score, _ = self._score_prediction(
            period,
            lstm_signal=market["signal"] if RECENT_MARKET_PATTERN in included_groups else "neutral",
            timesfm_signal=forecast["signal"] if TIME_SERIES_FORECAST in included_groups else "neutral",
            sentiment_score=sentiment["score"] if RECENT_NEWS_SENTIMENT in included_groups else 0.0,
            lstm_confidence=market["confidence"] if RECENT_MARKET_PATTERN in included_groups else None,
            timesfm_confidence=forecast["confidence"] if TIME_SERIES_FORECAST in included_groups else None,
            sentiment_confidence=(
                sentiment["confidence"] if RECENT_NEWS_SENTIMENT in included_groups else None
            ),
            sentiment_relevance=(
                sentiment["relevance"] if RECENT_NEWS_SENTIMENT in included_groups else None
            ),
            news_count=(
                int(sentiment["news_count"]) if RECENT_NEWS_SENTIMENT in included_groups else 0
            ),
        )
        return score

    def _public_signals(
        self,
        *,
        lstm_signal: PredictionSignal,
        timesfm_signal: PredictionSignal,
        sentiment_score: float | None,
        lstm_confidence: float | None,
        timesfm_confidence: float | None,
        sentiment_confidence: float | None,
        sentiment_relevance: float | None,
        news_count: int,
        sentiment_used: bool,
    ) -> dict[str, dict[str, object]]:
        return {
            RECENT_MARKET_PATTERN: {
                "direction": lstm_signal,
                "confidence": self._confidence_payload(lstm_confidence),
            },
            TIME_SERIES_FORECAST: {
                "direction": timesfm_signal,
                "confidence": self._confidence_payload(timesfm_confidence),
            },
            RECENT_NEWS_SENTIMENT: {
                "direction": self._sentiment_direction(sentiment_score),
                "confidence": self._confidence_payload(sentiment_confidence),
                "relevance": self._round_optional(sentiment_relevance),
                "news_count": news_count,
                "used_for_period": sentiment_used,
            },
        }

    def _build_summary(
        self,
        signal: PredictionSignal,
        period: PredictionPeriod,
        contributions: dict[str, float],
        public_signals: dict[str, dict[str, object]],
    ) -> str:
        strongest_group = self._strongest_contribution_group(contributions)
        strongest_text = self._group_summary_text(strongest_group, public_signals[strongest_group])

        summary = f"The pipeline predicts {signal} mainly because {strongest_text}."
        if period in SENTIMENT_PERIODS and contributions[RECENT_NEWS_SENTIMENT] != 0:
            summary += " Recent relevant news also affected this short-term prediction."
        elif period not in SENTIMENT_PERIODS:
            summary += " Recent news sentiment is not used for this longer prediction horizon."

        return summary

    def _decision_details(
        self,
        period: PredictionPeriod,
        contributions: dict[str, float],
        public_signals: dict[str, dict[str, object]],
    ) -> list[str]:
        details = []
        sorted_groups = sorted(
            PUBLIC_SIGNAL_GROUPS,
            key=lambda group: abs(contributions[group]),
            reverse=True,
        )

        for group in sorted_groups:
            contribution = contributions[group]
            signal = public_signals[group]
            if contribution == 0:
                continue

            direction = "upward" if contribution > 0 else "downward"
            article = "an" if direction == "upward" else "a"
            confidence_text = self._confidence_text(signal.get("confidence"))
            details.append(
                f"The {self._public_group_name(group)} had {article} {direction} influence "
                f"with {confidence_text} confidence."
            )

        market_direction = public_signals[RECENT_MARKET_PATTERN]["direction"]
        forecast_direction = public_signals[TIME_SERIES_FORECAST]["direction"]
        if market_direction != forecast_direction:
            details.append(
                "The market-pattern and broader time-series signals did not fully agree, "
                "so the final score reflects a balance between competing evidence."
            )

        if period in SENTIMENT_PERIODS:
            details.append(
                f"Because this is a {period} prediction, recent relevant news sentiment "
                "was allowed to influence the final result."
            )
        else:
            details.append(
                f"Because this is a {period} prediction, recent news sentiment was not "
                "used in the final score."
            )

        return details

    def _group_summary_text(
        self,
        group: str,
        signal: dict[str, object],
    ) -> str:
        confidence_text = self._confidence_text(signal.get("confidence"))
        direction = self._direction_phrase(signal["direction"])
        return (
            f"the {self._public_group_name(group)} {direction} "
            f"with {confidence_text} confidence"
        )

    def _direction_phrase(self, direction: object) -> str:
        if direction == "better":
            return "pointed toward a better outcome"
        if direction == "worse":
            return "pointed toward a worse outcome"
        if direction == "positive":
            return "was positive"
        if direction == "negative":
            return "was negative"
        return "was mostly neutral"

    def _public_group_name(self, group: str) -> str:
        names = {
            RECENT_MARKET_PATTERN: "recent market-pattern signal",
            TIME_SERIES_FORECAST: "broader time-series forecast signal",
            RECENT_NEWS_SENTIMENT: "recent news sentiment",
        }
        return names[group]

    def _strongest_contribution_group(self, contributions: dict[str, float]) -> str:
        return max(PUBLIC_SIGNAL_GROUPS, key=lambda group: abs(contributions[group]))

    def _confidence_text(self, confidence: object) -> str:
        if confidence is None:
            return "unspecified"
        if isinstance(confidence, dict):
            return str(confidence["label"])
        confidence = float(confidence)
        if confidence >= 0.75:
            return "high"
        if confidence >= 0.50:
            return "moderate"
        return "low"

    def _confidence_payload(self, confidence: float | None) -> dict[str, object]:
        if confidence is None:
            return {
                "score": None,
                "percentage": None,
                "label": "unspecified",
            }

        score = self._clip(confidence, 0.0, 1.0)
        return {
            "score": round(score, 4),
            "percentage": int(round(score * 100)),
            "label": self._confidence_text(score),
        }

    def _prediction_confidence(
        self,
        *,
        final_score: float,
        weights: dict[str, float],
        public_signals: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        weighted_confidence = 0.0
        weighted_total = 0.0
        for group, weight in weights.items():
            confidence = public_signals[group]["confidence"]["score"]
            if confidence is None or weight <= 0:
                continue
            weighted_confidence += float(confidence) * weight
            weighted_total += weight

        confidence_score = weighted_confidence / weighted_total if weighted_total > 0 else 0.5
        score_strength = min(1.0, abs(final_score))
        agreement_score = self._agreement_score(public_signals)
        overall_score = (
            confidence_score * 0.50
            + score_strength * 0.30
            + agreement_score * 0.20
        )

        return self._confidence_payload(overall_score)

    def _agreement_score(self, public_signals: dict[str, dict[str, object]]) -> float:
        directions = [
            public_signals[RECENT_MARKET_PATTERN]["direction"],
            public_signals[TIME_SERIES_FORECAST]["direction"],
        ]
        sentiment = public_signals[RECENT_NEWS_SENTIMENT]
        if sentiment["used_for_period"]:
            directions.append(sentiment["direction"])

        directional_scores = [self._public_direction_score(direction) for direction in directions]
        non_neutral = [score for score in directional_scores if score != 0]
        if not non_neutral:
            return 0.5

        positive = sum(1 for score in non_neutral if score > 0)
        negative = sum(1 for score in non_neutral if score < 0)
        return max(positive, negative) / len(non_neutral)

    def _public_direction_score(self, direction: object) -> int:
        if direction in {"better", "positive"}:
            return 1
        if direction in {"worse", "negative"}:
            return -1
        return 0

    def _sentiment_direction(self, sentiment_score: float | None) -> str:
        score = self._clip(sentiment_score or 0.0, -1.0, 1.0)
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    def _round_optional(self, value: float | None) -> float | None:
        if value is None:
            return None
        return round(float(value), 4)

    def _score_to_signal(self, score: float) -> PredictionSignal:
        if score > self.neutral_threshold:
            return "better"
        if score < -self.neutral_threshold:
            return "worse"
        return "neutral"

    def _sentiment_strength(
        self,
        *,
        sentiment_score: float,
        sentiment_confidence: float | None,
        sentiment_relevance: float | None,
        news_count: int,
    ) -> float:
        score_strength = abs(self._clip(sentiment_score, -1.0, 1.0))
        confidence = (
            self._clip(sentiment_confidence, 0.0, 1.0)
            if sentiment_confidence is not None
            else 1.0
        )
        relevance = (
            self._clip(sentiment_relevance, 0.0, 1.0)
            if sentiment_relevance is not None
            else 1.0
        )
        volume = self._news_volume_multiplier(news_count)

        return score_strength * confidence * relevance * volume

    def _news_volume_multiplier(self, news_count: int) -> float:
        if news_count <= 0:
            return 1.0
        return min(1.0, 0.50 + news_count * 0.10)

    def _confidence_multiplier(self, confidence: float | None) -> float:
        if confidence is None:
            return 1.0
        confidence = self._clip(confidence, 0.0, 1.0)
        return 0.50 + confidence * 0.50

    def _normalize(
        self,
        lstm_weight: float,
        timesfm_weight: float,
        sentiment_weight: float,
    ) -> AdjustedWeights:
        total = lstm_weight + timesfm_weight + sentiment_weight
        if total <= 0:
            raise ValueError("At least one adjusted weight must be positive")

        return AdjustedWeights(
            lstm=round(lstm_weight / total, 4),
            timesfm=round(timesfm_weight / total, 4),
            sentiment=round(sentiment_weight / total, 4),
        )

    def _validate_period(self, period: str) -> None:
        if period not in SUPPORTED_PERIODS:
            supported = ", ".join(SUPPORTED_PERIODS)
            raise ValueError(
                f"Unsupported prediction period '{period}'. Supported periods: {supported}"
            )

    def _validate_signal(self, signal: str) -> None:
        if signal not in SIGNAL_SCORES:
            supported = ", ".join(SIGNAL_SCORES)
            raise ValueError(
                f"Unsupported prediction signal '{signal}'. Supported signals: {supported}"
            )

    def _clip(self, value: float | None, minimum: float, maximum: float) -> float:
        if value is None:
            return minimum
        return max(minimum, min(maximum, float(value)))

    def _load_base_weights(
        self,
        weights_path: str | Path | None,
    ) -> dict[PredictionPeriod, AdjustedWeights]:
        weights = dict(self.BASE_WEIGHTS)
        if weights_path is None:
            return weights

        path = Path(weights_path)
        if not path.exists():
            return weights

        try:
            with path.open() as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return weights

        calibrated_weights = payload.get("weights", {})
        sentiment_calibrated = bool(payload.get("sentiment_calibrated", False))
        for period in SUPPORTED_PERIODS:
            period_weights = calibrated_weights.get(period)
            if not isinstance(period_weights, dict):
                continue

            try:
                sentiment_weight = float(period_weights.get("sentiment", 0.0))
                if period in SENTIMENT_PERIODS and not sentiment_calibrated:
                    sentiment_weight = self.BASE_WEIGHTS[period].sentiment

                weights[period] = AdjustedWeights(
                    lstm=float(period_weights["lstm"]),
                    timesfm=float(period_weights["timesfm"]),
                    sentiment=sentiment_weight,
                )
            except (KeyError, TypeError, ValueError):
                continue

        return weights
