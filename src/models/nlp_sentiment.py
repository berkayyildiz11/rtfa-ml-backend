import torch
from transformers import pipeline


POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"
NEUTRAL_LABEL = "neutral"
MIN_DIRECTIONAL_CONFIDENCE = 0.6
MAX_TEXT_CHARS = 1200


class FinBERTSentiment:
    def __init__(self):
        """
        Initializes the FinBERT model.
        """
        print("Loading FinBERT model into memory...")

        device = 0 if torch.cuda.is_available() else -1

        self.nlp = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=device,
        )
        print("FinBERT successfully loaded!")

    def _truncate_text(self, text: str) -> str:
        clean_text = " ".join((text or "").split())
        return clean_text[:MAX_TEXT_CHARS]

    def _score_text(self, text: str) -> dict:
        text = self._truncate_text(text)
        if not text:
            return {
                "score": 0.0,
                "label": NEUTRAL_LABEL,
                "confidence": 0.0,
            }

        result = self.nlp(text, truncation=True)[0]
        label = result["label"].lower()
        confidence = float(result["score"])

        if label not in {POSITIVE_LABEL, NEGATIVE_LABEL, NEUTRAL_LABEL}:
            label = NEUTRAL_LABEL

        if label == NEUTRAL_LABEL or confidence < MIN_DIRECTIONAL_CONFIDENCE:
            return {
                "score": 0.0,
                "label": NEUTRAL_LABEL,
                "confidence": round(confidence, 4),
            }

        signed_score = confidence if label == POSITIVE_LABEL else -confidence
        return {
            "score": signed_score,
            "label": label,
            "confidence": round(confidence, 4),
        }

    def _combined_label(self, score: float) -> str:
        if score > 0.1:
            return POSITIVE_LABEL
        if score < -0.1:
            return NEGATIVE_LABEL
        return NEUTRAL_LABEL

    def analyze_article(self, headline: str, summary: str = "") -> dict:
        """
        Returns normalized sentiment metadata for an article.
        """
        headline_result = self._score_text(headline)
        summary_result = self._score_text(summary)

        if summary:
            combined_score = (headline_result["score"] * 0.7) + (summary_result["score"] * 0.3)
            combined_confidence = (
                headline_result["confidence"] * 0.7
            ) + (summary_result["confidence"] * 0.3)
        else:
            combined_score = headline_result["score"]
            combined_confidence = headline_result["confidence"]

        label = self._combined_label(combined_score)
        if label == NEUTRAL_LABEL:
            combined_score = 0.0

        return {
            "sentiment_score": round(combined_score, 4),
            "sentiment_label": label,
            "sentiment_confidence": round(combined_confidence, 4),
        }

    def analyze_headline(self, text: str) -> float:
        """
        Compatibility wrapper that returns only a signed sentiment score.
        """
        return self.analyze_article(text)["sentiment_score"]


if __name__ == "__main__":
    analyzer = FinBERTSentiment()

    apple_news = {
        "ticker": "AAPL",
        "headline": "Atlassian job cuts raise the question: Is AI driving layoffs?",
        "summary": (
            "Atlassian (TEAM) plans to cut about 10% of its workforce as it shifts "
            "more resources toward artificial intelligence (AI). Investors are "
            "increasingly focused on productivity per employee, with companies like "
            "Nvidia (NVDA) and Apple (AAPL) leading the pack."
        ),
    }

    analysis = analyzer.analyze_article(apple_news["headline"], apple_news["summary"])
    score = analysis["sentiment_score"]

    if score > 0.1:
        badge = "BULLISH"
    elif score < -0.1:
        badge = "BEARISH"
    else:
        badge = "NEUTRAL"

    print(f"[{badge}] Score: {score: >7.4f} | {apple_news['headline']}")
