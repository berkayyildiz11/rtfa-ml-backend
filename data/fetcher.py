import httpx
import asyncio
import os, sys
import json
from datetime import datetime, timedelta, timezone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from src.utils.news_filters import analyze_article_relevance

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

BASE_URL = "https://finnhub.io/api/v1/company-news"

TICKERS = [ 
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "TSLA", "NVDA", "NFLX", "ADBE", "INTC",
    "CSCO", "PEP", "AVGO", "TXN", "QCOM", 
    "COST", "TMUS", "AMGN", "SBUX", "ISRG"
]

DEFAULT_MAX_ITEMS = 80
DEFAULT_LOOKBACK_DAYS = 7
MAX_CONCURRENT_NEWS_REQUESTS = int(os.getenv("NEWS_MAX_CONCURRENT_REQUESTS", "10"))
USE_FINBERT_FOR_NEWS = os.getenv("USE_FINBERT_FOR_NEWS", "false").lower() == "true"

POSITIVE_TERMS = {
    "beat", "beats", "strong", "surge", "surges", "rise", "rises", "gain", "gains",
    "upgrade", "upgraded", "bullish", "growth", "record", "profit", "profits",
    "outperform", "positive", "raises", "raised", "higher", "optimistic",
}
NEGATIVE_TERMS = {
    "miss", "misses", "weak", "fall", "falls", "drop", "drops", "downgrade",
    "downgraded", "bearish", "loss", "losses", "lawsuit", "probe", "warning",
    "cuts", "cut", "lower", "negative", "concern", "concerns", "slump",
}

# 1. async def yapıldı ve dışarıdan 'client' parametresi alacak şekilde güncellendi
async def fetch_daily_news(client: httpx.AsyncClient, ticker: str, lookback_days: int = 1) -> list:
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days = lookback_days)

    params = {
        'symbol': ticker,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'token': FINNHUB_API_KEY
    }

    # 2. İstek asenkron olarak (await) atılıyor
    try:
        response = await client.get(BASE_URL, params=params, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        print(f"Error Fetching {ticker}: {exc.response.status_code}")
    except httpx.HTTPError as exc:
        print(f"Error Fetching {ticker}: {exc}")

    return []


def _format_timestamp(raw_timestamp) -> str | None:
    if raw_timestamp is None:
        return None

    try:
        return datetime.fromtimestamp(raw_timestamp, tz=timezone.utc).isoformat()
    except (OSError, TypeError, ValueError):
        return None


def _news_key(item: dict) -> str:
    return f"{item['url']}-{item['timestamp']}-{item['ticker']}"


def _dedupe_and_sort_news(items: list[dict]) -> list[dict]:
    unique_news = {}
    for item in items:
        unique_key = _news_key(item)
        if unique_key not in unique_news:
            unique_news[unique_key] = item

    sorted_items = list(unique_news.values())
    sorted_items.sort(key=lambda x: x["timestamp"], reverse=True)
    return sorted_items


def _limit_news_by_ticker(items: list[dict], max_items: int | None) -> list[dict]:
    if max_items is None or len(items) <= max_items:
        return items

    grouped_items = {ticker: [] for ticker in TICKERS}
    for item in items:
        grouped_items.setdefault(item["ticker"], []).append(item)

    selected = []
    seen_keys = set()
    while len(selected) < max_items:
        added_this_round = False
        for ticker in TICKERS:
            ticker_items = grouped_items.get(ticker, [])
            if not ticker_items:
                continue

            item = ticker_items.pop(0)
            unique_key = _news_key(item)
            if unique_key in seen_keys:
                continue

            selected.append(item)
            seen_keys.add(unique_key)
            added_this_round = True

            if len(selected) >= max_items:
                break

        if not added_this_round:
            break

    selected.sort(key=lambda x: x["timestamp"], reverse=True)
    return selected


def _lightweight_sentiment(headline: str, summary: str = "") -> dict:
    text = f"{headline} {summary}".lower()
    words = {
        word.strip(".,:;!?()[]{}\"'")
        for word in text.split()
    }
    positive_hits = len(words & POSITIVE_TERMS)
    negative_hits = len(words & NEGATIVE_TERMS)
    net_score = positive_hits - negative_hits

    if net_score > 0:
        score = min(0.75, 0.25 + net_score * 0.15)
        label = "positive"
    elif net_score < 0:
        score = max(-0.75, -0.25 + net_score * 0.15)
        label = "negative"
    else:
        score = 0.0
        label = "neutral"

    confidence = min(0.75, 0.45 + abs(net_score) * 0.1)
    return {
        "sentiment_score": round(score, 4),
        "sentiment_label": label,
        "sentiment_confidence": round(confidence, 4),
    }


def _build_sentiment_analyzer():
    if not USE_FINBERT_FOR_NEWS:
        return None

    from src.models.nlp_sentiment import FinBERTSentiment

    return FinBERTSentiment()


def _analyze_news_sentiment(analyzer, headline: str, summary: str) -> dict:
    if analyzer is None:
        return _lightweight_sentiment(headline, summary)
    return analyzer.analyze_article(headline, summary)


# 3. Ana pipeline fonksiyonu async yapıldı
async def run_news_pipeline(
    max_items: int | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
):
    max_items = DEFAULT_MAX_ITEMS if max_items is None else max_items
    all_news_data = []
    print("Starting News Fetch Pipeline...")
    sentiment_analyzer = _build_sentiment_analyzer()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_NEWS_REQUESTS)

    async def fetch_for_ticker(ticker: str):
        async with semaphore:
            print(f"Fetching news for {ticker}...")
            news_items = await fetch_daily_news(client, ticker, lookback_days=lookback_days)
            return ticker, news_items

    async with httpx.AsyncClient() as client:
        fetch_results = await asyncio.gather(
            *(fetch_for_ticker(ticker) for ticker in TICKERS)
        )

        for ticker, news_items in fetch_results:
            if not news_items:
                continue

            for item in news_items:
                headline = item.get("headline") or ""
                summary = item.get("summary") or ""
                timestamp = _format_timestamp(item.get("datetime"))

                if not headline or not timestamp:
                    continue

                relevance = analyze_article_relevance(ticker, headline, summary)
                if not relevance["is_relevant"]:
                    continue

                sentiment = _analyze_news_sentiment(sentiment_analyzer, headline, summary)
                clean_item = {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "headline": headline,
                    "summary": summary,
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    **sentiment,
                    "relevance_score": relevance["relevance_score"],
                    "relevance_reason": relevance["relevance_reason"],
                    "matched_aliases": relevance["matched_aliases"],
                }
                all_news_data.append(clean_item)

    all_news_data = _limit_news_by_ticker(
        _dedupe_and_sort_news(all_news_data),
        max_items=max_items,
    )

    print(f"\nPipeline complete. Fetched and verified {len(all_news_data)} highly relevant articles.")

    # JSON Kaydetme İşlemi
    os.makedirs("data/local_storage", exist_ok=True)
    with open("data/local_storage/latest_news.json", "w") as f:
        json.dump(all_news_data, f, indent=4)

    return all_news_data

# 7. Dosya doğrudan çalıştırıldığında hata vermemesi için asyncio.run eklendi
if __name__ == "__main__":
    latest_news = asyncio.run(run_news_pipeline())

    if latest_news:
        print("\nSample output for news:")
        print(latest_news[0])
