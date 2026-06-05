import httpx
import asyncio
import os, sys
import json
from datetime import datetime, timedelta, timezone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from src.models.nlp_sentiment import FinBERTSentiment
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
MAX_CONCURRENT_NEWS_REQUESTS = 5

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


# 3. Ana pipeline fonksiyonu async yapıldı
async def run_news_pipeline(
    max_items: int | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
):
    max_items = DEFAULT_MAX_ITEMS if max_items is None else max_items
    all_news_data = []
    seen_news_keys = set()
    print("Starting News Fetch Pipeline...")
    sentiment_analyzer = FinBERTSentiment()

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

                sentiment = sentiment_analyzer.analyze_article(headline, summary)
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
                unique_key = _news_key(clean_item)
                if unique_key in seen_news_keys:
                    continue

                seen_news_keys.add(unique_key)
                all_news_data.append(clean_item)

                if max_items is not None and len(all_news_data) >= max_items:
                    break

            if max_items is not None and len(all_news_data) >= max_items:
                break

    print(f"\nPipeline complete. Fetched and verified {len(all_news_data)} highly relevant articles.")

    # JSON Kaydetme İşlemi
    os.makedirs("data/local_storage", exist_ok=True)
    with open("data/local_storage/latest_news.json", "w") as f:
        unique_news = {}

        for item in all_news_data:
            unique_key = _news_key(item)

            if unique_key not in unique_news:
                unique_news[unique_key] = item

        all_news_data = list(unique_news.values())
        all_news_data.sort(key=lambda x: x["timestamp"], reverse=True)
        json.dump(all_news_data, f, indent=4)

    return all_news_data

# 7. Dosya doğrudan çalıştırıldığında hata vermemesi için asyncio.run eklendi
if __name__ == "__main__":
    latest_news = asyncio.run(run_news_pipeline())

    if latest_news:
        print("\nSample output for news:")
        print(latest_news[0])
