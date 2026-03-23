import httpx
import time
import os, sys
from datetime import datetime, timedelta, timezone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from utils.news_filters import is_article_relevant

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

if not FINNHUB_API_KEY:
    raise ValueError("API Key is not found! Please check your .env file.")

BASE_URL = "https://finnhub.io/api/v1/company-news"

TICKERS = [ 
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "TSLA", "NVDA", "NFLX", "ADBE", "INTC",
    "CSCO", "PEP", "AVGO", "TXN", "QCOM", 
    "COST", "TMUS", "AMGN", "SBUX", "ISRG"
]

def fetch_daily_news(ticker: str, lookback_days: int = 1) -> list:
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days = lookback_days)

    params = {
        'symbol': ticker,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'token': FINNHUB_API_KEY
    }

    with httpx.Client() as client:
        response = client.get(BASE_URL, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error Fetching {ticker}: {response.status_code}")
            return []

def run_news_pipeline():
    all_news_data = []
    print("Starting News Fetch Pipeline...")

    for ticker in TICKERS:
        print(f"Fetching news for {ticker}...")
        news_items = fetch_daily_news(ticker, lookback_days=1)

        for item in news_items:
            headline = item['headline']
            summary = item['summary']

            if is_article_relevant(ticker, headline, summary):
                clean_item = {
                    "ticker": ticker,
                    "timestamp": datetime.fromtimestamp(item['datetime'], tz=timezone.utc).isoformat(),
                    "headline": headline,
                    "summary": summary,
                    "source": item['source'],
                    "url": item['url']
                }
                all_news_data.append(clean_item)

        time.sleep(1)

    print(f"\nPipeline complete. Fetched and verified {len(all_news_data)} highly relevant articles.")

    import json
    import os

    os.makedirs("data/local_storage", exist_ok=True)
    with open("data/local_storage/latest_news.json", "w") as f:
        json.dump(all_news_data[:10], f, indent=4)

    return all_news_data

if __name__ == "__main__":
    latest_news = run_news_pipeline()

    if latest_news:
        print("Sample output for news:")
        print(latest_news[0])
