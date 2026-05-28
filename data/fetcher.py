import httpx
import asyncio
import os, sys
import json
from datetime import datetime, timedelta, timezone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from src.utils.news_filters import is_article_relevant

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

BASE_URL = "https://finnhub.io/api/v1/company-news"

TICKERS = [ 
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "TSLA", "NVDA", "NFLX", "ADBE", "INTC",
    "CSCO", "PEP", "AVGO", "TXN", "QCOM", 
    "COST", "TMUS", "AMGN", "SBUX", "ISRG"
]

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
    response = await client.get(BASE_URL, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error Fetching {ticker}: {response.status_code}")
        return []

# 3. Ana pipeline fonksiyonu async yapıldı
async def run_news_pipeline():
    all_news_data = []
    print("Starting News Fetch Pipeline...")

    # 4. Client oturumu döngünün DIŞINDA açılıyor (Çok ciddi performans artışı sağlar)
    async with httpx.AsyncClient() as client:
        for ticker in TICKERS:
            print(f"Fetching news for {ticker}...")
            
            # 5. Alt fonksiyonu await ile bekliyoruz
            news_items = await fetch_daily_news(client, ticker, lookback_days=1)

            # API'den boş dönme ihtimaline karşı güvenlik kontrolü
            if not news_items:
                continue

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

            # 6. KRİTİK DEĞİŞİKLİK: Sunucuyu donduran time.sleep() yerine asenkron bekleme
            await asyncio.sleep(1)

    print(f"\nPipeline complete. Fetched and verified {len(all_news_data)} highly relevant articles.")

    # JSON Kaydetme İşlemi
    os.makedirs("data/local_storage", exist_ok=True)
    with open("data/local_storage/latest_news.json", "w") as f:
        unique_news = {}

        for item in all_news_data:
            unique_key = f"{item['url']}-{item['timestamp']}-{item['ticker']}"

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