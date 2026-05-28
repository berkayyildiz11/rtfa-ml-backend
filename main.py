# main.py
import os
import asyncio
import httpx
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

# Haber sistemi importu (Önceki mesajda async yaptığımız dosya)
from data.fetcher import run_news_pipeline

# --- 1. AYARLAR VE VERİTABANI BAĞLANTISI ---
# NOT: Güvenliğin için şifreni ve API anahtarını sansürledim. 
# Kendi bilgisayarında çalıştırırken bu kısımlara kendi gerçek şifreni yazabilirsin, 
# ancak GitHub'a atarken .env kullanmayı unutma.
MONGO_URI = os.getenv("MONGO_URI")
FINNHUB_API_KEY2 = os.getenv("FINNHUB_API_KEY2")

STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "INTC", 
    "CSCO", "ADBE", "QCOM", "PYPL", "AMD", "TMUS", "COST", "BKNG", "AMGN", "SBUX"
]

client = AsyncIOMotorClient(MONGO_URI)
db = client.stock_tracking_db
trades_col = db.sp500deneme  # Time Series Collection

# --- 2. BORSA WORKER FONKSİYONLARI ---
async def fetch_single_stock(http_client, symbol):
    """Tek bir hissenin verisini Finnhub'dan çeker."""
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY2}"
    try:
        response = await http_client.get(url, timeout=5.0)
        data = response.json()
        
        if 'c' in data and data['c'] > 0:
            return {
                "symbol": symbol,
                "price": data['c'],
                "date": datetime.now(timezone.utc) 
            }
    except Exception as e:
        print(f"Hata: {symbol} verisi çekilemedi. Detay: {e}")
    
    return None

async def poll_stocks_every_25_seconds():
    """Her 25 saniyede bir tüm hisseleri çeker ve MongoDB'ye kaydeder."""
    print("Veri çekme motoru başlatıldı...")
    
    async with httpx.AsyncClient() as http_client:
        while True:
            start_time = asyncio.get_event_loop().time()
            
            tasks = [fetch_single_stock(http_client, symbol) for symbol in STOCKS]
            results = await asyncio.gather(*tasks)
            
            valid_documents = [doc for doc in results if doc is not None]
            
            if valid_documents:
                try:
                    await trades_col.insert_many(valid_documents)
                    print(f"{datetime.now().strftime('%H:%M:%S')} - {len(valid_documents)} hisse başarıyla kaydedildi.")
                except Exception as e:
                    print(f"Veritabanına yazma hatası: {e}")
            
            elapsed_time = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, 25 - elapsed_time)
            await asyncio.sleep(sleep_time)

# --- 3. FASTAPI YAŞAM DÖNGÜSÜ (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Uygulama başlarken arka plan görevini tetikle
    worker_task = asyncio.create_task(poll_stocks_every_25_seconds())
    yield
    # Uygulama kapanırken arka plan görevini iptal et
    worker_task.cancel()

# --- 4. FASTAPI UYGULAMASI VE CORS ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ], 
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. HABERLER ROTASI ---
NEWS_CACHE = {
    "data": [],
    "last_updated": None
}

@app.get("/api/news")
async def get_latest_news(page: int = 1, limit: int = 10):
    now = datetime.now(timezone.utc)

    cache_expired = (
        NEWS_CACHE["last_updated"] is None or
        now - NEWS_CACHE["last_updated"] > timedelta(minutes=30)
    )

    if cache_expired:
        NEWS_CACHE["data"] = await run_news_pipeline()
        NEWS_CACHE["last_updated"] = now

    news_data = NEWS_CACHE["data"]
    total = len(news_data)
    start = (page - 1) * limit
    end = start + limit

    return {
        "status": "success",
        "page": page,
        "limit": limit,
        "total": total,
        "totalPages": (total + limit - 1) // limit,
        "data": news_data[start:end],
    }

# --- 6. BORSA ROTALARI ---
@app.get("/api/admin/latest")
async def get_latest_inserts(limit: int = 20):
    """Sistemin çalışıp çalışmadığını kontrol etmek için gizli panel."""
    try:
        cursor = trades_col.find({}).sort("date", -1).limit(limit)
        results = await cursor.to_list(length=limit)
        formatted_data = [{"symbol": doc["symbol"], "price": doc["price"], "date": doc["date"].strftime("%H:%M:%S")} for doc in results]
        return {"status": "ok", "son_veriler": formatted_data}
    except Exception as e:
        return {"status": "error", "detay": str(e)}

@app.get("/api/stocks/{symbol}")
async def get_stock_history(symbol: str, timeframe: str = Query("1D")):
    """
    1D: Son 24 saatin tüm 25 saniyelik ham verilerini döndürür.
    1W, 1M, 3M, 1Y, 3Y: MongoDB aggregation ile günlük kapanış fiyatlarını döndürür.
    """
    now = datetime.now(timezone.utc)
    
    if timeframe == "1D":
        start_date = now - timedelta(days=1)
        cursor = trades_col.find({
            "symbol": symbol.upper(),
            "date": {"$gte": start_date}
        }).sort("date", 1)
        
        results = await cursor.to_list(length=None)
        formatted_data = [{"time": doc["date"].timestamp() * 1000, "price": doc["price"]} for doc in results]
        return {"symbol": symbol, "data": formatted_data}

    else:
        if timeframe == "1W":
            start_date = now - timedelta(weeks=1)
        elif timeframe == "1M":
            start_date = now - timedelta(days=30)
        elif timeframe == "3M":
            start_date = now - timedelta(days=90)
        elif timeframe == "1Y":
            start_date = now - timedelta(days=365)
        elif timeframe == "3Y":
            start_date = now - timedelta(days=365 * 3)

        pipeline = [
            {
                "$match": {
                    "symbol": symbol.upper(),
                    "date": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$date"}},
                    "close_price": {"$last": "$price"},
                    "date": {"$last": "$date"} 
                }
            },
            {
                "$sort": {"date": 1}
            }
        ]

        cursor = trades_col.aggregate(pipeline)
        results = await cursor.to_list(length=None)
        formatted_data = [{"time": doc["date"].timestamp() * 1000, "price": doc["close_price"]} for doc in results]
        return {"symbol": symbol, "data": formatted_data}