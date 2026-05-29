# main.py - Güncel ve İzole Edilmiş Hali

import os
import asyncio
import httpx
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# --- 1. AYARLAR VE KESİN AYRILMIŞ ANAHTARLAR ---
MONGO_URI = os.environ.get("MONGO_URI")

# İki anahtarı birbirinden tamamen bağımsız değişkenlere eşitleyelim
STOCK_KEY = os.environ.get("FINNHUB_API_KEY2")
NEWS_KEY = os.environ.get("FINNHUB_API_KEY")

STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "INTC", 
    "CSCO", "ADBE", "QCOM", "PYPL", "AMD", "TMUS", "COST", "BKNG", "AMGN", "SBUX"
]

client = AsyncIOMotorClient(MONGO_URI)
db = client.stock_tracking_db
trades_col = db.sp500_datas2

# --- 2. BORSA WORKER FONKSİYONLARI ---
# DİKKAT: Artık api_key'i dışarıdan parametre olarak alıyor, global değişkenle karışamaz!
async def fetch_single_stock(http_client, symbol, api_key):
    """Tek bir hissenin verisini kendisine verilen özel anahtarla çeker."""
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
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

async def poll_stocks_every_50_seconds():
    """Hisseleri izole edilmiş STOCK_KEY ile güvenli aralıklarla çeker."""
    print("Veri çekme motoru başlatıldı...")
    
    async with httpx.AsyncClient() as http_client:
        while True:
            start_time = asyncio.get_event_loop().time()
            valid_documents = []
            
            for symbol in STOCKS:
                # KESİN ÇÖZÜM: Fonksiyona sadece borsa anahtarını (STOCK_KEY) gönderiyoruz
                doc = await fetch_single_stock(http_client, symbol, STOCK_KEY)
                if doc is not None:
                    valid_documents.append(doc)
                
                # Güvenli saniyelik limit aralığı
                await asyncio.sleep(0.3)
            
            if valid_documents:
                try:
                    await trades_col.insert_many(valid_documents)
                    print(f"{datetime.now().strftime('%H:%M:%S')} - {len(valid_documents)} hisse başarıyla kaydedildi. ✅")
                except Exception as e:
                    print(f"Veritabanına yazma hatası: {e}")
            
            elapsed_time = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, 50 - elapsed_time)
            await asyncio.sleep(sleep_time)

# --- 3. FASTAPI YAŞAM DÖNGÜSÜ (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(poll_stocks_every_50_seconds())
    yield
    worker_task.cancel()

# --- 4. FASTAPI UYGULAMASI VE CORS ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
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

# Haber fetcher importunu fonksiyonun hemen üzerinde yapalım ki iç içe geçmesin
from data.fetcher import run_news_pipeline

@app.get("/api/news")
async def get_latest_news(page: int = 1, limit: int = 10):
    now = datetime.now(timezone.utc)

    cache_expired = (
        NEWS_CACHE["last_updated"] is None or
        now - NEWS_CACHE["last_updated"] > timedelta(minutes=30)
    )

    if cache_expired:
        # Eğer fetcher.py içinde run_news_pipeline fonksiyonunu güncelleyebiliyorsan 
        # ona da NEWS_KEY'i parametre olarak paslayabilirsin. 
        # Güncelleyemiyorsan bu şekilde çağır, o zaten os.getenv("FINNHUB_NEWS_API_KEY") okuyor.
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

# --- 6. ADMİN PANELİ VE GİZLİ ROTACIKLAR ---
@app.get("/api/admin/latest")
async def get_latest_inserts(limit: int = 20):
    try:
        cursor = trades_col.find({}).sort("date", -1).limit(limit)
        results = await cursor.to_list(length=limit)
        formatted_data = [{"symbol": doc["symbol"], "price": doc["price"], "date": doc["date"].strftime("%H:%M:%S")} for doc in results]
        return {"status": "ok", "son_veriler": formatted_data}
    except Exception as e:
        return {"status": "error", "detay": str(e)}