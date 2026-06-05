# main.py - Güncel ve İzole Edilmiş Hali

import os
import asyncio
import json
import math
import httpx
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

from src.services.prediction_service import build_prediction_response

# .env dosyasını yükle
load_dotenv()

# --- 1. AYARLAR VE KESİN AYRILMIŞ ANAHTARLAR ---
MONGO_URI = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")

# İki anahtarı birbirinden tamamen bağımsız değişkenlere eşitleyelim
STOCK_KEY = os.environ.get("FINNHUB_API_KEY2")
NEWS_KEY = os.environ.get("FINNHUB_API_KEY")

# Arka plan veri çekme işlemini açıp kapatmak için bir bayrak (Deploy için varsayılanı "true" yaptık) (Local testler için "false" yapın) ("ENABLE_POLLER" bunu yanındaki değeri.)
ENABLE_POLLER = os.environ.get("ENABLE_POLLER", "true").lower() == "true"

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
    worker_task = None
    if ENABLE_POLLER:
        worker_task = asyncio.create_task(poll_stocks_every_50_seconds())
    yield
    if worker_task:
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
async def get_latest_news(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
):
    now = datetime.now(timezone.utc)
    requested_items = page * limit

    cache_expired = (
        NEWS_CACHE["last_updated"] is None or
        now - NEWS_CACHE["last_updated"] > timedelta(minutes=30)
    )
    cache_missing_requested_page = len(NEWS_CACHE["data"]) < requested_items

    if cache_expired or cache_missing_requested_page:
        # Eğer fetcher.py içinde run_news_pipeline fonksiyonunu güncelleyebiliyorsan 
        # ona da NEWS_KEY'i parametre olarak paslayabilirsin. 
        # Güncelleyemiyorsan bu şekilde çağır, o zaten os.getenv("FINNHUB_NEWS_API_KEY") okuyor.
        NEWS_CACHE["data"] = await run_news_pipeline(max_items=requested_items)
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

# --- 7. CHART VERİSİ ROTASI ---
@app.get("/api/stocks/{ticker}/chart")
async def get_stock_chart_data(ticker: str, period: str = Query("1m", description="Zaman aralığı: 1d, 1w, 1m, 3m, 1y, 2y, 3y")):
    ticker = ticker.upper() # Küçük harfle istek gelirse düzelt (Örn: aapl -> AAPL)
    now = datetime.now(timezone.utc)
    
    if period == "1d":
        start_date = now - timedelta(days=1)
    elif period == "1w":
        start_date = now - timedelta(days=7)
    elif period == "1m":
        start_date = now - timedelta(days=30)
    elif period == "3m":
        start_date = now - timedelta(days=90)
    elif period == "1y":
        start_date = now - timedelta(days=365)
    elif period == "2y":
        start_date = now - timedelta(days=730)
    elif period == "3y":
        start_date = now - timedelta(days=1095)
    else:
        return {"status": "error", "message": "Geçersiz periyot"}

    try:
        # 1 günlük periyotta sadece anlık (intraday) verileri getir
        if period == "1d":
            query = {"symbol": ticker, "date": {"$gte": start_date}}
        else:
            # 1 günün üzerindeki periyotlarda geçmiş günlük verileri getir
            query = {"ticker": ticker, "date": {"$gte": start_date}}
            
        # Bazı veritabanı versiyonlarında length=None hatası almamak için güvenli bir limit veriyoruz
        cursor = trades_col.find(query).sort("date", 1)
        results = await cursor.to_list(length=100000)
        
        # Eğer sp500_datas2 boş döndüyse ve geçmiş veri istiyorsak, historical_prices koleksiyonunu da kontrol et
        if not results and period != "1d":
            cursor = db.historical_prices.find(query).sort("date", 1)
            results = await cursor.to_list(length=100000)
        
        formatted_data = []
        for doc in results:
            price = doc.get("close") if doc.get("close") is not None else doc.get("price")
            # Eğer fiyat NaN (Not a Number) ise frontend'i çökertmemesi için atla
            if price is not None and not math.isnan(price):
                formatted_data.append({
                    "date": doc["date"].isoformat() if hasattr(doc["date"], "isoformat") else doc["date"],
                    "price": price
                })

        # Eğer periyot 1 günden büyükse, grafiğin sağ ucuna en son anlık fiyatı da (real-time) ekle
        if period != "1d":
            latest_realtime = await trades_col.find_one({"symbol": ticker}, sort=[("date", -1)])
            if latest_realtime and latest_realtime.get("price") is not None:
                if not math.isnan(latest_realtime["price"]):
                    formatted_data.append({
                        "date": latest_realtime["date"].isoformat() if hasattr(latest_realtime["date"], "isoformat") else latest_realtime["date"],
                        "price": latest_realtime["price"]
                    })
            
        return {"status": "success", "ticker": ticker, "period": period, "data": formatted_data}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

def load_recent_news_for_prediction() -> list[dict]:
    if NEWS_CACHE["data"]:
        return NEWS_CACHE["data"]

    news_path = Path("data/local_storage/latest_news.json")
    if not news_path.exists():
        return []

    try:
        with news_path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


@app.get("/api/predict/{ticker}")
async def predict_stock_price(
    ticker: str,
    period: Literal["1d", "1w", "1m", "3m", "6m", "1y"] = "1m",
    explain: bool = Query(True, description="Return user-facing decision explanation"),
):
    ticker = ticker.upper()
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=1095)

    try:
        cursor = db.historical_prices.find(
            {"ticker": ticker, "date": {"$gte": start_date}},
            {"_id": 0},
        ).sort("date", 1)
        historical_prices = await cursor.to_list(length=100000)

        sp500_cursor = db.sp500_datas.find(
            {"date": {"$gte": start_date}},
            {"_id": 0},
        ).sort("date", 1)
        sp500_prices = await sp500_cursor.to_list(length=100000)

        latest_realtime = await trades_col.find_one(
            {"symbol": ticker},
            {"_id": 0},
            sort=[("date", -1)],
        )

        return build_prediction_response(
            ticker=ticker,
            period=period,
            historical_prices=historical_prices,
            sp500_prices=sp500_prices,
            latest_realtime=latest_realtime,
            news_items=load_recent_news_for_prediction(),
            explain=explain,
            now=now,
        )
    except Exception as e:
        return {"status": "error", "ticker": ticker, "period": period, "message": str(e)}
