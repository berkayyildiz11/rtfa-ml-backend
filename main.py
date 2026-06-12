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

from src.services.prediction_service import (
    build_prediction_response_async,
    build_recent_sentiment_signal,
)
from src.services.investor_agent import (
    get_agent_history,
    get_agent_status,
    liquidate_agent_run,
    run_daily_agent_cycle,
    start_agent_run,
)
from src.services.investor_agent_v2 import (
    get_agent_v2_history,
    get_agent_v2_status,
    liquidate_agent_v2_run,
    run_daily_agent_v2_cycle,
    start_agent_v2_run,
)

# .env dosyasını yükle
load_dotenv()

# --- 1. AYARLAR VE KESİN AYRILMIŞ ANAHTARLAR ---
MONGO_URI = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")

# İki anahtarı birbirinden tamamen bağımsız değişkenlere eşitleyelim
STOCK_KEY = os.environ.get("FINNHUB_API_KEY2")
NEWS_KEY = os.environ.get("FINNHUB_API_KEY")

# Arka plan veri çekme işlemini açıp kapatmak için bir bayrak (Deploy için varsayılanı "true" yaptık) (Local testler için "false" yapın) ("ENABLE_POLLER" bunu yanındaki değeri.)
ENABLE_POLLER = os.environ.get("ENABLE_POLLER", "true").lower() == "true"
ENABLE_INVESTOR_AGENT = os.environ.get("ENABLE_INVESTOR_AGENT", "true").lower() == "true"
ENABLE_INVESTOR_AGENT_V2 = os.environ.get("ENABLE_INVESTOR_AGENT_V2", "true").lower() == "true"
INVESTOR_AGENT_CHECK_INTERVAL_SECONDS = int(os.environ.get("INVESTOR_AGENT_CHECK_INTERVAL_SECONDS", "3600"))
INVESTOR_AGENT_INITIAL_DELAY_SECONDS = int(os.environ.get("INVESTOR_AGENT_INITIAL_DELAY_SECONDS", "30"))

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


async def run_investor_agent_scheduler():
    """Runs the paper-trading agent once per calendar day while a run is active."""
    await asyncio.sleep(INVESTOR_AGENT_INITIAL_DELAY_SECONDS)
    while True:
        try:
            active_run = await db.agent_runs.find_one({"status": "active"}, {"_id": 0})
            if active_run:
                result = await run_daily_agent_cycle(
                    db,
                    tickers=STOCKS,
                    build_prediction_bundle=build_agent_prediction_bundle,
                )
                if result.get("status") not in {"already_ran_today", "success", "completed"}:
                    print(f"Investor agent cycle skipped: {result}")
        except Exception as exc:
            print(f"Investor agent scheduler error: {exc}")

        await asyncio.sleep(INVESTOR_AGENT_CHECK_INTERVAL_SECONDS)


async def run_investor_agent_v2_scheduler():
    """Runs the v2 paper-trading agent once per calendar day while a run is active."""
    await asyncio.sleep(INVESTOR_AGENT_INITIAL_DELAY_SECONDS)
    while True:
        try:
            active_run = await db.agent_v2_runs.find_one({"status": "active"}, {"_id": 0})
            if active_run:
                result = await run_daily_agent_v2_cycle(
                    db,
                    tickers=STOCKS,
                    build_prediction_bundle=build_agent_prediction_bundle,
                )
                if result.get("status") not in {"already_ran_today", "success", "completed"}:
                    print(f"Investor agent v2 cycle skipped: {result}")
        except Exception as exc:
            print(f"Investor agent v2 scheduler error: {exc}")

        await asyncio.sleep(INVESTOR_AGENT_CHECK_INTERVAL_SECONDS)

# --- 3. FASTAPI YAŞAM DÖNGÜSÜ (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = None
    news_task = None
    investor_agent_task = None
    investor_agent_v2_task = None
    if ENABLE_POLLER:
        worker_task = asyncio.create_task(poll_stocks_every_50_seconds())
    if ENABLE_INVESTOR_AGENT:
        investor_agent_task = asyncio.create_task(run_investor_agent_scheduler())
    if ENABLE_INVESTOR_AGENT_V2:
        investor_agent_v2_task = asyncio.create_task(run_investor_agent_v2_scheduler())
    warm_news_cache_from_disk()
    news_task = asyncio.create_task(refresh_news_every_hour())
    yield
    if worker_task:
        worker_task.cancel()
    if news_task:
        news_task.cancel()
    if investor_agent_task:
        investor_agent_task.cancel()
    if investor_agent_v2_task:
        investor_agent_v2_task.cancel()
    if NEWS_REFRESH_TASK:
        NEWS_REFRESH_TASK.cancel()

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
NEWS_CACHE_TARGET_ITEMS = 80
NEWS_REFRESH_INTERVAL_SECONDS = int(os.environ.get("NEWS_REFRESH_INTERVAL_SECONDS", "3600"))
NEWS_LOOKBACK_DAYS = int(os.environ.get("NEWS_LOOKBACK_DAYS", "7"))
NEWS_INITIAL_REFRESH_DELAY_SECONDS = int(os.environ.get("NEWS_INITIAL_REFRESH_DELAY_SECONDS", "15"))
NEWS_CACHE_TTL = timedelta(seconds=NEWS_REFRESH_INTERVAL_SECONDS)
NEWS_STORAGE_PATH = Path("data/local_storage/latest_news.json")
NEWS_REFRESH_TASK = None
NEWS_MIN_RELEVANCE_SCORE = float(os.environ.get("NEWS_MIN_RELEVANCE_SCORE", "0.55"))
PREDICTION_SENTIMENT_CACHE = {
    "data": {},
    "last_updated": None,
}

# Haber fetcher importunu fonksiyonun hemen üzerinde yapalım ki iç içe geçmesin
from data.fetcher import run_news_pipeline


def load_news_from_disk() -> list[dict]:
    if not NEWS_STORAGE_PATH.exists():
        return []

    try:
        with NEWS_STORAGE_PATH.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    return data if isinstance(data, list) else []


def warm_news_cache_from_disk():
    disk_news = load_news_from_disk()
    if disk_news:
        NEWS_CACHE["data"] = disk_news
        NEWS_CACHE["last_updated"] = datetime.now(timezone.utc)
        refresh_prediction_sentiment_cache_from_news(
            disk_news,
            now=NEWS_CACHE["last_updated"],
            use_finbert=False,
        )


async def refresh_news_cache():
    global NEWS_REFRESH_TASK

    try:
        NEWS_CACHE["data"] = await run_news_pipeline(
            max_items=NEWS_CACHE_TARGET_ITEMS,
            lookback_days=NEWS_LOOKBACK_DAYS,
        )
        NEWS_CACHE["last_updated"] = datetime.now(timezone.utc)
        try:
            await asyncio.to_thread(
                refresh_prediction_sentiment_cache_from_news,
                NEWS_CACHE["data"],
                NEWS_CACHE["last_updated"],
                True,
            )
        except Exception as exc:
            print(f"FinBERT prediction sentiment cache refresh failed: {exc}")
            await asyncio.to_thread(
                refresh_prediction_sentiment_cache_from_news,
                NEWS_CACHE["data"],
                NEWS_CACHE["last_updated"],
                False,
            )
    finally:
        NEWS_REFRESH_TASK = None


def refresh_prediction_sentiment_cache_from_news(
    news_items: list[dict],
    now: datetime,
    use_finbert: bool = True,
):
    cache_data = {}
    for ticker in STOCKS:
        cache_data[ticker] = {
            "1d": build_recent_sentiment_signal(
                ticker=ticker,
                period="1d",
                news_items=news_items,
                now=now,
                use_finbert=use_finbert,
            ),
            "1w": build_recent_sentiment_signal(
                ticker=ticker,
                period="1w",
                news_items=news_items,
                now=now,
                use_finbert=use_finbert,
            ),
        }

    PREDICTION_SENTIMENT_CACHE["data"] = cache_data
    PREDICTION_SENTIMENT_CACHE["last_updated"] = now


def get_cached_prediction_sentiment(ticker: str, period: str) -> dict | None:
    if period not in {"1d", "1w"}:
        return None

    ticker_cache = PREDICTION_SENTIMENT_CACHE["data"].get(ticker.upper(), {})
    signal = ticker_cache.get(period)
    return signal if isinstance(signal, dict) else None


def rank_news_for_display(news_items: list[dict]) -> list[dict]:
    ranked_items = [
        item
        for item in news_items
        if _news_relevance(item) >= NEWS_MIN_RELEVANCE_SCORE
    ]
    ranked_items.sort(
        key=lambda item: (
            _news_effect_score(item),
            _parse_news_timestamp(item.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    return ranked_items


def _news_effect_score(item: dict) -> float:
    relevance = _news_relevance(item)
    confidence = _float_value(item.get("sentiment_confidence"), default=0.45)
    sentiment_strength = abs(_float_value(item.get("sentiment_score"), default=0.0))
    return relevance * (0.50 + confidence * 0.50) * (0.60 + sentiment_strength * 0.40)


def _news_relevance(item: dict) -> float:
    return _float_value(item.get("relevance_score"), default=1.0)


def _float_value(value, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def _parse_news_timestamp(value) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def ensure_news_refresh_started():
    global NEWS_REFRESH_TASK

    if NEWS_REFRESH_TASK is None or NEWS_REFRESH_TASK.done():
        NEWS_REFRESH_TASK = asyncio.create_task(refresh_news_cache())


async def refresh_news_every_hour():
    await asyncio.sleep(NEWS_INITIAL_REFRESH_DELAY_SECONDS)
    while True:
        ensure_news_refresh_started()
        await asyncio.sleep(NEWS_REFRESH_INTERVAL_SECONDS)

@app.get("/api/news")
async def get_latest_news(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
):
    now = datetime.now(timezone.utc)

    cache_expired = (
        NEWS_CACHE["last_updated"] is None or
        now - NEWS_CACHE["last_updated"] > NEWS_CACHE_TTL
    )

    if not NEWS_CACHE["data"]:
        disk_news = load_news_from_disk()
        if disk_news:
            NEWS_CACHE["data"] = disk_news
            NEWS_CACHE["last_updated"] = now

    if cache_expired:
        ensure_news_refresh_started()

    if not NEWS_CACHE["data"]:
        ensure_news_refresh_started()
        NEWS_CACHE["last_updated"] = now

    news_data = rank_news_for_display(NEWS_CACHE["data"])
    total = len(news_data)
    start = (page - 1) * limit
    end = start + limit

    return {
        "status": "success",
        "page": page,
        "limit": limit,
        "total": total,
        "totalPages": (total + limit - 1) // limit,
        "refreshing": NEWS_REFRESH_TASK is not None and not NEWS_REFRESH_TASK.done(),
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
        using_realtime_series = False
        if period == "1d":
            query = {"symbol": ticker, "date": {"$gte": start_date}}
            using_realtime_series = True
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

        # Günlük geçmiş veri son 1 haftayı kapsamıyorsa, 1w grafiğini anlık veriden doldur
        if not results and period == "1w":
            query = {"symbol": ticker, "date": {"$gte": start_date}}
            cursor = trades_col.find(query).sort("date", 1)
            results = await cursor.to_list(length=100000)
            using_realtime_series = True

        if period == "1w" and using_realtime_series:
            formatted_data = build_daily_points_from_realtime(results)
            return {"status": "success", "ticker": ticker, "period": period, "data": formatted_data}

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
        if period != "1d" and not using_realtime_series:
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


def build_daily_points_from_realtime(results: list[dict]) -> list[dict]:
    buckets = {}

    for doc in results:
        price = doc.get("price")
        date = doc.get("date")
        if price is None or date is None:
            continue

        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        if math.isnan(price):
            continue

        day = date.date().isoformat() if hasattr(date, "date") else str(date).split("T")[0].split(" ")[0]
        bucket = buckets.setdefault(
            day,
            {
                "date": date,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            },
        )
        bucket["date"] = date
        bucket["high"] = max(bucket["high"], price)
        bucket["low"] = min(bucket["low"], price)
        bucket["close"] = price

    formatted_data = []
    for bucket in buckets.values():
        date = bucket["date"]
        formatted_data.append(
            {
                "date": date.isoformat() if hasattr(date, "isoformat") else date,
                "price": bucket["close"],
                "open": bucket["open"],
                "high": bucket["high"],
                "low": bucket["low"],
                "close": bucket["close"],
            }
        )

    return sorted(formatted_data, key=lambda point: point["date"])
    

def load_recent_news_for_prediction() -> list[dict]:
    if NEWS_CACHE["data"]:
        return NEWS_CACHE["data"]

    return load_news_from_disk()


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

        return await build_prediction_response_async(
            ticker=ticker,
            period=period,
            historical_prices=historical_prices,
            sp500_prices=sp500_prices,
            latest_realtime=latest_realtime,
            news_items=load_recent_news_for_prediction(),
            precomputed_sentiment_signal=get_cached_prediction_sentiment(ticker, period),
            explain=explain,
            now=now,
            use_finbert_for_prediction=False,
        )
    except Exception as e:
        return {"status": "error", "ticker": ticker, "period": period, "message": str(e)}


async def build_agent_prediction_bundle(ticker: str) -> dict[str, dict[str, object]]:
    return {
        "1d": await build_agent_prediction(ticker, "1d"),
        "1w": await build_agent_prediction(ticker, "1w"),
    }


async def build_agent_prediction(
    ticker: str,
    period: Literal["1d", "1w"],
) -> dict[str, object]:
    ticker = ticker.upper()
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=1095)

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

    return await build_prediction_response_async(
        ticker=ticker,
        period=period,
        historical_prices=historical_prices,
        sp500_prices=sp500_prices,
        latest_realtime=latest_realtime,
        news_items=load_recent_news_for_prediction(),
        precomputed_sentiment_signal=get_cached_prediction_sentiment(ticker, period),
        explain=False,
        now=now,
        use_finbert_for_prediction=False,
    )


async def lookup_agent_liquidation_price(ticker: str) -> float | None:
    bundle = await build_agent_prediction_bundle(ticker)
    prediction = bundle.get("1d") or bundle.get("1w")
    if not prediction:
        return None
    return float(prediction["latest_price"])


@app.post("/api/agent/start")
async def start_investor_agent():
    if not ENABLE_INVESTOR_AGENT:
        return {"status": "disabled", "message": "Investor agent is disabled by ENABLE_INVESTOR_AGENT."}

    start_result = await start_agent_run(db)
    if start_result["status"] != "started":
        return start_result

    first_cycle = await run_daily_agent_cycle(
        db,
        tickers=STOCKS,
        build_prediction_bundle=build_agent_prediction_bundle,
    )
    return {
        **start_result,
        "first_cycle": first_cycle,
    }


@app.post("/api/agent/run-daily-cycle")
async def run_investor_agent_daily_cycle(
    force: bool = Query(False, description="Allow another decision cycle for today's date"),
):
    if not ENABLE_INVESTOR_AGENT:
        return {"status": "disabled", "message": "Investor agent is disabled by ENABLE_INVESTOR_AGENT."}

    return await run_daily_agent_cycle(
        db,
        tickers=STOCKS,
        build_prediction_bundle=build_agent_prediction_bundle,
        force=force,
    )


@app.post("/api/agent/liquidate")
async def liquidate_investor_agent(
    reason: str = Query("manual_liquidation"),
):
    return await liquidate_agent_run(
        db,
        price_lookup=lookup_agent_liquidation_price,
        reason=reason,
    )


@app.get("/api/agent/status")
async def get_investor_agent_status():
    return await get_agent_status(db)


@app.get("/api/agent/history")
async def get_investor_agent_history(
    limit: int = Query(100, ge=1, le=1000),
):
    return await get_agent_history(db, limit=limit)


@app.post("/api/agent-v2/start")
async def start_investor_agent_v2():
    if not ENABLE_INVESTOR_AGENT_V2:
        return {"status": "disabled", "message": "Investor agent v2 is disabled by ENABLE_INVESTOR_AGENT_V2."}

    start_result = await start_agent_v2_run(db)
    if start_result["status"] != "started":
        return start_result

    first_cycle = await run_daily_agent_v2_cycle(
        db,
        tickers=STOCKS,
        build_prediction_bundle=build_agent_prediction_bundle,
    )
    return {
        **start_result,
        "first_cycle": first_cycle,
    }


@app.post("/api/agent-v2/run-daily-cycle")
async def run_investor_agent_v2_daily_cycle(
    force: bool = Query(False, description="Allow another decision cycle for today's date"),
):
    if not ENABLE_INVESTOR_AGENT_V2:
        return {"status": "disabled", "message": "Investor agent v2 is disabled by ENABLE_INVESTOR_AGENT_V2."}

    return await run_daily_agent_v2_cycle(
        db,
        tickers=STOCKS,
        build_prediction_bundle=build_agent_prediction_bundle,
        force=force,
    )


@app.post("/api/agent-v2/liquidate")
async def liquidate_investor_agent_v2(
    reason: str = Query("manual_liquidation"),
):
    return await liquidate_agent_v2_run(
        db,
        price_lookup=lookup_agent_liquidation_price,
        reason=reason,
    )


@app.get("/api/agent-v2/status")
async def get_investor_agent_v2_status():
    return await get_agent_v2_status(db)


@app.get("/api/agent-v2/history")
async def get_investor_agent_v2_history(
    limit: int = Query(100, ge=1, le=1000),
):
    return await get_agent_v2_history(db, limit=limit)
