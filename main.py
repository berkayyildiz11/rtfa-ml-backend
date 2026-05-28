# main.py
from fastapi import FastAPI
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import datetime, timezone, timedelta
from data.fetcher import run_news_pipeline
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow your Next.js frontend (usually localhost:3000) to talk to this API
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

NEWS_CACHE = {
    "data": [],
    "last_updated": None
}

@app.get("/api/news")
def get_latest_news(page: int = 1, limit: int = 10):
    now = datetime.now(timezone.utc)

    cache_expired = (
        NEWS_CACHE["last_updated"] is None or
        now - NEWS_CACHE["last_updated"] > timedelta(minutes=30)
    )

    if cache_expired:
        NEWS_CACHE["data"] = run_news_pipeline()
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