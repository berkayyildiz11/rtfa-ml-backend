# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# Allow your Next.js frontend (usually localhost:3000) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/news")
def get_latest_news():
    """Reads the local JSON file and serves it to Next.js"""
    try:
        with open("data/local_storage/latest_news.json", "r") as f:
            news_data = json.load(f)
        return {"status": "success", "total": len(news_data), "data": news_data}
    except FileNotFoundError:
        return {"status": "error", "message": "News data not found. Run fetcher.py first."}