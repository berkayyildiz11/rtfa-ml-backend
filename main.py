# main.py
from fastapi import FastAPI
from fastapi import Query
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
def get_latest_news(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50)
):
    try:
        with open("data/local_storage/latest_news.json", "r") as f:
            news_data = json.load(f)

        total = len(news_data)

        start = (page - 1) * limit
        end = start + limit

        paginated_news = news_data[start:end]

        return {
            "status": "success",
            "page": page,
            "limit": limit,
            "total": total,
            "totalPages": (total + limit - 1) // limit,
            "data": paginated_news
        }

    except FileNotFoundError:
        return {
            "status": "error",
            "message": "News data not found. Run fetcher.py first."
        }
    


@app.get("/health")
def health():
    return {"status": "ok"}