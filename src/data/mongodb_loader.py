import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "INTC",
    "CSCO", "ADBE", "QCOM", "PYPL", "AMD", "TMUS", "COST", "BKNG", "AMGN", "SBUX"
]

def get_database():
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI is not set in environment variables.")
    
    client = MongoClient(MONGODB_URI)
    return client["stock_tracking_db"]

def load_stock_data() -> pd.DataFrame:
    db = get_database()
    collection = db["historical_prices"]

    cursor = collection.find(
        {"ticker": {"$in": TICKERS}},
        {"_id": 0}
    )

    df = pd.DataFrame(list(cursor))

    if df.empty:
        raise ValueError("No stock data found in MongoDB.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


def load_sp500_data() -> pd.DataFrame:
    db = get_database()
    collection = db["sp500_datas"]

    cursor = collection.find(
        {},
        {"_id": 0}
    )

    df = pd.DataFrame(list(cursor))

    if df.empty:
        raise ValueError("No SP500 data found in MongoDB.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df