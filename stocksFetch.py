import yfinance as yf
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta

# 1. Setup Database Connection
# Replace with your actual Atlas URI (use .env in production!)
MONGO_URI = "mongodb+srv://yorgahuseyin_db_user:212123@cluster0.qmrzavq.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["stock_database"]
collection = db["historical_data"]

# 2. Define your 20 stocks and the timeframe
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "TSM", "V", 
           "JPM", "WMT", "MA", "PG", "UNH", "HD", "DIS", "PYPL", "BAC", "ADBE"]

end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

def fetch_and_store_stocks():
    all_data = []

    print(f"Fetching data for {len(tickers)} stocks...")

    for ticker in tickers:
        # Download data from Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Reset index to turn 'Date' into a column
        df.reset_index(inplace=True)
        
        # Convert DataFrame to a list of dictionaries
        for _, row in df.iterrows():
            data_point = {
                "ticker": ticker,
                "date": row['Date'],
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "adj_close": float(row['Adj Close']),
                "volume": int(row['Volume'])
            }
            all_data.append(data_point)
            
    # 3. Bulk insert into MongoDB
    if all_data:
        # Delete old data first if you want a fresh start
        # collection.delete_many({"ticker": {"$in": tickers}}) 
        
        result = collection.insert_many(all_data)
        print(f"Successfully inserted {len(result.inserted_ids)} records.")

if __name__ == "__main__":
    fetch_and_store_stocks()