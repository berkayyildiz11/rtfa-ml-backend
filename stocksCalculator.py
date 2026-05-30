import pymongo
import pandas as pd
import numpy as np
from pymongo import UpdateOne
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Database Connection Setup
# Replace with your actual MongoDB connection string and database/collection names
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "stock_tracking_db"
COLLECTION_NAME = "historical_prices"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def calculate_indicators(df):
    """Calculates Log Returns, MACD, and 14-day RSI for a single stock's dataframe."""
    # Ensure data is sorted by date from oldest to newest
    df = df.sort_values('date')
    
    # Calculate Log Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate MACD (12-day EMA - 26-day EMA)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    
    # Calculate 14-day RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    return df

def main():
    print("Fetching data from MongoDB...")
    # Fetch all records. Assuming your documents have 'ticker', 'date', and 'close' fields.
    cursor = collection.find({})
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        print("No data found in the collection.")
        return

    print("Calculating indicators...")
    # Group by ticker so indicators are calculated per stock, not across the whole dataset
    enriched_dfs = []
    for ticker, group in df.groupby('ticker'):
        processed_group = calculate_indicators(group.copy())
        enriched_dfs.append(processed_group)
        
    final_df = pd.concat(enriched_dfs)
    
    # Replace NaN values with None (MongoDB cannot store standard IEEE NaNs properly)
    final_df = final_df.replace({np.nan: None})
    
    print("Preparing bulk update for MongoDB...")
    bulk_operations = []
    
    for _, row in final_df.iterrows():
        # Using the unique MongoDB _id to update the exact document
        doc_id = row['_id']
        
        update_fields = {
            "log_return": row['log_return'],
            "macd": row['macd'],
            "rsi_14": row['rsi_14']
        }
        
        # Create an update operation for the bulk write
        operation = UpdateOne(
            {"_id": doc_id},
            {"$set": update_fields}
        )
        bulk_operations.append(operation)
        
    if bulk_operations:
        print(f"Executing {len(bulk_operations)} updates...")
        result = collection.bulk_write(bulk_operations)
        print(f"Successfully updated {result.modified_count} documents.")
    else:
        print("No operations to execute.")

if __name__ == "__main__":
    main()