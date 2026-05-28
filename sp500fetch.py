import yfinance as yf
from pymongo import MongoClient, UpdateOne
import pandas as pd
from datetime import datetime

def update_mongodb_with_sp500():
    # 1. MongoDB Bağlantısı
    client = MongoClient("mongodb+srv://yorgahuseyin_db_user:212123@cluster0.qmrzavq.mongodb.net/?appName=Cluster0")
    db = client["stock_tracking_db"]
    collection = db["historical_prices"]

    min_doc = collection.find_one(sort=[("date", 1)])
    max_doc = collection.find_one(sort=[("date", -1)])

    if not min_doc or not max_doc:
        print("Koleksiyonda veri bulunamadı.")
        return

    # Tarih string olarak ('2023-04-06T00:00:00.000+00:00') gelirse datetime objesine çevirir,
    # MongoDB'den zaten datetime olarak gelirse olduğu gibi bırakır.
    def parse_date(date_val):
        if isinstance(date_val, str):
            return datetime.fromisoformat(date_val)
        return date_val

    # 2. Veritabanındaki minimum ve maksimum tarihleri bulma
    min_date = parse_date(min_doc["date"])
    max_date = parse_date(max_doc["date"])

    start_date = min_date.strftime('%Y-%m-%d')
    end_date = (max_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"{start_date} ile {end_date} aralığı için S&P 500 verisi çekiliyor...")

    # 3. S&P 500 Verisini Çekme (^GSPC ticker'ı ile)
    sp500_data = yf.download("^GSPC", start=start_date, end=end_date)
    
    if isinstance(sp500_data.columns, pd.MultiIndex):
        close_prices = sp500_data['Close']['^GSPC']
    else:
        close_prices = sp500_data['Close']

    sp500_dict = close_prices.to_dict()

    # 4. Veritabanını Güncelleme (Bulk Write İşlemi)
    operations = []
    batch_size = 1000  
    updated_count = 0

    cursor = collection.find({"sp500_value": {"$exists": False}})

    for doc in cursor:
        # Dokümandaki tarihi parse edip saat/timezone kısmını atarak eşleştirme yapıyoruz
        doc_date_obj = parse_date(doc["date"])
        date_key = pd.Timestamp(doc_date_obj.date())

        if date_key in sp500_dict and not pd.isna(sp500_dict[date_key]):
            sp_value = float(sp500_dict[date_key])
            
            operations.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": {"sp500_value": sp_value}}
                )
            )

        if len(operations) >= batch_size:
            collection.bulk_write(operations)
            updated_count += len(operations)
            print(f"{updated_count} kayıt güncellendi...")
            operations = []

    if operations:
        collection.bulk_write(operations)
        updated_count += len(operations)

    print(f"İşlem tamamlandı! Toplam {updated_count} kayda S&P 500 değeri eklendi.")

if __name__ == "__main__":
    update_mongodb_with_sp500()