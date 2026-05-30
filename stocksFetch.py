import yfinance as yf
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta

# 2. MongoDB Bağlantı Bilgileri
# BURAYI GÜNCELLE: <password> kısmına Atlas şifreni yaz
MONGO_URI = "mongodb+srv://yorgahuseyin_db_user:212123@cluster0.qmrzavq.mongodb.net/?appName=Cluster0"

client = MongoClient(MONGO_URI)
db = client["stock_tracking_db"]
collection = db["sp500_datas2"]

# 2. Define your 20 stocks and the timeframe
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "INTC", 
           "CSCO", "ADBE", "QCOM", "PYPL", "AMD", "TMUS", "COST", "BKNG", "AMGN", "SBUX"]

end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

def fetch_and_store():
    all_data = []
    
    print(f"--- İşlem Başladı: {len(tickers)} hisse senedi taranıyor ---")

    for ticker in tickers:
        try:
            print(f"Veri çekiliyor: {ticker}...")
            # auto_adjust=True: Temiz fiyat verisi sağlar
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                print(f"Uyarı: {ticker} için veri bulunamadı.")
                continue

            # --- KRİTİK DÜZELTME: MultiIndex Sütunları Temizleme ---
            # Eğer sütunlar ('Close', 'AAPL') gibiyse sadece 'Close' kısmını alıyoruz
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.reset_index(inplace=True)

            # Veriyi MongoDB formatına (Dictionary) çevirme
            for _, row in df.iterrows():
                # Aldığın 'Series' hatasını önlemek için her değeri güvenli bir şekilde çekiyoruz
                def get_val(col_name):
                    val = row[col_name]
                    # Eğer değer bir Series ise ilk elemanını al, değilse kendisini al
                    return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

                data_point = {
                    "ticker": ticker,
                    "date": row['Date'],
                    "open": get_val('Open'),
                    "high": get_val('High'),
                    "low": get_val('Low'),
                    "close": get_val('Close'),
                    "volume": int(row['Volume'].iloc[0]) if isinstance(row['Volume'], pd.Series) else int(row['Volume']),
                    "updated_at": datetime.utcnow()
                }
                all_data.append(data_point)

        except Exception as e:
            print(f"HATA: {ticker} işlenirken bir sorun oluştu: {e}")

    # 4. MongoDB'ye Toplu (Bulk) Yükleme
    if all_data:
        print(f"\n--- Veritabanına Yazılıyor: {len(all_data)} satır ---")
        try:
            # Önce eski verileri temizlemek istersen (opsiyonel):
            # collection.delete_many({"ticker": {"$in": tickers}})
            
            result = collection.insert_many(all_data)
            print(f"BAŞARILI: {len(result.inserted_ids)} adet döküman kaydedildi.")
        except Exception as e:
            print(f"MongoDB Yazma Hatası: {e}")
    else:
        print("Hiç veri toplanamadı.")

if __name__ == "__main__":
    fetch_and_store()