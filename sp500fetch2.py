import yfinance as yf
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def calculate_technical_indicators():
    # 1. Tarih Aralığı Belirleme (Bitiş tarihini yfinance dahil etsin diye 1 gün ileri alıyoruz)
    start_date = "2023-04-03"
    end_date = "2026-04-02"

    print("Yahoo Finance üzerinden S&P 500 ve VIX verileri çekiliyor...")
    
    # Hem S&P 500'ü hem de Volatilite Endeksini (VIX) aynı anda çekiyoruz
    tickers = ["^GSPC", "^VIX"]
    data = yf.download(tickers, start=start_date, end=end_date)

    # MultiIndex veri yapısını çözümleyerek ayırma
    sp500_close = data['Close']['^GSPC']
    sp500_volume = data['Volume']['^GSPC']
    vix_close = data['Close']['^VIX']

    # İşlenebilir bir Pandas DataFrame oluşturma
    df = pd.DataFrame({
        'close': sp500_close,
        'volume': sp500_volume,
        'vix': vix_close
    })

    # Tatil günleri gibi eksik verileri veri setinden temizliyoruz
    df.dropna(inplace=True)

    print("İndikatörler (RSI, Beta, Korelasyon) hesaplanıyor...")

    # --- 1. RSI (14 Günlük) Hesaplaması (Wilder's Smoothing Metodu) ---
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # --- 2. Beta Hesaplaması ---
    # S&P 500'ün kendisine karşı Beta'sı 1'dir. 
    # Not: Eğer bir hisse için yapacaksan: Hisse_Getirisi.cov(SP500_Getirisi) / SP500_Getirisi.var()
    df['beta'] = 1.0

    # --- 3. Korelasyon Hesaplaması (S&P 500 ve VIX) ---
    # Fiyatların günlük yüzdelik değişimlerini (return) buluyoruz
    sp500_return = df['close'].pct_change()
    vix_return = df['vix'].pct_change()
    
    # 30 günlük hareketli korelasyon (Rolling Correlation)
    df['correlation_sp_vix_30d'] = sp500_return.rolling(window=30).corr(vix_return)

    # RSI 14 gün, Korelasyon 30 gün geriye baktığı için ilk 30 günün verisi "NaN" (boş) çıkar.
    # Tahmin modellerini bozmaması için bu NaN değerleri siliyoruz.
    df.dropna(inplace=True)

    # 3. MongoDB'ye Yeni Koleksiyon Olarak Yazma
    print("Hesaplamalar tamamlandı, MongoDB'ye yazılıyor...")
    mongo_uri = os.environ.get("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client["stock_tracking_db"]  # Kendi veritabanı adınla değiştir
    new_collection = db["sp500_datas"] # Yeni oluşturulacak koleksiyon

    # Eğer kodu tekrar tekrar test edeceksen, eski kayıtları silip üzerine yazmasını sağlar
    new_collection.delete_many({})

    # DataFrame'i MongoDB'nin kabul edeceği Dictionary (JSON) formatına çevirme
    records = []
    for date, row in df.iterrows():
        records.append({
            "date": date,  # MongoDB'de otomatik olarak BSON ISODate formatına dönüşür
            "close": float(row['close']),
            "volume": int(row['volume']),
            "vix": float(row['vix']),
            "rsi_14": float(row['rsi_14']),
            "beta": float(row['beta']),
            "correlation_sp_vix": float(row['correlation_sp_vix_30d'])
        })

    # Veritabanına toplu (bulk) ekleme
    if records:
        new_collection.insert_many(records)
        # Hızlı sorgular için tarih alanına indeks (index) atıyoruz
        new_collection.create_index("date", unique=True)
        print(f"Başarılı! Toplam {len(records)} günlük analiz verisi 'sp500_analiz' koleksiyonuna eklendi.")
    else:
        print("Yazılacak veri bulunamadı.")

if __name__ == "__main__":
    calculate_technical_indicators()