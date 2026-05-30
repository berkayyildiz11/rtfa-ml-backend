import os
import asyncio
from datetime import datetime, timezone
import httpx
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

# --- AYARLAR ---
# Bu değerleri daha sonra Railway'de Environment Variables (Ortam Değişkenleri) kısmına ekleyeceğiz.
MONGO_URI = os.environ.get("MONGO_URI")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# Takip edilecek 20 hisse senedi (Örnek listeyi kendi hisselerinle değiştir)
STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "INTC", 
           "CSCO", "ADBE", "QCOM", "PYPL", "AMD", "TMUS", "COST", "BKNG", "AMGN", "SBUX"
]

# --- VERİTABANI BAĞLANTISI ---
client = AsyncIOMotorClient(MONGO_URI) #commit için ekledim
db = client.stock_tracking_db
trades_col = db.sp500deneme  # Time Series Collection olarak ayarlanmış koleksiyon

async def fetch_single_stock(http_client, symbol):
    """Tek bir hissenin verisini Finnhub'dan çeker."""
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
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

async def poll_stocks_every_25_seconds():
    """Her 25 saniyede bir tüm hisseleri çeker ve MongoDB'ye kaydeder."""
    print("Veri çekme motoru başlatıldı...")
    
    # HTTP istekleri için paylaşılan ve asenkron bir oturum açıyoruz
    async with httpx.AsyncClient() as http_client:
        while True:
            start_time = asyncio.get_event_loop().time()
            
            # 1. Aşama: Tüm hisseler için eşzamanlı istek (Concurrent fetching) oluştur
            tasks = [fetch_single_stock(http_client, symbol) for symbol in STOCKS]
            results = await asyncio.gather(*tasks)
            
            # 2. Aşama: Başarısız olan istekleri (None dönenleri) filtrele
            valid_documents = [doc for doc in results if doc is not None]
            
            # 3. Aşama: Veritabanına tek seferde (Batch Insert) kaydet
            if valid_documents:
                try:
                    await trades_col.insert_many(valid_documents)
                    print(f"{datetime.now().strftime('%H:%M:%S')} - {len(valid_documents)} hisse başarıyla kaydedildi.")
                except Exception as e:
                    print(f"Veritabanına yazma hatası: {e}")
            
            # 4. Aşama: 25 saniyelik döngüyü ayarla
            # İşlem süresini hesaplayıp, tam 25 saniyede bir çalışmasını garantiliyoruz
            elapsed_time = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, 25 - elapsed_time)
            await asyncio.sleep(sleep_time)

if __name__ == "__main__":
    # Kodu doğrudan çalıştırdığında döngüyü başlatır
    asyncio.run(poll_stocks_every_25_seconds())