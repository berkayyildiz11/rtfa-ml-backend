# utils/news_filters.py

COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl", "iphone", "macbook", "tim cook"],
    "MSFT": ["microsoft", "msft", "windows", "azure", "satya nadella"],
    "GOOGL": ["google", "googl", "alphabet", "sundar pichai", "youtube"],
    "AMZN": ["amazon", "amzn", "aws", "andy jassy", "jeff bezos"],
    "META": ["meta", "facebook", "zuckerberg", "instagram", "whatsapp"],
    "TSLA": ["tesla", "tsla", "elon musk", "model 3", "cybertruck"],
    "NVDA": ["nvidia", "nvda", "jensen huang", "gpu", "geforce"],
    "NFLX": ["netflix", "nflx", "ted sarandos", "streaming"],
    "ADBE": ["adobe", "adbe", "photoshop", "shantanu narayen"],
    "INTC": ["intel", "intc", "pat gelsinger", "pentium", "xeon"],
    "CSCO": ["cisco", "csco", "chuck robbins", "networking"],
    "PEP": ["pepsico", "pepsi", "pep", "ramon laguarta"],
    "AVGO": ["broadcom", "avgo", "hock tan", "vmware"],
    "TXN": ["texas instruments", "txn", "haviv ilan"],
    "QCOM": ["qualcomm", "qcom", "snapdragon", "cristiano amon"],
    "COST": ["costco", "cost", "craig jelinek", "wholesale"],
    "TMUS": ["t-mobile", "tmus", "mike sievert", "sprint"],
    "AMGN": ["amgen", "amgn", "robert bradway"],
    "SBUX": ["starbucks", "sbux", "laxman narasimhan", "coffee"],
    "ISRG": ["intuitive surgical", "isrg", "da vinci", "gary guthart"]
}

def is_article_relevant(ticker: str, headline: str, summary: str) -> bool:
    """
    Checks if given article is relevant and returns a boolean value.
    """
    
    if ticker not in COMPANY_ALIASES:
        return True
    
    full_text = f"{headline}: {summary}".lower()

    for alias in COMPANY_ALIASES[ticker]:
        if alias in full_text:
            return True
        
    return False