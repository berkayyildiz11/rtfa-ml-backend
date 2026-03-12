import torch
from transformers import pipeline

class FinBERTSentiment:
    def __init__(self):
        """
        Initializes the FinBERT model
        """
        print("Loading FinBERT model into memory...")

        device=0 if torch.cuda.is_available() else -1

        self.nlp = pipeline(
            "sentiment-analysis",
            model = "ProsusAI/finbert",
            device=device
        )
        print("FinBERT successfully loaded!")

    def analyze_headline(self, text: str) -> float:
        """
        Takes a string and returns a float number for related company and news
        How would the new affect the company between -1.0 and 1.0
        """
        result = self.nlp(text)[0]
        label = result['label']
        confidence = result['score']

        if label == "positive":
            return round(confidence, 4)
        elif label == "negative":
            return round(-confidence, 4)
        else:
            return 0.0
        

if __name__ == "__main__":

    analyzer = FinBERTSentiment()

    apple_new = {'ticker': 'AAPL', 'headline': 'Apple vs Tesla in 2026: Which Stock Will Anchor Your Retirement' 
                 'and Which Will Wreck It?', 'summary': 'Apple (NASDAQ: AAPL) and Tesla (NASDAQ: TSLA) both may' 
                 'command intense customer loyalty through a shared focus on premium design and disruptive innovation,' 
                 'but they sit at opposite ends of the retirement-suitability spectrum in 2026. One has spent the past '
                 'year building durable competitive advantages and rewarding shareholders with consistency. The other is' 
                 ' executing an ... Apple vs Tesla in 2026: Which Stock Will Anchor Your Retirement and Which Will Wreck It?'}
    
    ticker = apple_new['ticker']
    headline = apple_new['headline']
    summary = apple_new['summary']

    full_text = f"{headline}. {summary}"

    score = analyzer.analyze_headline(full_text)

    if score > 0.1:
            badge = "🟢 BULLISH"
    elif score < -0.1:
        badge = "🔴 BEARISH"
    else:
        badge = "⚪️ NEUTRAL"
        
    # We still just print the headline to keep the terminal clean
    print(f"[{badge}] Score: {score: >7.4f} | {headline}")