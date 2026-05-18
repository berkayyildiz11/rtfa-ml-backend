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

    apple_new = {'ticker': 'AAPL', 'headline': 'Atlassian job cuts raise the question: '
                 'Is AI driving layoffs?', 'summary': 'Atlassian (TEAM) plans to cut about 10%' 
                 'of its workforce as it shifts more resources toward artificial intelligence ('
                 'AI). Investopedia editor in chief Caleb Silver joins Morning Brief host Julie '
                 'Hyman to discuss whether companies truly are cutting jobs because of AI, '
                 'highlighting how investors are increasingly focused on productivity per '
                 'employee, with companies like Nvidia (NVDA) and Apple (AAPL) leading the '
                 'pack. To watch more expert insights and analysis on the latest market '
                 'action, check out more Morning Brief.'}
    
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