# ai/news_agent.py

import requests
import os
import time
import schedule
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

NEWS_ENDPOINT = "https://newsapi.org/v2/everything"
KEYWORDS = ["forex", "central bank", "interest rate", "usd", "eur", "inflation", "ecb", "federal reserve"]

def fetch_news(query="forex", hours=24):
    from_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    params = {
        'q': query,
        'from': from_time,
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': NEWS_API_KEY,
        'pageSize': 10,
    }
    res = requests.get(NEWS_ENDPOINT, params=params)
    if res.status_code != 200:
        raise Exception(f"News API Error: {res.status_code}")
    
    return res.json().get("articles", [])

def summarize_and_score(articles):
    from transformers import pipeline

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    sentiment = pipeline("sentiment-analysis")

    summaries = []
    for article in articles:
        content = article.get("description") or article.get("content") or ""
        if len(content) < 20:
            continue

        summary = summarizer(content, max_length=60, min_length=15, do_sample=False)[0]['summary_text']
        senti = sentiment(summary)[0]

        summaries.append({
            "title": article["title"],
            "summary": summary,
            "sentiment": senti["label"],
            "confidence": senti["score"],
            "url": article["url"],
            "publishedAt": article["publishedAt"]
        })

    return summaries

def run_news_agent():
    print(f"\n[{datetime.utcnow().isoformat()}] Fetching economic news...")
    all_articles = []

    for keyword in KEYWORDS:
        try:
            articles = fetch_news(query=keyword)
            all_articles.extend(articles)
        except Exception as e:
            print(f"Failed fetching for {keyword}: {e}")

    unique_articles = {a['url']: a for a in all_articles}.values()
    print(f"Fetched {len(unique_articles)} unique articles.")

    results = summarize_and_score(list(unique_articles))
    for item in results:
        print(f"\n📰 {item['title']}")
        print(f"→ Summary: {item['summary']}")
        print(f"→ Sentiment: {item['sentiment']} ({item['confidence']:.2f})")
        print(f"→ Link: {item['url']}")

if __name__ == "__main__":
    schedule.every(2).hours.do(run_news_agent)
    run_news_agent()

    while True:
        schedule.run_pending()
        time.sleep(10)



# /*
# Below is a module that fetches 24h economic headlines, summarizes them, and performs basic sentiment analysis (buy/bearish/neutral).

# We’ll use:

# newsapi.org (free-tier API for headlines)

# OpenAI GPT or HuggingFace (for summarizing and scoring sentiment)

# schedule (to run every 1 hour)

# 📄 Create an API Key
# Sign up at https://newsapi.org

# //Save your key in a .env file or config
