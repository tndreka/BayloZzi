# ai/reasoning.py

import os
import json
from openai import OpenAI
from datetime import datetime

# Optional: load from .env if using dotenv
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

NEWS_PATH = "data/news_sentiment.json"  # Expected format: list of dicts

def load_recent_news():
    if not os.path.exists(NEWS_PATH):
        return []

    with open(NEWS_PATH, "r") as f:
        data = json.load(f)

    # Optional: filter only last 24h
    now = datetime.utcnow()
    filtered = [
        item for item in data
        if (now - datetime.fromisoformat(item['publishedAt'].replace("Z", ""))).total_seconds() < 86400
    ]
    return filtered

def ask_ai(news_items):
    summaries = "\n".join([f"- {item['summary']} (Sentiment: {item['sentiment']})" for item in news_items])

    prompt = f"""
    You are a Forex macroeconomic analyst assistant.

    Based on the following economic summaries from the last 24h, identify key patterns and reasoning.

    Summaries:
    {summaries}

    1. What currencies are likely to be bullish or bearish based on this?
    2. Should we avoid any trades today due to uncertainty?
    3. What should a trader watch for in the upcoming hours?
    Provide reasoning and clarity in bullet points.
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def run_reasoning_engine():
    news = load_recent_news()
    if not news:
        print("No recent news found.")
        return

    print("\n🤖 Running reasoning engine...")
    thoughts = ask_ai(news)
    print("\n💡 Market Insights:\n")
    print(thoughts)

if __name__ == "__main__":
    run_reasoning_engine()


# 🔧 Requirements
# Add to requirements.txt:

# nginx
# Copy
# Edit
# openai
# python-dotenv
# And .env:

# env
# Copy
# Edit
# OPENAI_API_KEY=your_openai_key



# 🔮 Next Level Ideas
# Connect reasoning.py to modify bot behavior (e.g., block USD trades if risky).

# Use memory (ChromaDB) to store past reasoning and compare day-to-day.

# Let LLM revise trading rules live based on news.

