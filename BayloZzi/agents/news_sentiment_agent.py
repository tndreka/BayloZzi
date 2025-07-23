# agents/news_sentiment_agent.py

import asyncio
import logging
import numpy as np
import pandas as pd
import requests
import json
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import nltk
import yfinance as yf
import re
from collections import defaultdict
import schedule
import time
import asyncio
from threading import Thread
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class NewsImpact(Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    SLIGHTLY_BULLISH = "slightly_bullish"
    NEUTRAL = "neutral"
    SLIGHTLY_BEARISH = "slightly_bearish"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"

class NewsImportance(Enum):
    CRITICAL = "critical"      # Market-moving events
    HIGH = "high"             # Important economic data
    MEDIUM = "medium"         # Regular economic updates
    LOW = "low"               # Minor news

@dataclass
class NewsEvent:
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    currency_pairs: List[str]
    impact: NewsImpact
    importance: NewsImportance
    confidence: float
    sentiment_score: float
    keywords: List[str]
    market_reaction_prediction: Dict[str, float]

@dataclass
class SentimentAnalysis:
    overall_sentiment: NewsImpact
    confidence: float
    currency_bias: Dict[str, float]  # USD: 0.8 (bullish), EUR: -0.3 (bearish)
    key_events: List[NewsEvent]
    market_risk_level: str
    time_sensitivity: str  # immediate, hours, days
    conflicting_signals: bool

class NewsSentimentAgent:
    """
    Advanced News Sentiment Agent for Multi-Agent Forex Trading System.
    Provides comprehensive fundamental analysis through news sentiment.
    """
    
    def __init__(self, agent_id: str = "news_sentiment"):
        self.agent_id = agent_id
        
        # Redis connection for inter-agent communication
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # API configurations
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # News sources and endpoints
        self.news_sources = {
            'newsapi': {
                'endpoint': 'https://newsapi.org/v2/everything',
                'headers': {'X-API-Key': self.news_api_key}
            },
            'alpha_vantage': {
                'endpoint': 'https://www.alphavantage.co/query',
                'key': os.getenv("ALPHA_VANTAGE_KEY")
            }
        }
        
        # Currency-specific keywords
        self.currency_keywords = {
            'USD': ['federal reserve', 'fed', 'jerome powell', 'dollar', 'usd', 'us economy', 'treasury', 'fomc'],
            'EUR': ['ecb', 'european central bank', 'christine lagarde', 'euro', 'eur', 'eurozone', 'eu economy'],
            'GBP': ['bank of england', 'boe', 'pound', 'gbp', 'uk economy', 'brexit', 'andrew bailey'],
            'JPY': ['bank of japan', 'boj', 'yen', 'jpy', 'japan economy', 'kuroda', 'ueda'],
            'CHF': ['swiss national bank', 'snb', 'franc', 'chf', 'switzerland'],
            'CAD': ['bank of canada', 'boc', 'canadian dollar', 'cad', 'canada economy'],
            'AUD': ['reserve bank australia', 'rba', 'australian dollar', 'aud', 'australia economy'],
            'NZD': ['reserve bank new zealand', 'rbnz', 'new zealand dollar', 'nzd']
        }
        
        # Economic indicators and their impacts
        self.economic_indicators = {
            'interest_rate': ['interest rate', 'rate hike', 'rate cut', 'monetary policy'],
            'inflation': ['inflation', 'cpi', 'pce', 'price index'],
            'employment': ['employment', 'jobs', 'unemployment', 'nonfarm payrolls'],
            'gdp': ['gdp', 'economic growth', 'recession'],
            'trade': ['trade balance', 'exports', 'imports', 'trade war'],
            'manufacturing': ['manufacturing', 'industrial production', 'pmi'],
            'consumer': ['consumer confidence', 'retail sales', 'consumer spending']
        }
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
        # News cache and history
        self.news_cache = {}
        self.sentiment_history = []
        self.analysis_history = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_analyses': 0,
            'accurate_predictions': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'false_positive_rate': 0.0
        }
        
        # Real-time news monitoring
        self.monitoring_active = False
        self.last_update = datetime.now()
        
        # Currency pair impact weights
        self.currency_weights = {
            'EURUSD': {'EUR': 0.6, 'USD': 0.4},
            'GBPUSD': {'GBP': 0.6, 'USD': 0.4},
            'USDJPY': {'USD': 0.6, 'JPY': 0.4},
            'USDCHF': {'USD': 0.6, 'CHF': 0.4},
            'AUDUSD': {'AUD': 0.6, 'USD': 0.4},
            'USDCAD': {'USD': 0.6, 'CAD': 0.4},
            'NZDUSD': {'NZD': 0.6, 'USD': 0.4}
        }
    
    def _initialize_nlp_models(self):
        """Initialize NLP models for sentiment analysis."""
        try:
            # Financial sentiment model
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_pipeline = pipeline(
                "sentiment-analysis", 
                model=self.finbert_model, 
                tokenizer=self.finbert_tokenizer
            )
            
            # General sentiment pipeline
            self.general_sentiment = pipeline("sentiment-analysis", 
                                            model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            # Text summarization
            self.summarizer = pipeline("summarization", 
                                     model="facebook/bart-large-cnn",
                                     max_length=150, min_length=50)
            
            # Download required NLTK data
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
            except:
                pass
            
            logger.info("NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {str(e)}")
            # Fallback to basic models
            self.finbert_pipeline = None
            self.general_sentiment = pipeline("sentiment-analysis")
    
    async def analyze_market_sentiment(self, symbol: str, hours_back: int = 24) -> SentimentAnalysis:
        """
        Comprehensive market sentiment analysis based on recent news.
        
        Args:
            symbol: Currency pair symbol
            hours_back: Hours to look back for news
            
        Returns:
            Complete sentiment analysis
        """
        try:
            # Fetch relevant news
            news_events = await self._fetch_relevant_news(symbol, hours_back)
            
            if not news_events:
                return self._create_neutral_sentiment(symbol)
            
            # Analyze sentiment for each event
            analyzed_events = []
            for event in news_events:
                analyzed_event = await self._analyze_single_event(event, symbol)
                if analyzed_event:
                    analyzed_events.append(analyzed_event)
            
            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(analyzed_events, symbol)
            
            # Determine currency bias
            currency_bias = self._calculate_currency_bias(analyzed_events, symbol)
            
            # Assess market risk level
            risk_level = self._assess_market_risk(analyzed_events)
            
            # Determine time sensitivity
            time_sensitivity = self._determine_time_sensitivity(analyzed_events)
            
            # Check for conflicting signals
            conflicting_signals = self._detect_conflicting_signals(analyzed_events)
            
            # Calculate confidence
            confidence = self._calculate_sentiment_confidence(analyzed_events, conflicting_signals)
            
            sentiment_analysis = SentimentAnalysis(
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                currency_bias=currency_bias,
                key_events=analyzed_events[:5],  # Top 5 most important
                market_risk_level=risk_level,
                time_sensitivity=time_sensitivity,
                conflicting_signals=conflicting_signals
            )
            
            # Store analysis
            self.analysis_history.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'analysis': sentiment_analysis
            })
            
            # Send to other agents if significant
            if confidence > 0.7 or any(event.importance in [NewsImportance.CRITICAL, NewsImportance.HIGH] 
                                     for event in analyzed_events):
                await self._send_sentiment_signal(symbol, sentiment_analysis)
            
            logger.info(f"Sentiment analysis completed for {symbol}: {overall_sentiment.value} (confidence: {confidence:.2f})")
            
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return self._create_neutral_sentiment(symbol)
    
    async def _fetch_relevant_news(self, symbol: str, hours_back: int) -> List[Dict[str, Any]]:
        """Fetch news relevant to the currency pair."""
        try:
            # Get currencies from symbol
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
            
            all_news = []
            
            # Fetch from NewsAPI
            news_articles = await self._fetch_from_newsapi(base_currency, quote_currency, hours_back)
            all_news.extend(news_articles)
            
            # Fetch from Alpha Vantage (if available)
            if self.news_sources['alpha_vantage']['key']:
                av_news = await self._fetch_from_alpha_vantage(base_currency, quote_currency)
                all_news.extend(av_news)
            
            # Remove duplicates and filter relevant
            unique_news = self._remove_duplicates(all_news)
            relevant_news = self._filter_relevant_news(unique_news, base_currency, quote_currency)
            
            return relevant_news
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
    
    async def _fetch_from_newsapi(self, base_currency: str, quote_currency: str, hours_back: int) -> List[Dict]:
        """Fetch news from NewsAPI."""
        try:
            if not self.news_api_key:
                return []
            
            # Build search query
            keywords = []
            keywords.extend(self.currency_keywords.get(base_currency, []))
            keywords.extend(self.currency_keywords.get(quote_currency, []))
            
            from_time = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            
            params = {
                'q': ' OR '.join(keywords[:10]),  # Limit to avoid API issues
                'from': from_time,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 50
            }
            
            headers = {'X-API-Key': self.news_api_key}
            
            response = requests.get(self.news_sources['newsapi']['endpoint'], 
                                  params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
            else:
                logger.warning(f"NewsAPI error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {str(e)}")
            return []
    
    async def _fetch_from_alpha_vantage(self, base_currency: str, quote_currency: str) -> List[Dict]:
        """Fetch news from Alpha Vantage."""
        try:
            if not self.news_sources['alpha_vantage']['key']:
                return []
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': f'{base_currency},{quote_currency}',
                'apikey': self.news_sources['alpha_vantage']['key']
            }
            
            response = requests.get(self.news_sources['alpha_vantage']['endpoint'], 
                                  params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('feed', [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error fetching from Alpha Vantage: {str(e)}")
            return []
    
    def _remove_duplicates(self, news_list: List[Dict]) -> List[Dict]:
        """Remove duplicate news articles."""
        seen_titles = set()
        unique_news = []
        
        for article in news_list:
            title = article.get('title', '').lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(article)
        
        return unique_news
    
    def _filter_relevant_news(self, news_list: List[Dict], base_currency: str, quote_currency: str) -> List[Dict]:
        """Filter news for relevance to currency pair."""
        relevant_news = []
        
        for article in news_list:
            title = article.get('title', '').lower()
            content = article.get('description', '').lower() + ' ' + article.get('content', '').lower()
            
            # Check for currency-specific keywords
            base_keywords = self.currency_keywords.get(base_currency, [])
            quote_keywords = self.currency_keywords.get(quote_currency, [])
            
            relevance_score = 0
            
            # Check for direct currency mentions
            for keyword in base_keywords + quote_keywords:
                if keyword.lower() in title:
                    relevance_score += 3
                elif keyword.lower() in content:
                    relevance_score += 1
            
            # Check for economic indicator keywords
            for indicator_keywords in self.economic_indicators.values():
                for keyword in indicator_keywords:
                    if keyword.lower() in title:
                        relevance_score += 2
                    elif keyword.lower() in content:
                        relevance_score += 0.5
            
            # Include if relevance score is high enough
            if relevance_score >= 2:
                article['relevance_score'] = relevance_score
                relevant_news.append(article)
        
        # Sort by relevance and recency
        relevant_news.sort(key=lambda x: (x.get('relevance_score', 0), 
                                        x.get('publishedAt', '')), reverse=True)
        
        return relevant_news[:20]  # Top 20 most relevant
    
    async def _analyze_single_event(self, article: Dict, symbol: str) -> Optional[NewsEvent]:
        """Analyze sentiment and impact of a single news event."""
        try:
            title = article.get('title', '')
            content = article.get('description', '') or article.get('content', '')
            
            if not title or not content:
                return None
            
            # Combine title and content for analysis
            text_to_analyze = f"{title}. {content}"
            
            # Financial sentiment analysis
            fin_sentiment = self._analyze_financial_sentiment(text_to_analyze)
            
            # General sentiment analysis
            gen_sentiment = self._analyze_general_sentiment(text_to_analyze)
            
            # Combine sentiments
            combined_sentiment = self._combine_sentiments(fin_sentiment, gen_sentiment)
            
            # Extract keywords
            keywords = self._extract_keywords(text_to_analyze)
            
            # Determine importance
            importance = self._determine_importance(title, content, keywords)
            
            # Predict market impact
            market_impact = self._predict_market_impact(text_to_analyze, symbol, importance)
            
            # Determine affected currency pairs
            currency_pairs = self._determine_affected_pairs(keywords)
            
            # Calculate confidence
            confidence = self._calculate_event_confidence(fin_sentiment, gen_sentiment, importance)
            
            return NewsEvent(
                title=title,
                content=content[:500],  # Truncate for storage
                source=article.get('source', {}).get('name', 'Unknown'),
                published_at=self._parse_date(article.get('publishedAt', '')),
                url=article.get('url', ''),
                currency_pairs=currency_pairs,
                impact=combined_sentiment,
                importance=importance,
                confidence=confidence,
                sentiment_score=self._sentiment_to_score(combined_sentiment),
                keywords=keywords,
                market_reaction_prediction=market_impact
            )
            
        except Exception as e:
            logger.error(f"Error analyzing single event: {str(e)}")
            return None
    
    def _analyze_financial_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using financial-specific model."""
        try:
            if self.finbert_pipeline and len(text.strip()) > 10:
                # Truncate text for model limits
                text = text[:512]
                result = self.finbert_pipeline(text)[0]
                return {
                    'label': result['label'].lower(),
                    'score': result['score'],
                    'model': 'finbert'
                }
            else:
                return {'label': 'neutral', 'score': 0.5, 'model': 'fallback'}
                
        except Exception as e:
            logger.error(f"Error in financial sentiment analysis: {str(e)}")
            return {'label': 'neutral', 'score': 0.5, 'model': 'error'}
    
    def _analyze_general_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using general model."""
        try:
            if len(text.strip()) > 10:
                text = text[:512]
                result = self.general_sentiment(text)[0]
                return {
                    'label': result['label'].lower(),
                    'score': result['score'],
                    'model': 'general'
                }
            else:
                return {'label': 'neutral', 'score': 0.5, 'model': 'fallback'}
                
        except Exception as e:
            logger.error(f"Error in general sentiment analysis: {str(e)}")
            # Fallback to TextBlob
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    return {'label': 'positive', 'score': min(abs(polarity) + 0.5, 1.0), 'model': 'textblob'}
                elif polarity < -0.1:
                    return {'label': 'negative', 'score': min(abs(polarity) + 0.5, 1.0), 'model': 'textblob'}
                else:
                    return {'label': 'neutral', 'score': 0.5, 'model': 'textblob'}
            except:
                return {'label': 'neutral', 'score': 0.5, 'model': 'error'}
    
    def _combine_sentiments(self, fin_sentiment: Dict, gen_sentiment: Dict) -> NewsImpact:
        """Combine financial and general sentiment scores."""
        try:
            # Weight financial sentiment more heavily
            fin_weight = 0.7
            gen_weight = 0.3
            
            # Convert labels to scores
            fin_score = self._label_to_score(fin_sentiment['label']) * fin_sentiment['score']
            gen_score = self._label_to_score(gen_sentiment['label']) * gen_sentiment['score']
            
            combined_score = (fin_score * fin_weight) + (gen_score * gen_weight)
            
            # Convert back to impact level
            if combined_score >= 0.8:
                return NewsImpact.VERY_BULLISH
            elif combined_score >= 0.6:
                return NewsImpact.BULLISH
            elif combined_score >= 0.3:
                return NewsImpact.SLIGHTLY_BULLISH
            elif combined_score >= -0.3:
                return NewsImpact.NEUTRAL
            elif combined_score >= -0.6:
                return NewsImpact.SLIGHTLY_BEARISH
            elif combined_score >= -0.8:
                return NewsImpact.BEARISH
            else:
                return NewsImpact.VERY_BEARISH
                
        except Exception as e:
            logger.error(f"Error combining sentiments: {str(e)}")
            return NewsImpact.NEUTRAL
    
    def _label_to_score(self, label: str) -> float:
        """Convert sentiment label to numerical score."""
        label = label.lower()
        if label in ['positive', 'bullish']:
            return 1.0
        elif label in ['negative', 'bearish']:
            return -1.0
        else:
            return 0.0
    
    def _sentiment_to_score(self, sentiment: NewsImpact) -> float:
        """Convert NewsImpact to numerical score."""
        impact_scores = {
            NewsImpact.VERY_BULLISH: 1.0,
            NewsImpact.BULLISH: 0.6,
            NewsImpact.SLIGHTLY_BULLISH: 0.3,
            NewsImpact.NEUTRAL: 0.0,
            NewsImpact.SLIGHTLY_BEARISH: -0.3,
            NewsImpact.BEARISH: -0.6,
            NewsImpact.VERY_BEARISH: -1.0
        }
        return impact_scores.get(sentiment, 0.0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        try:
            text = text.lower()
            keywords = []
            
            # Check for currency keywords
            for currency, currency_keywords in self.currency_keywords.items():
                for keyword in currency_keywords:
                    if keyword in text:
                        keywords.append(keyword)
            
            # Check for economic indicator keywords
            for indicator, indicator_keywords in self.economic_indicators.items():
                for keyword in indicator_keywords:
                    if keyword in text:
                        keywords.append(keyword)
            
            # Add specific financial terms
            financial_terms = ['rate hike', 'rate cut', 'hawkish', 'dovish', 'qe', 'quantitative easing',
                             'tapering', 'stimulus', 'recession', 'recovery', 'volatility']
            
            for term in financial_terms:
                if term in text:
                    keywords.append(term)
            
            return list(set(keywords))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def _determine_importance(self, title: str, content: str, keywords: List[str]) -> NewsImportance:
        """Determine the importance level of the news."""
        try:
            text = (title + ' ' + content).lower()
            
            # Critical indicators
            critical_terms = ['federal reserve meeting', 'ecb meeting', 'rate decision', 'employment report',
                            'gdp release', 'inflation data', 'central bank', 'emergency meeting']
            
            for term in critical_terms:
                if term in text:
                    return NewsImportance.CRITICAL
            
            # High importance indicators
            high_terms = ['interest rate', 'monetary policy', 'jobs report', 'trade war', 'brexit',
                        'economic outlook', 'financial crisis']
            
            for term in high_terms:
                if term in text:
                    return NewsImportance.HIGH
            
            # Medium importance
            medium_terms = ['economic data', 'manufacturing', 'retail sales', 'consumer confidence']
            
            for term in medium_terms:
                if term in text:
                    return NewsImportance.MEDIUM
            
            # Default to low if no specific indicators
            return NewsImportance.LOW
            
        except Exception as e:
            logger.error(f"Error determining importance: {str(e)}")
            return NewsImportance.LOW
    
    def _predict_market_impact(self, text: str, symbol: str, importance: NewsImportance) -> Dict[str, float]:
        """Predict potential market impact."""
        try:
            text = text.lower()
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
            
            impact_prediction = {
                'immediate_impact': 0.0,
                'short_term_impact': 0.0,
                'volatility_increase': 0.0
            }
            
            # Base impact based on importance
            importance_multipliers = {
                NewsImportance.CRITICAL: 1.0,
                NewsImportance.HIGH: 0.7,
                NewsImportance.MEDIUM: 0.4,
                NewsImportance.LOW: 0.2
            }
            
            base_impact = importance_multipliers.get(importance, 0.2)
            
            # Adjust based on content
            if any(term in text for term in ['rate hike', 'hawkish', 'tightening']):
                impact_prediction['immediate_impact'] = base_impact * 0.8
                impact_prediction['volatility_increase'] = base_impact * 0.6
            elif any(term in text for term in ['rate cut', 'dovish', 'easing']):
                impact_prediction['immediate_impact'] = base_impact * 0.8
                impact_prediction['volatility_increase'] = base_impact * 0.6
            
            # Currency-specific adjustments
            base_keywords = self.currency_keywords.get(base_currency, [])
            quote_keywords = self.currency_keywords.get(quote_currency, [])
            
            base_mentions = sum(1 for keyword in base_keywords if keyword in text)
            quote_mentions = sum(1 for keyword in quote_keywords if keyword in text)
            
            if base_mentions > quote_mentions:
                impact_prediction['immediate_impact'] *= 1.2
            elif quote_mentions > base_mentions:
                impact_prediction['immediate_impact'] *= 1.1
            
            impact_prediction['short_term_impact'] = impact_prediction['immediate_impact'] * 0.7
            
            return impact_prediction
            
        except Exception as e:
            logger.error(f"Error predicting market impact: {str(e)}")
            return {'immediate_impact': 0.0, 'short_term_impact': 0.0, 'volatility_increase': 0.0}
    
    def _determine_affected_pairs(self, keywords: List[str]) -> List[str]:
        """Determine which currency pairs might be affected."""
        affected_currencies = set()
        
        for keyword in keywords:
            for currency, currency_keywords in self.currency_keywords.items():
                if keyword in currency_keywords:
                    affected_currencies.add(currency)
        
        # Generate pairs with USD as the most liquid
        pairs = []
        for currency in affected_currencies:
            if currency != 'USD':
                pairs.append(f"{currency}USD")
                pairs.append(f"USD{currency}")
        
        return pairs[:5]  # Limit to top 5
    
    def _calculate_event_confidence(self, fin_sentiment: Dict, gen_sentiment: Dict, 
                                  importance: NewsImportance) -> float:
        """Calculate confidence score for the event analysis."""
        try:
            # Base confidence from sentiment scores
            fin_confidence = fin_sentiment.get('score', 0.5)
            gen_confidence = gen_sentiment.get('score', 0.5)
            
            # Weighted average
            sentiment_confidence = (fin_confidence * 0.7) + (gen_confidence * 0.3)
            
            # Importance boost
            importance_boost = {
                NewsImportance.CRITICAL: 0.2,
                NewsImportance.HIGH: 0.1,
                NewsImportance.MEDIUM: 0.05,
                NewsImportance.LOW: 0.0
            }
            
            total_confidence = min(sentiment_confidence + importance_boost.get(importance, 0.0), 1.0)
            
            return max(total_confidence, 0.1)  # Minimum confidence
            
        except Exception as e:
            logger.error(f"Error calculating event confidence: {str(e)}")
            return 0.5
    
    def _parse_date(self, date_string: str) -> datetime:
        """Parse various date formats."""
        try:
            # Try ISO format first
            return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        except:
            try:
                # Try without timezone
                return datetime.fromisoformat(date_string.replace('Z', ''))
            except:
                # Default to current time
                return datetime.now()
    
    def _calculate_overall_sentiment(self, events: List[NewsEvent], symbol: str) -> NewsImpact:
        """Calculate overall sentiment from multiple events."""
        if not events:
            return NewsImpact.NEUTRAL
        
        # Weight events by importance and recency
        total_score = 0
        total_weight = 0
        
        now = datetime.now()
        
        for event in events:
            # Importance weight
            importance_weights = {
                NewsImportance.CRITICAL: 1.0,
                NewsImportance.HIGH: 0.7,
                NewsImportance.MEDIUM: 0.4,
                NewsImportance.LOW: 0.2
            }
            
            importance_weight = importance_weights.get(event.importance, 0.2)
            
            # Recency weight (events in last 6 hours get full weight)
            hours_ago = (now - event.published_at).total_seconds() / 3600
            recency_weight = max(0.1, 1.0 - (hours_ago / 24))  # Decay over 24 hours
            
            # Confidence weight
            confidence_weight = event.confidence
            
            total_event_weight = importance_weight * recency_weight * confidence_weight
            total_score += event.sentiment_score * total_event_weight
            total_weight += total_event_weight
        
        if total_weight == 0:
            return NewsImpact.NEUTRAL
        
        average_score = total_score / total_weight
        
        # Convert to NewsImpact
        if average_score >= 0.6:
            return NewsImpact.VERY_BULLISH
        elif average_score >= 0.3:
            return NewsImpact.BULLISH
        elif average_score >= 0.1:
            return NewsImpact.SLIGHTLY_BULLISH
        elif average_score >= -0.1:
            return NewsImpact.NEUTRAL
        elif average_score >= -0.3:
            return NewsImpact.SLIGHTLY_BEARISH
        elif average_score >= -0.6:
            return NewsImpact.BEARISH
        else:
            return NewsImpact.VERY_BEARISH
    
    def _calculate_currency_bias(self, events: List[NewsEvent], symbol: str) -> Dict[str, float]:
        """Calculate bias for individual currencies."""
        base_currency = symbol[:3]
        quote_currency = symbol[3:6]
        
        currency_scores = defaultdict(list)
        
        for event in events:
            for keyword in event.keywords:
                for currency, currency_keywords in self.currency_keywords.items():
                    if keyword in currency_keywords:
                        # Weight by importance and confidence
                        weight = event.confidence
                        if event.importance == NewsImportance.CRITICAL:
                            weight *= 1.5
                        elif event.importance == NewsImportance.HIGH:
                            weight *= 1.2
                        
                        currency_scores[currency].append(event.sentiment_score * weight)
        
        # Calculate average scores
        currency_bias = {}
        for currency in [base_currency, quote_currency]:
            scores = currency_scores.get(currency, [0])
            currency_bias[currency] = np.mean(scores)
        
        return currency_bias
    
    def _assess_market_risk(self, events: List[NewsEvent]) -> str:
        """Assess overall market risk level."""
        if not events:
            return "low"
        
        critical_count = sum(1 for event in events if event.importance == NewsImportance.CRITICAL)
        high_count = sum(1 for event in events if event.importance == NewsImportance.HIGH)
        
        very_negative_count = sum(1 for event in events 
                                if event.impact in [NewsImpact.VERY_BEARISH, NewsImpact.BEARISH])
        very_positive_count = sum(1 for event in events 
                                if event.impact in [NewsImpact.VERY_BULLISH, NewsImpact.BULLISH])
        
        if critical_count >= 2 or (critical_count >= 1 and high_count >= 3):
            return "critical"
        elif critical_count >= 1 or high_count >= 2:
            return "high"
        elif very_negative_count >= 3 or very_positive_count >= 3:
            return "medium"
        else:
            return "low"
    
    def _determine_time_sensitivity(self, events: List[NewsEvent]) -> str:
        """Determine how quickly the market might react."""
        if not events:
            return "days"
        
        immediate_terms = ['breaking', 'emergency', 'urgent', 'just in']
        hours_terms = ['meeting', 'decision', 'announcement']
        
        for event in events:
            title_content = (event.title + ' ' + event.content).lower()
            
            if any(term in title_content for term in immediate_terms):
                return "immediate"
            elif event.importance == NewsImportance.CRITICAL:
                return "hours"
            elif any(term in title_content for term in hours_terms):
                return "hours"
        
        return "days"
    
    def _detect_conflicting_signals(self, events: List[NewsEvent]) -> bool:
        """Detect if there are conflicting sentiment signals."""
        if len(events) < 2:
            return False
        
        bullish_count = sum(1 for event in events 
                          if event.impact in [NewsImpact.BULLISH, NewsImpact.VERY_BULLISH])
        bearish_count = sum(1 for event in events 
                          if event.impact in [NewsImpact.BEARISH, NewsImpact.VERY_BEARISH])
        
        # Conflicting if both bullish and bearish signals are significant
        return bullish_count >= 1 and bearish_count >= 1 and len(events) >= 3
    
    def _calculate_sentiment_confidence(self, events: List[NewsEvent], conflicting: bool) -> float:
        """Calculate overall confidence in sentiment analysis."""
        if not events:
            return 0.0
        
        # Average event confidence
        avg_confidence = np.mean([event.confidence for event in events])
        
        # Reduce confidence if conflicting signals
        if conflicting:
            avg_confidence *= 0.7
        
        # Boost confidence with more high-quality events
        quality_events = sum(1 for event in events 
                           if event.importance in [NewsImportance.CRITICAL, NewsImportance.HIGH])
        
        if quality_events >= 3:
            avg_confidence *= 1.1
        elif quality_events >= 2:
            avg_confidence *= 1.05
        
        return min(avg_confidence, 1.0)
    
    def _create_neutral_sentiment(self, symbol: str) -> SentimentAnalysis:
        """Create neutral sentiment analysis when no news is available."""
        return SentimentAnalysis(
            overall_sentiment=NewsImpact.NEUTRAL,
            confidence=0.5,
            currency_bias={symbol[:3]: 0.0, symbol[3:6]: 0.0},
            key_events=[],
            market_risk_level="low",
            time_sensitivity="days",
            conflicting_signals=False
        )
    
    async def _send_sentiment_signal(self, symbol: str, sentiment: SentimentAnalysis):
        """Send sentiment signal to other agents."""
        try:
            signal_data = {
                'symbol': symbol,
                'sentiment': sentiment.overall_sentiment.value,
                'confidence': sentiment.confidence,
                'currency_bias': sentiment.currency_bias,
                'market_risk_level': sentiment.market_risk_level,
                'time_sensitivity': sentiment.time_sensitivity,
                'conflicting_signals': sentiment.conflicting_signals,
                'key_events_count': len(sentiment.key_events),
                'source': 'news_sentiment'
            }
            
            message = {
                'sender': self.agent_id,
                'receiver': 'risk_manager',
                'message_type': 'sentiment_analysis',
                'data': signal_data,
                'confidence': sentiment.confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to risk manager
            channel = "agent_messages_risk_manager"
            self.redis_client.publish(channel, json.dumps(message, default=str))
            
            # Send to chart analyzer for confluence
            message['receiver'] = 'chart_analyzer'
            channel = "agent_messages_chart_analyzer"
            self.redis_client.publish(channel, json.dumps(message, default=str))
            
            logger.info(f"Sentiment signal sent: {sentiment.overall_sentiment.value} for {symbol}")
            
        except Exception as e:
            logger.error(f"Error sending sentiment signal: {str(e)}")
    
    def start_monitoring(self):
        """Start continuous news monitoring."""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Monitor major pairs
                    major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
                    
                    for pair in major_pairs:
                        asyncio.run(self.analyze_market_sentiment(pair, hours_back=1))
                        time.sleep(2)  # Rate limiting
                    
                    time.sleep(300)  # 5-minute intervals
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    time.sleep(60)
        
        self.monitoring_active = True
        monitor_thread = Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        logger.info("News monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous news monitoring."""
        self.monitoring_active = False
        logger.info("News monitoring stopped")
    
    def update_performance(self, prediction_result: Dict):
        """Update performance metrics based on actual results."""
        try:
            self.performance_metrics['total_analyses'] += 1
            
            if prediction_result.get('accurate', False):
                self.performance_metrics['accurate_predictions'] += 1
            
            self.performance_metrics['accuracy'] = (
                self.performance_metrics['accurate_predictions'] / 
                self.performance_metrics['total_analyses']
            )
            
            logger.info(f"Sentiment performance updated: {self.performance_metrics['accuracy']:.1%} accuracy")
            
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis summary."""
        return {
            'agent_id': self.agent_id,
            'performance': self.performance_metrics,
            'total_analyses': len(self.analysis_history),
            'recent_analyses': [
                {
                    'symbol': analysis['symbol'],
                    'sentiment': analysis['analysis'].overall_sentiment.value,
                    'confidence': analysis['analysis'].confidence,
                    'timestamp': analysis['timestamp']
                }
                for analysis in self.analysis_history[-5:]
            ],
            'monitoring_active': self.monitoring_active,
            'last_update': self.last_update
        }