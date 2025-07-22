# data/historical_data_manager.py

import asyncio
import logging
import numpy as np
import pandas as pd
import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import pickle
import redis
import requests
import yfinance as yf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    name: str
    type: str  # 'api', 'file', 'database'
    endpoint: str
    api_key: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    cost_per_request: float = 0.0
    reliability: float = 0.95

@dataclass
class MarketDataPoint:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str  # '1m', '5m', '1h', '1d', etc.
    source: str

@dataclass
class AnalysisRecord:
    id: str
    agent_id: str
    symbol: str
    timestamp: datetime
    analysis_type: str
    result: Dict[str, Any]
    confidence: float
    accuracy_score: Optional[float] = None  # Set after validation

class HistoricalDataManager:
    """
    Comprehensive Historical Data Management System for Multi-Agent Forex Trading.
    Handles data storage, retrieval, validation, and performance tracking.
    """
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Database setup
        self.db_path = self.data_dir / "forex_data.db"
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Data sources configuration
        self.data_sources = {
            'alpha_vantage': DataSource(
                name='Alpha Vantage',
                type='api',
                endpoint='https://www.alphavantage.co/query',
                api_key=os.getenv('ALPHA_VANTAGE_KEY'),
                rate_limit=5,  # 5 requests per minute for free tier
                reliability=0.95
            ),
            'yahoo_finance': DataSource(
                name='Yahoo Finance',
                type='api',
                endpoint='https://query1.finance.yahoo.com/v8/finance/chart/',
                rate_limit=2000,  # Very high limit
                reliability=0.90
            ),
            'fred': DataSource(
                name='FRED Economic Data',
                type='api',
                endpoint='https://api.stlouisfed.org/fred/series/observations',
                api_key=os.getenv('FRED_API_KEY'),
                rate_limit=120,
                reliability=0.98
            )
        }
        
        # Cache settings
        self.cache_settings = {
            'market_data_ttl': 300,    # 5 minutes for real-time data
            'daily_data_ttl': 3600,   # 1 hour for daily data
            'analysis_ttl': 1800,     # 30 minutes for analysis results
            'max_cache_size': 1000    # Maximum cached items
        }
        
        # Initialize database
        self._initialize_database()
        
        # Data validation rules
        self.validation_rules = {
            'max_price_change': 0.10,  # 10% max change between consecutive data points
            'min_volume': 0,           # Minimum volume (0 allows missing volume)
            'max_gap_hours': 72,       # Maximum gap in data (hours)
            'required_fields': ['open', 'high', 'low', 'close']
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'data_quality_score': 0.95
        }
        
        logger.info("Historical Data Manager initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Market data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL,
                        timeframe TEXT NOT NULL,
                        source TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timestamp, timeframe, source)
                    )
                ''')
                
                # Analysis results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        analysis_type TEXT NOT NULL,
                        result TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        accuracy_score REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Economic indicators table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS economic_indicators (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        indicator_name TEXT NOT NULL,
                        country TEXT NOT NULL,
                        value REAL NOT NULL,
                        release_date DATETIME NOT NULL,
                        source TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(indicator_name, country, release_date, source)
                    )
                ''')
                
                # News events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS news_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        content TEXT,
                        source TEXT NOT NULL,
                        published_at DATETIME NOT NULL,
                        sentiment_score REAL,
                        impact_level TEXT,
                        currencies TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Performance tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_id TEXT NOT NULL,
                        prediction_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        prediction_date DATETIME NOT NULL,
                        prediction_direction TEXT NOT NULL,
                        prediction_confidence REAL NOT NULL,
                        actual_direction TEXT,
                        actual_return REAL,
                        accuracy_score REAL,
                        evaluated_at DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_symbol_time ON analysis_results(symbol, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_agent ON performance_tracking(agent_id)')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    async def get_market_data(self, symbol: str, timeframe: str, 
                            start_date: datetime, end_date: datetime,
                            source: str = 'auto') -> pd.DataFrame:
        """
        Get market data with intelligent caching and source selection.
        
        Args:
            symbol: Currency pair symbol
            timeframe: Data timeframe ('1m', '5m', '1h', '1d', etc.)
            start_date: Start date for data
            end_date: End date for data
            source: Data source ('auto', 'alpha_vantage', 'yahoo_finance')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check cache first
            cache_key = f"market_data:{symbol}:{timeframe}:{start_date.date()}:{end_date.date()}"
            cached_data = await self._get_from_cache(cache_key)
            
            if cached_data is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_data
            
            self.performance_stats['cache_misses'] += 1
            
            # Check database
            db_data = await self._get_from_database(symbol, timeframe, start_date, end_date)
            
            if not db_data.empty:
                # Check if we have complete data
                expected_periods = self._calculate_expected_periods(start_date, end_date, timeframe)
                data_completeness = len(db_data) / expected_periods
                
                if data_completeness > 0.95:  # 95% complete
                    await self._cache_data(cache_key, db_data)
                    return db_data
            
            # Fetch from external source
            if source == 'auto':
                source = self._select_best_source(symbol, timeframe)
            
            external_data = await self._fetch_from_source(symbol, timeframe, start_date, end_date, source)
            
            if not external_data.empty:
                # Validate and store data
                validated_data = self._validate_market_data(external_data)
                await self._store_market_data(validated_data, source)
                await self._cache_data(cache_key, validated_data)
                return validated_data
            
            # Return whatever we have from database
            return db_data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from Redis cache."""
        try:
            cached_json = self.redis_client.get(cache_key)
            if cached_json:
                data_dict = json.loads(cached_json)
                df = pd.DataFrame(data_dict)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                return df
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval error: {str(e)}")
            return None
    
    async def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data in Redis."""
        try:
            # Convert to JSON-serializable format
            data_copy = data.copy()
            if data_copy.index.name == 'timestamp':
                data_copy.reset_index(inplace=True)
                data_copy['timestamp'] = data_copy['timestamp'].dt.isoformat()
            
            data_json = json.dumps(data_copy.to_dict(orient='records'))
            
            # Determine TTL based on data type
            ttl = self.cache_settings['daily_data_ttl']
            
            self.redis_client.setex(cache_key, ttl, data_json)
            
        except Exception as e:
            logger.warning(f"Cache storage error: {str(e)}")
    
    async def _get_from_database(self, symbol: str, timeframe: str, 
                               start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get data from SQLite database."""
        try:
            query = '''
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ? AND timeframe = ? 
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(symbol, timeframe, start_date, end_date),
                    parse_dates=['timestamp']
                )
            
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Database retrieval error: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_expected_periods(self, start_date: datetime, end_date: datetime, timeframe: str) -> int:
        """Calculate expected number of data periods."""
        delta = end_date - start_date
        
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60,
            '4h': 240, '1d': 1440, '1w': 10080
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        total_minutes = delta.total_seconds() / 60
        
        # Account for weekends (forex is closed)
        if timeframe in ['1d', '1w']:
            weekdays = np.busday_count(start_date.date(), end_date.date())
            return weekdays
        else:
            # Rough estimate: 5/7 of total time (excluding weekends)
            return int(total_minutes / minutes * 5/7)
    
    def _select_best_source(self, symbol: str, timeframe: str) -> str:
        """Select the best data source based on availability and reliability."""
        # For forex data, prioritize based on timeframe
        if timeframe in ['1m', '5m', '15m']:
            # High-frequency data - prefer Alpha Vantage
            return 'alpha_vantage' if self.data_sources['alpha_vantage'].api_key else 'yahoo_finance'
        else:
            # Daily data - Yahoo Finance is reliable and free
            return 'yahoo_finance'
    
    async def _fetch_from_source(self, symbol: str, timeframe: str, 
                               start_date: datetime, end_date: datetime, source: str) -> pd.DataFrame:
        """Fetch data from external source."""
        try:
            self.performance_stats['total_requests'] += 1
            
            if source == 'yahoo_finance':
                return await self._fetch_yahoo_finance(symbol, timeframe, start_date, end_date)
            elif source == 'alpha_vantage':
                return await self._fetch_alpha_vantage(symbol, timeframe, start_date, end_date)
            else:
                logger.warning(f"Unknown data source: {source}")
                return pd.DataFrame()
                
        except Exception as e:
            self.performance_stats['failed_requests'] += 1
            logger.error(f"Error fetching from {source}: {str(e)}")
            return pd.DataFrame()
    
    async def _fetch_yahoo_finance(self, symbol: str, timeframe: str, 
                                 start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
            # Convert forex symbol format (EURUSD -> EURUSD=X)
            yahoo_symbol = f"{symbol}=X"
            
            # Map timeframe to Yahoo Finance intervals
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if not data.empty:
                # Standardize column names
                data = data.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                })
                
                # Ensure required columns exist
                for col in ['open', 'high', 'low', 'close']:
                    if col not in data.columns:
                        logger.warning(f"Missing column {col} in Yahoo Finance data")
                        return pd.DataFrame()
                
                if 'volume' not in data.columns:
                    data['volume'] = 0  # Forex typically doesn't have volume
                
                data['symbol'] = symbol
                data['timeframe'] = timeframe
                
                self.performance_stats['successful_requests'] += 1
                return data
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Yahoo Finance fetch error: {str(e)}")
            return pd.DataFrame()
    
    async def _fetch_alpha_vantage(self, symbol: str, timeframe: str, 
                                 start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from Alpha Vantage."""
        try:
            if not self.data_sources['alpha_vantage'].api_key:
                logger.warning("Alpha Vantage API key not configured")
                return pd.DataFrame()
            
            # Map timeframe to Alpha Vantage function
            function_map = {
                '1m': 'FX_INTRADAY', '5m': 'FX_INTRADAY', '15m': 'FX_INTRADAY',
                '30m': 'FX_INTRADAY', '1h': 'FX_INTRADAY', '1d': 'FX_DAILY'
            }
            
            function = function_map.get(timeframe, 'FX_DAILY')
            
            # Prepare parameters
            params = {
                'function': function,
                'from_symbol': symbol[:3],
                'to_symbol': symbol[3:6],
                'apikey': self.data_sources['alpha_vantage'].api_key,
                'outputsize': 'full'
            }
            
            if function == 'FX_INTRADAY':
                params['interval'] = timeframe
            
            # Make request
            response = requests.get(self.data_sources['alpha_vantage'].endpoint, params=params)
            
            if response.status_code == 200:
                data_json = response.json()
                
                # Extract time series data
                if function == 'FX_INTRADAY':
                    time_series_key = f'Time Series FX ({timeframe})'
                else:
                    time_series_key = 'Time Series FX (Daily)'
                
                if time_series_key in data_json:
                    time_series = data_json[time_series_key]
                    
                    # Convert to DataFrame
                    df_data = []
                    for timestamp, values in time_series.items():
                        df_data.append({
                            'timestamp': pd.to_datetime(timestamp),
                            'open': float(values.get('1. open', 0)),
                            'high': float(values.get('2. high', 0)),
                            'low': float(values.get('3. low', 0)),
                            'close': float(values.get('4. close', 0)),
                            'volume': 0,  # Forex doesn't have volume in Alpha Vantage
                            'symbol': symbol,
                            'timeframe': timeframe
                        })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                    
                    # Filter by date range
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    self.performance_stats['successful_requests'] += 1
                    return df
                else:
                    logger.warning(f"Unexpected Alpha Vantage response format: {list(data_json.keys())}")
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Alpha Vantage fetch error: {str(e)}")
            return pd.DataFrame()
    
    def _validate_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate market data quality and fix common issues."""
        try:
            if data.empty:
                return data
            
            validated_data = data.copy()
            
            # Check for required fields
            for field in self.validation_rules['required_fields']:
                if field not in validated_data.columns:
                    logger.warning(f"Missing required field: {field}")
                    return pd.DataFrame()
            
            # Remove rows with invalid prices
            validated_data = validated_data[validated_data['high'] >= validated_data['low']]
            validated_data = validated_data[validated_data['high'] >= validated_data['open']]
            validated_data = validated_data[validated_data['high'] >= validated_data['close']]
            validated_data = validated_data[validated_data['low'] <= validated_data['open']]
            validated_data = validated_data[validated_data['low'] <= validated_data['close']]
            
            # Remove extreme price changes
            if len(validated_data) > 1:
                price_changes = validated_data['close'].pct_change().abs()
                max_change = self.validation_rules['max_price_change']
                validated_data = validated_data[price_changes <= max_change]
            
            # Fill missing volume with 0 (common for forex)
            if 'volume' in validated_data.columns:
                validated_data['volume'] = validated_data['volume'].fillna(0)
            
            # Remove duplicates
            validated_data = validated_data[~validated_data.index.duplicated(keep='first')]
            
            # Sort by timestamp
            validated_data = validated_data.sort_index()
            
            logger.info(f"Data validation completed: {len(data)} -> {len(validated_data)} records")
            
            return validated_data
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            return pd.DataFrame()
    
    async def _store_market_data(self, data: pd.DataFrame, source: str):
        """Store market data in database."""
        try:
            if data.empty:
                return
            
            # Prepare data for insertion
            data_copy = data.copy()
            data_copy.reset_index(inplace=True)
            data_copy['source'] = source
            
            # Convert to list of tuples for efficient insertion
            records = []
            for _, row in data_copy.iterrows():
                records.append((
                    row['symbol'], row['timestamp'], row['open'], row['high'],
                    row['low'], row['close'], row.get('volume', 0),
                    row['timeframe'], row['source']
                ))
            
            # Insert into database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany('''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume, timeframe, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', records)
                conn.commit()
            
            logger.info(f"Stored {len(records)} market data records")
            
        except Exception as e:
            logger.error(f"Error storing market data: {str(e)}")
    
    async def store_analysis_result(self, analysis: AnalysisRecord):
        """Store analysis result in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO analysis_results 
                    (id, agent_id, symbol, timestamp, analysis_type, result, confidence, accuracy_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis.id, analysis.agent_id, analysis.symbol,
                    analysis.timestamp, analysis.analysis_type,
                    json.dumps(analysis.result), analysis.confidence,
                    analysis.accuracy_score
                ))
                conn.commit()
            
            logger.info(f"Stored analysis result: {analysis.id}")
            
        except Exception as e:
            logger.error(f"Error storing analysis result: {str(e)}")
    
    async def get_analysis_history(self, agent_id: str = None, symbol: str = None, 
                                 days_back: int = 30) -> List[AnalysisRecord]:
        """Get analysis history with optional filtering."""
        try:
            query = '''
                SELECT id, agent_id, symbol, timestamp, analysis_type, result, confidence, accuracy_score
                FROM analysis_results
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days_back)
            
            params = []
            
            if agent_id:
                query += ' AND agent_id = ?'
                params.append(agent_id)
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            query += ' ORDER BY timestamp DESC'
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
            
            # Convert to AnalysisRecord objects
            records = []
            for row in rows:
                record = AnalysisRecord(
                    id=row[0],
                    agent_id=row[1],
                    symbol=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    analysis_type=row[4],
                    result=json.loads(row[5]),
                    confidence=row[6],
                    accuracy_score=row[7]
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving analysis history: {str(e)}")
            return []
    
    async def update_analysis_accuracy(self, analysis_id: str, accuracy_score: float):
        """Update accuracy score for an analysis after validation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE analysis_results 
                    SET accuracy_score = ?
                    WHERE id = ?
                ''', (accuracy_score, analysis_id))
                conn.commit()
            
            logger.info(f"Updated accuracy for analysis {analysis_id}: {accuracy_score}")
            
        except Exception as e:
            logger.error(f"Error updating analysis accuracy: {str(e)}")
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get data statistics
                cursor.execute('''
                    SELECT 
                        symbol,
                        COUNT(*) as total_records,
                        MIN(timestamp) as earliest_date,
                        MAX(timestamp) as latest_date,
                        COUNT(DISTINCT source) as source_count
                    FROM market_data
                    GROUP BY symbol
                ''')
                data_stats = cursor.fetchall()
                
                # Get analysis statistics
                cursor.execute('''
                    SELECT 
                        agent_id,
                        COUNT(*) as total_analyses,
                        AVG(confidence) as avg_confidence,
                        AVG(accuracy_score) as avg_accuracy
                    FROM analysis_results
                    WHERE accuracy_score IS NOT NULL
                    GROUP BY agent_id
                ''')
                analysis_stats = cursor.fetchall()
            
            report = {
                'data_statistics': {
                    'symbols': [
                        {
                            'symbol': stat[0],
                            'total_records': stat[1],
                            'earliest_date': stat[2],
                            'latest_date': stat[3],
                            'source_count': stat[4]
                        }
                        for stat in data_stats
                    ]
                },
                'analysis_statistics': {
                    'agents': [
                        {
                            'agent_id': stat[0],
                            'total_analyses': stat[1],
                            'avg_confidence': round(stat[2], 3) if stat[2] else None,
                            'avg_accuracy': round(stat[3], 3) if stat[3] else None
                        }
                        for stat in analysis_stats
                    ]
                },
                'performance_stats': self.performance_stats,
                'cache_efficiency': {
                    'hit_rate': self.performance_stats['cache_hits'] / 
                               (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'])
                               if (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) > 0 else 0,
                    'total_requests': self.performance_stats['total_requests']
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {str(e)}")
            return {}
    
    async def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to manage storage."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean old market data (keep more recent data)
                cursor.execute('''
                    DELETE FROM market_data 
                    WHERE timestamp < ? AND timeframe IN ('1m', '5m', '15m')
                ''', (cutoff_date - timedelta(days=30),))  # Keep intraday data for 30 days
                
                # Clean old analysis results
                cursor.execute('''
                    DELETE FROM analysis_results 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                
                # Clean old news events
                cursor.execute('''
                    DELETE FROM news_events 
                    WHERE published_at < ?
                ''', (cutoff_date,))
                
                conn.commit()
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    async def backup_database(self, backup_path: str = None):
        """Create database backup."""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.data_dir / f"backup_forex_data_{timestamp}.db"
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error backing up database: {str(e)}")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        return {
            'data_sources': {name: source.name for name, source in self.data_sources.items()},
            'performance_stats': self.performance_stats,
            'validation_rules': self.validation_rules,
            'cache_settings': self.cache_settings,
            'database_path': str(self.db_path),
            'data_directory': str(self.data_dir)
        }