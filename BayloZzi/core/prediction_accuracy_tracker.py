# core/prediction_accuracy_tracker.py

import asyncio
import logging
import numpy as np
import pandas as pd
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    DIRECTION = "direction"          # Bullish/Bearish/Neutral
    PRICE_TARGET = "price_target"    # Specific price levels
    VOLATILITY = "volatility"        # Volatility forecast
    TIMEFRAME = "timeframe"          # Time to reach target
    RISK_LEVEL = "risk_level"        # Risk assessment

class AccuracyMetric(Enum):
    DIRECTIONAL = "directional"      # Did price move in predicted direction?
    MAGNITUDE = "magnitude"          # How close was the predicted magnitude?
    TIMING = "timing"               # How accurate was the timing?
    CONFIDENCE = "confidence"       # How well calibrated was confidence?
    OVERALL = "overall"             # Combined accuracy score

@dataclass
class Prediction:
    id: str
    agent_id: str
    symbol: str
    timestamp: datetime
    prediction_type: PredictionType
    predicted_direction: str        # 'bullish', 'bearish', 'neutral'
    predicted_target: Optional[float] = None
    predicted_timeframe: Optional[str] = None
    confidence: float = 0.5
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    expiry_date: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class PredictionOutcome:
    prediction_id: str
    evaluation_date: datetime
    actual_direction: str
    actual_price_change: float
    actual_max_gain: float
    actual_max_loss: float
    time_to_target: Optional[float] = None  # Hours
    target_hit: bool = False
    stop_loss_hit: bool = False
    accuracy_scores: Dict[AccuracyMetric, float] = None

@dataclass
class AgentPerformance:
    agent_id: str
    total_predictions: int
    directional_accuracy: float
    target_accuracy: float
    avg_confidence: float
    confidence_calibration: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_holding_period: float
    best_prediction_accuracy: float
    worst_prediction_accuracy: float
    recent_trend: str  # 'improving', 'declining', 'stable'

class PredictionAccuracyTracker:
    """
    Comprehensive Prediction Accuracy Tracking System for Multi-Agent Forex Trading.
    Tracks, evaluates, and analyzes prediction performance across all agents.
    """
    
    def __init__(self, db_path: str = "data/forex_data.db"):
        self.db_path = db_path
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Active predictions waiting for evaluation
        self.active_predictions = {}
        
        # Performance tracking
        self.agent_performance = {}
        self.system_performance = {
            'total_predictions': 0,
            'overall_accuracy': 0.0,
            'best_performing_agent': None,
            'accuracy_trend': [],
            'last_evaluation': None
        }
        
        # Evaluation parameters
        self.evaluation_params = {
            'min_evaluation_period': 1,     # Minimum hours before evaluation
            'max_evaluation_period': 168,   # Maximum hours (1 week)
            'price_tolerance': 0.001,       # 0.1% tolerance for target hits
            'confidence_bins': 10,          # Bins for confidence calibration
            'trend_lookback_days': 30       # Days to look back for trend analysis
        }
        
        # Accuracy calculation weights
        self.accuracy_weights = {
            AccuracyMetric.DIRECTIONAL: 0.4,
            AccuracyMetric.MAGNITUDE: 0.25,
            AccuracyMetric.TIMING: 0.15,
            AccuracyMetric.CONFIDENCE: 0.2
        }
        
        # Initialize database tables
        self._initialize_tracking_tables()
        
        # Load active predictions
        self._load_active_predictions()
        
        logger.info("Prediction Accuracy Tracker initialized")
    
    def _initialize_tracking_tables(self):
        """Initialize database tables for prediction tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        prediction_type TEXT NOT NULL,
                        predicted_direction TEXT NOT NULL,
                        predicted_target REAL,
                        predicted_timeframe TEXT,
                        confidence REAL NOT NULL,
                        current_price REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        expiry_date DATETIME,
                        metadata TEXT,
                        status TEXT DEFAULT 'active',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Prediction outcomes table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS prediction_outcomes (
                        prediction_id TEXT PRIMARY KEY,
                        evaluation_date DATETIME NOT NULL,
                        actual_direction TEXT NOT NULL,
                        actual_price_change REAL NOT NULL,
                        actual_max_gain REAL NOT NULL,
                        actual_max_loss REAL NOT NULL,
                        time_to_target REAL,
                        target_hit BOOLEAN NOT NULL,
                        stop_loss_hit BOOLEAN NOT NULL,
                        directional_accuracy REAL NOT NULL,
                        magnitude_accuracy REAL NOT NULL,
                        timing_accuracy REAL NOT NULL,
                        confidence_accuracy REAL NOT NULL,
                        overall_accuracy REAL NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (prediction_id) REFERENCES predictions (id)
                    )
                ''')
                
                # Agent performance summary table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS agent_performance (
                        agent_id TEXT PRIMARY KEY,
                        total_predictions INTEGER NOT NULL,
                        directional_accuracy REAL NOT NULL,
                        target_accuracy REAL NOT NULL,
                        avg_confidence REAL NOT NULL,
                        confidence_calibration REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        win_rate REAL NOT NULL,
                        avg_holding_period REAL NOT NULL,
                        best_prediction_accuracy REAL NOT NULL,
                        worst_prediction_accuracy REAL NOT NULL,
                        recent_trend TEXT NOT NULL,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_agent ON predictions(agent_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcomes_evaluation ON prediction_outcomes(evaluation_date)')
                
                conn.commit()
                logger.info("Prediction tracking tables initialized")
                
        except Exception as e:
            logger.error(f"Error initializing tracking tables: {str(e)}")
    
    def _load_active_predictions(self):
        """Load active predictions from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM predictions 
                    WHERE status = 'active' 
                    AND (expiry_date IS NULL OR expiry_date > datetime('now'))
                ''')
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    
                    prediction = Prediction(
                        id=row_dict['id'],
                        agent_id=row_dict['agent_id'],
                        symbol=row_dict['symbol'],
                        timestamp=datetime.fromisoformat(row_dict['timestamp']),
                        prediction_type=PredictionType(row_dict['prediction_type']),
                        predicted_direction=row_dict['predicted_direction'],
                        predicted_target=row_dict['predicted_target'],
                        predicted_timeframe=row_dict['predicted_timeframe'],
                        confidence=row_dict['confidence'],
                        current_price=row_dict['current_price'],
                        stop_loss=row_dict['stop_loss'],
                        take_profit=row_dict['take_profit'],
                        expiry_date=datetime.fromisoformat(row_dict['expiry_date']) if row_dict['expiry_date'] else None,
                        metadata=json.loads(row_dict['metadata']) if row_dict['metadata'] else {}
                    )
                    
                    self.active_predictions[prediction.id] = prediction
            
            logger.info(f"Loaded {len(self.active_predictions)} active predictions")
            
        except Exception as e:
            logger.error(f"Error loading active predictions: {str(e)}")
    
    async def record_prediction(self, prediction: Prediction) -> str:
        """Record a new prediction for tracking."""
        try:
            # Generate ID if not provided
            if not prediction.id:
                prediction.id = f"{prediction.agent_id}_{prediction.symbol}_{int(prediction.timestamp.timestamp())}"
            
            # Set expiry date if not provided (default 1 week)
            if not prediction.expiry_date:
                prediction.expiry_date = prediction.timestamp + timedelta(days=7)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO predictions 
                    (id, agent_id, symbol, timestamp, prediction_type, predicted_direction,
                     predicted_target, predicted_timeframe, confidence, current_price,
                     stop_loss, take_profit, expiry_date, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction.id, prediction.agent_id, prediction.symbol,
                    prediction.timestamp.isoformat(), prediction.prediction_type.value,
                    prediction.predicted_direction, prediction.predicted_target,
                    prediction.predicted_timeframe, prediction.confidence,
                    prediction.current_price, prediction.stop_loss,
                    prediction.take_profit, prediction.expiry_date.isoformat(),
                    json.dumps(prediction.metadata) if prediction.metadata else None
                ))
                conn.commit()
            
            # Add to active predictions
            self.active_predictions[prediction.id] = prediction
            
            logger.info(f"Recorded prediction {prediction.id} from {prediction.agent_id}")
            return prediction.id
            
        except Exception as e:
            logger.error(f"Error recording prediction: {str(e)}")
            return ""
    
    async def evaluate_predictions(self, current_prices: Dict[str, float]) -> List[PredictionOutcome]:
        """Evaluate all active predictions against current market data."""
        try:
            outcomes = []
            predictions_to_remove = []
            
            current_time = datetime.now()
            
            for pred_id, prediction in self.active_predictions.items():
                # Check if prediction should be evaluated
                if self._should_evaluate_prediction(prediction, current_time):
                    
                    current_price = current_prices.get(prediction.symbol)
                    if current_price is None:
                        continue
                    
                    # Evaluate the prediction
                    outcome = await self._evaluate_single_prediction(prediction, current_price, current_time)
                    
                    if outcome:
                        outcomes.append(outcome)
                        predictions_to_remove.append(pred_id)
                        
                        # Store outcome in database
                        await self._store_prediction_outcome(outcome)
            
            # Remove evaluated predictions
            for pred_id in predictions_to_remove:
                del self.active_predictions[pred_id]
                await self._mark_prediction_completed(pred_id)
            
            if outcomes:
                # Update agent performance metrics
                await self._update_agent_performance(outcomes)
                logger.info(f"Evaluated {len(outcomes)} predictions")
            
            return outcomes
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {str(e)}")
            return []
    
    def _should_evaluate_prediction(self, prediction: Prediction, current_time: datetime) -> bool:
        """Determine if a prediction should be evaluated."""
        # Check minimum evaluation period
        time_elapsed = (current_time - prediction.timestamp).total_seconds() / 3600  # hours
        
        if time_elapsed < self.evaluation_params['min_evaluation_period']:
            return False
        
        # Check if expired
        if prediction.expiry_date and current_time > prediction.expiry_date:
            return True
        
        # Check if maximum evaluation period reached
        if time_elapsed > self.evaluation_params['max_evaluation_period']:
            return True
        
        # For timeframe-specific predictions, check if timeframe elapsed
        if prediction.predicted_timeframe:
            timeframe_hours = self._parse_timeframe_to_hours(prediction.predicted_timeframe)
            if timeframe_hours and time_elapsed >= timeframe_hours:
                return True
        
        return False
    
    def _parse_timeframe_to_hours(self, timeframe: str) -> Optional[float]:
        """Parse timeframe string to hours."""
        timeframe_map = {
            '1h': 1, '4h': 4, '1d': 24, '1w': 168,
            'hour': 1, 'day': 24, 'week': 168
        }
        return timeframe_map.get(timeframe.lower())
    
    async def _evaluate_single_prediction(self, prediction: Prediction, 
                                        current_price: float, evaluation_time: datetime) -> Optional[PredictionOutcome]:
        """Evaluate a single prediction."""
        try:
            # Calculate price change
            price_change = (current_price - prediction.current_price) / prediction.current_price
            
            # Determine actual direction
            if price_change > 0.001:  # 0.1% threshold
                actual_direction = 'bullish'
            elif price_change < -0.001:
                actual_direction = 'bearish'
            else:
                actual_direction = 'neutral'
            
            # Get historical price data to calculate max gain/loss
            max_gain, max_loss = await self._calculate_max_gain_loss(
                prediction.symbol, prediction.timestamp, evaluation_time, prediction.current_price
            )
            
            # Check target and stop loss hits
            target_hit = self._check_target_hit(prediction, current_price, max_gain, max_loss)
            stop_loss_hit = self._check_stop_loss_hit(prediction, current_price, max_gain, max_loss)
            
            # Calculate time to target (if applicable)
            time_to_target = None
            if target_hit:
                time_to_target = (evaluation_time - prediction.timestamp).total_seconds() / 3600
            
            # Calculate accuracy scores
            accuracy_scores = self._calculate_accuracy_scores(
                prediction, actual_direction, price_change, time_to_target, target_hit
            )
            
            return PredictionOutcome(
                prediction_id=prediction.id,
                evaluation_date=evaluation_time,
                actual_direction=actual_direction,
                actual_price_change=price_change,
                actual_max_gain=max_gain,
                actual_max_loss=max_loss,
                time_to_target=time_to_target,
                target_hit=target_hit,
                stop_loss_hit=stop_loss_hit,
                accuracy_scores=accuracy_scores
            )
            
        except Exception as e:
            logger.error(f"Error evaluating prediction {prediction.id}: {str(e)}")
            return None
    
    async def _calculate_max_gain_loss(self, symbol: str, start_time: datetime, 
                                     end_time: datetime, entry_price: float) -> Tuple[float, float]:
        """Calculate maximum gain and loss during the prediction period."""
        try:
            # In production, this would fetch actual market data
            # For demo, simulate realistic max gain/loss
            
            time_hours = (end_time - start_time).total_seconds() / 3600
            volatility = 0.015  # Assume 1.5% daily volatility
            
            # Simulate random walk with volatility
            max_gain = min(volatility * np.sqrt(time_hours / 24) * 2, 0.05)  # Cap at 5%
            max_loss = -min(volatility * np.sqrt(time_hours / 24) * 2, 0.05)  # Cap at -5%
            
            return max_gain, max_loss
            
        except Exception as e:
            logger.error(f"Error calculating max gain/loss: {str(e)}")
            return 0.0, 0.0
    
    def _check_target_hit(self, prediction: Prediction, current_price: float, 
                         max_gain: float, max_loss: float) -> bool:
        """Check if prediction target was hit."""
        if not prediction.predicted_target:
            return False
        
        target_change = (prediction.predicted_target - prediction.current_price) / prediction.current_price
        tolerance = self.evaluation_params['price_tolerance']
        
        if prediction.predicted_direction == 'bullish':
            return max_gain >= (target_change - tolerance)
        elif prediction.predicted_direction == 'bearish':
            return max_loss <= (target_change + tolerance)
        
        return False
    
    def _check_stop_loss_hit(self, prediction: Prediction, current_price: float, 
                           max_gain: float, max_loss: float) -> bool:
        """Check if stop loss was hit."""
        if not prediction.stop_loss:
            return False
        
        stop_change = (prediction.stop_loss - prediction.current_price) / prediction.current_price
        
        if prediction.predicted_direction == 'bullish':
            return max_loss <= stop_change
        elif prediction.predicted_direction == 'bearish':
            return max_gain >= stop_change
        
        return False
    
    def _calculate_accuracy_scores(self, prediction: Prediction, actual_direction: str, 
                                 actual_change: float, time_to_target: Optional[float], 
                                 target_hit: bool) -> Dict[AccuracyMetric, float]:
        """Calculate comprehensive accuracy scores."""
        scores = {}
        
        # Directional accuracy
        if prediction.predicted_direction == actual_direction:
            scores[AccuracyMetric.DIRECTIONAL] = 1.0
        elif (prediction.predicted_direction == 'neutral' and abs(actual_change) < 0.005) or \
             (actual_direction == 'neutral' and abs(actual_change) < 0.005):
            scores[AccuracyMetric.DIRECTIONAL] = 0.5  # Partial credit for near-neutral
        else:
            scores[AccuracyMetric.DIRECTIONAL] = 0.0
        
        # Magnitude accuracy (for price targets)
        if prediction.predicted_target:
            predicted_change = (prediction.predicted_target - prediction.current_price) / prediction.current_price
            magnitude_error = abs(predicted_change - actual_change)
            scores[AccuracyMetric.MAGNITUDE] = max(0, 1 - magnitude_error / 0.05)  # Scale by 5%
        else:
            scores[AccuracyMetric.MAGNITUDE] = 0.5  # Neutral if no target
        
        # Timing accuracy
        if time_to_target and prediction.predicted_timeframe:
            predicted_hours = self._parse_timeframe_to_hours(prediction.predicted_timeframe)
            if predicted_hours:
                timing_error = abs(time_to_target - predicted_hours) / predicted_hours
                scores[AccuracyMetric.TIMING] = max(0, 1 - timing_error)
            else:
                scores[AccuracyMetric.TIMING] = 0.5
        else:
            scores[AccuracyMetric.TIMING] = 0.5  # Neutral if no timing prediction
        
        # Confidence calibration (simplified)
        directional_correct = scores[AccuracyMetric.DIRECTIONAL] > 0.5
        confidence_accuracy = 1 - abs(prediction.confidence - (1.0 if directional_correct else 0.0))
        scores[AccuracyMetric.CONFIDENCE] = max(0, confidence_accuracy)
        
        # Overall accuracy (weighted combination)
        overall = sum(score * self.accuracy_weights.get(metric, 0.25) 
                     for metric, score in scores.items())
        scores[AccuracyMetric.OVERALL] = overall
        
        return scores
    
    async def _store_prediction_outcome(self, outcome: PredictionOutcome):
        """Store prediction outcome in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO prediction_outcomes 
                    (prediction_id, evaluation_date, actual_direction, actual_price_change,
                     actual_max_gain, actual_max_loss, time_to_target, target_hit,
                     stop_loss_hit, directional_accuracy, magnitude_accuracy, timing_accuracy,
                     confidence_accuracy, overall_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    outcome.prediction_id, outcome.evaluation_date.isoformat(),
                    outcome.actual_direction, outcome.actual_price_change,
                    outcome.actual_max_gain, outcome.actual_max_loss,
                    outcome.time_to_target, outcome.target_hit, outcome.stop_loss_hit,
                    outcome.accuracy_scores[AccuracyMetric.DIRECTIONAL],
                    outcome.accuracy_scores[AccuracyMetric.MAGNITUDE],
                    outcome.accuracy_scores[AccuracyMetric.TIMING],
                    outcome.accuracy_scores[AccuracyMetric.CONFIDENCE],
                    outcome.accuracy_scores[AccuracyMetric.OVERALL]
                ))
                conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing prediction outcome: {str(e)}")
    
    async def _mark_prediction_completed(self, prediction_id: str):
        """Mark prediction as completed in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE predictions SET status = 'completed' WHERE id = ?
                ''', (prediction_id,))
                conn.commit()
            
        except Exception as e:
            logger.error(f"Error marking prediction completed: {str(e)}")
    
    async def _update_agent_performance(self, outcomes: List[PredictionOutcome]):
        """Update agent performance metrics based on new outcomes."""
        try:
            agent_outcomes = {}
            
            # Group outcomes by agent
            for outcome in outcomes:
                # Get prediction details
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT agent_id FROM predictions WHERE id = ?', (outcome.prediction_id,))
                    result = cursor.fetchone()
                    
                    if result:
                        agent_id = result[0]
                        if agent_id not in agent_outcomes:
                            agent_outcomes[agent_id] = []
                        agent_outcomes[agent_id].append(outcome)
            
            # Update performance for each agent
            for agent_id, agent_outcomes_list in agent_outcomes.items():
                await self._calculate_agent_performance(agent_id)
            
        except Exception as e:
            logger.error(f"Error updating agent performance: {str(e)}")
    
    async def _calculate_agent_performance(self, agent_id: str) -> AgentPerformance:
        """Calculate comprehensive performance metrics for an agent."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all outcomes for this agent
                cursor.execute('''
                    SELECT po.*, p.confidence, p.predicted_direction, p.timestamp
                    FROM prediction_outcomes po
                    JOIN predictions p ON po.prediction_id = p.id
                    WHERE p.agent_id = ?
                    ORDER BY po.evaluation_date DESC
                ''', (agent_id,))
                
                rows = cursor.fetchall()
                
                if not rows:
                    return AgentPerformance(
                        agent_id=agent_id, total_predictions=0, directional_accuracy=0,
                        target_accuracy=0, avg_confidence=0, confidence_calibration=0,
                        sharpe_ratio=0, max_drawdown=0, win_rate=0, avg_holding_period=0,
                        best_prediction_accuracy=0, worst_prediction_accuracy=0, recent_trend='stable'
                    )
                
                # Calculate metrics
                total_predictions = len(rows)
                directional_accuracies = [row[9] for row in rows]  # directional_accuracy column
                overall_accuracies = [row[13] for row in rows]     # overall_accuracy column
                confidences = [row[14] for row in rows]            # confidence from join
                
                directional_accuracy = np.mean(directional_accuracies)
                target_accuracy = np.mean([row[7] for row in rows])  # target_hit
                avg_confidence = np.mean(confidences)
                
                # Confidence calibration (simplified)
                confidence_calibration = 1 - np.mean([abs(conf - acc) for conf, acc in zip(confidences, directional_accuracies)])
                
                # Calculate returns for Sharpe ratio
                returns = [row[3] for row in rows]  # actual_price_change
                if len(returns) > 1:
                    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                    max_drawdown = self._calculate_max_drawdown(returns)
                else:
                    sharpe_ratio = 0
                    max_drawdown = 0
                
                win_rate = len([r for r in returns if r > 0]) / len(returns)
                
                # Average holding period (simplified)
                holding_periods = [row[6] for row in rows if row[6]]  # time_to_target
                avg_holding_period = np.mean(holding_periods) if holding_periods else 24  # Default 24h
                
                best_accuracy = np.max(overall_accuracies)
                worst_accuracy = np.min(overall_accuracies)
                
                # Recent trend analysis
                recent_trend = self._analyze_recent_trend(overall_accuracies[-10:])  # Last 10 predictions
                
                performance = AgentPerformance(
                    agent_id=agent_id,
                    total_predictions=total_predictions,
                    directional_accuracy=directional_accuracy,
                    target_accuracy=target_accuracy,
                    avg_confidence=avg_confidence,
                    confidence_calibration=confidence_calibration,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    win_rate=win_rate,
                    avg_holding_period=avg_holding_period,
                    best_prediction_accuracy=best_accuracy,
                    worst_prediction_accuracy=worst_accuracy,
                    recent_trend=recent_trend
                )
                
                # Store in database
                await self._store_agent_performance(performance)
                
                # Update in-memory cache
                self.agent_performance[agent_id] = performance
                
                return performance
                
        except Exception as e:
            logger.error(f"Error calculating agent performance: {str(e)}")
            return AgentPerformance(
                agent_id=agent_id, total_predictions=0, directional_accuracy=0,
                target_accuracy=0, avg_confidence=0, confidence_calibration=0,
                sharpe_ratio=0, max_drawdown=0, win_rate=0, avg_holding_period=0,
                best_prediction_accuracy=0, worst_prediction_accuracy=0, recent_trend='stable'
            )
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        return abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
    
    def _analyze_recent_trend(self, recent_accuracies: List[float]) -> str:
        """Analyze recent performance trend."""
        if len(recent_accuracies) < 3:
            return 'stable'
        
        # Simple trend analysis
        first_half = np.mean(recent_accuracies[:len(recent_accuracies)//2])
        second_half = np.mean(recent_accuracies[len(recent_accuracies)//2:])
        
        if second_half > first_half + 0.05:
            return 'improving'
        elif second_half < first_half - 0.05:
            return 'declining'
        else:
            return 'stable'
    
    async def _store_agent_performance(self, performance: AgentPerformance):
        """Store agent performance in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO agent_performance 
                    (agent_id, total_predictions, directional_accuracy, target_accuracy,
                     avg_confidence, confidence_calibration, sharpe_ratio, max_drawdown,
                     win_rate, avg_holding_period, best_prediction_accuracy,
                     worst_prediction_accuracy, recent_trend)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance.agent_id, performance.total_predictions,
                    performance.directional_accuracy, performance.target_accuracy,
                    performance.avg_confidence, performance.confidence_calibration,
                    performance.sharpe_ratio, performance.max_drawdown,
                    performance.win_rate, performance.avg_holding_period,
                    performance.best_prediction_accuracy, performance.worst_prediction_accuracy,
                    performance.recent_trend
                ))
                conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing agent performance: {str(e)}")
    
    def get_agent_performance(self, agent_id: str) -> Optional[AgentPerformance]:
        """Get performance metrics for a specific agent."""
        return self.agent_performance.get(agent_id)
    
    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get overall system performance summary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Overall statistics
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_predictions,
                        AVG(overall_accuracy) as avg_accuracy,
                        AVG(directional_accuracy) as avg_directional_accuracy,
                        AVG(confidence_accuracy) as avg_confidence_accuracy
                    FROM prediction_outcomes
                ''')
                
                overall_stats = cursor.fetchone()
                
                # Agent rankings
                cursor.execute('''
                    SELECT agent_id, directional_accuracy, total_predictions
                    FROM agent_performance
                    ORDER BY directional_accuracy DESC
                    LIMIT 5
                ''')
                
                top_agents = cursor.fetchall()
            
            return {
                'total_predictions': overall_stats[0] if overall_stats[0] else 0,
                'system_accuracy': overall_stats[1] if overall_stats[1] else 0,
                'directional_accuracy': overall_stats[2] if overall_stats[2] else 0,
                'confidence_accuracy': overall_stats[3] if overall_stats[3] else 0,
                'top_performing_agents': [
                    {'agent_id': agent[0], 'accuracy': agent[1], 'predictions': agent[2]}
                    for agent in top_agents
                ],
                'active_predictions': len(self.active_predictions),
                'evaluation_params': self.evaluation_params
            }
            
        except Exception as e:
            logger.error(f"Error getting system performance summary: {str(e)}")
            return {}
    
    async def generate_performance_report(self, agent_id: str = None, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            report = {
                'report_date': datetime.now().isoformat(),
                'period_days': days_back,
                'system_summary': self.get_system_performance_summary()
            }
            
            if agent_id:
                # Single agent report
                performance = await self._calculate_agent_performance(agent_id)
                report['agent_performance'] = asdict(performance)
                
                # Recent predictions
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT p.*, po.overall_accuracy, po.actual_direction
                        FROM predictions p
                        LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
                        WHERE p.agent_id = ? AND p.timestamp >= datetime('now', '-{} days')
                        ORDER BY p.timestamp DESC
                        LIMIT 20
                    '''.format(days_back), (agent_id,))
                    
                    recent_predictions = cursor.fetchall()
                    report['recent_predictions'] = [
                        {
                            'id': pred[0], 'symbol': pred[2], 'timestamp': pred[3],
                            'predicted_direction': pred[5], 'confidence': pred[8],
                            'accuracy': pred[-2] if pred[-2] else None,
                            'actual_direction': pred[-1] if pred[-1] else None
                        }
                        for pred in recent_predictions
                    ]
            else:
                # All agents report
                all_agents = {}
                for agent_id in self.agent_performance:
                    all_agents[agent_id] = asdict(self.agent_performance[agent_id])
                
                report['all_agents'] = all_agents
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {}
    
    def get_prediction_status(self) -> Dict[str, Any]:
        """Get current prediction tracking status."""
        return {
            'active_predictions': len(self.active_predictions),
            'predictions_by_agent': {
                agent_id: len([p for p in self.active_predictions.values() if p.agent_id == agent_id])
                for agent_id in set(p.agent_id for p in self.active_predictions.values())
            },
            'predictions_by_symbol': {
                symbol: len([p for p in self.active_predictions.values() if p.symbol == symbol])
                for symbol in set(p.symbol for p in self.active_predictions.values())
            },
            'evaluation_params': self.evaluation_params,
            'system_performance': self.system_performance
        }