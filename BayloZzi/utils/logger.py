# utils/logger.py
import logging
import logging.handlers
import os
import json
from datetime import datetime
import sys
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'trade_id'):
            log_entry['trade_id'] = record.trade_id
        if hasattr(record, 'symbol'):
            log_entry['symbol'] = record.symbol
        if hasattr(record, 'price'):
            log_entry['price'] = record.price
        if hasattr(record, 'pnl'):
            log_entry['pnl'] = record.pnl
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, default=str)

class TradingLogger:
    """
    Production-grade logging system for forex trading.
    Implements structured logging with multiple outputs and log levels.
    """
    
    def __init__(self, name: str = "BayloZzi", log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir
        self.logger = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up comprehensive logging configuration."""
        # Create logs directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for all logs (JSON format)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(self.log_dir, 'trading.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)
        
        # Separate handler for trading-specific logs
        trading_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(self.log_dir, 'trades.log'),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10
        )
        trading_handler.setLevel(logging.INFO)
        trading_handler.setFormatter(JSONFormatter())
        trading_handler.addFilter(self._trading_filter)
        self.logger.addHandler(trading_handler)
        
        # Error handler for critical issues
        error_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(self.log_dir, 'errors.log'),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(error_handler)
    
    def _trading_filter(self, record):
        """Filter for trading-related logs."""
        trading_keywords = ['trade', 'position', 'signal', 'pnl', 'entry', 'exit', 'stop', 'profit']
        message = record.getMessage().lower()
        return any(keyword in message for keyword in trading_keywords)
    
    def log_trade_signal(self, symbol: str, signal: int, confidence: float, 
                        price: float, additional_data: Dict = None):
        """Log a trading signal with structured data."""
        extra_data = {
            'symbol': symbol,
            'signal': 'BUY' if signal == 1 else 'SELL',
            'confidence': confidence,
            'price': price
        }
        if additional_data:
            extra_data.update(additional_data)
            
        self.logger.info(
            f"Trading signal: {extra_data['signal']} {symbol} at {price:.5f} (confidence: {confidence:.3f})",
            extra=extra_data
        )

# Global logger instance
_global_logger = None

def get_logger(name: str = "BayloZzi") -> TradingLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = TradingLogger(name)
    return _global_logger

# For backward compatibility
logger = get_logger().logger
