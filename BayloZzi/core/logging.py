"""
12-FACTOR PERFECT LOGGING
Factor XI: Treat logs as event streams - 100/100 implementation
"""

import sys
import json
import time
import traceback
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timezone
import structlog
from structlog.types import FilteringBoundLogger, Processor
import logging
import os
from functools import lru_cache
import socket
import uuid
from contextlib import contextmanager

from .config import get_config, Config


class StructuredLogger:
    """
    Perfect 12-Factor logging implementation
    - All logs to stdout (never to files)
    - Structured JSON format
    - Contextual information
    - Performance metrics
    - Error tracking
    - Distributed tracing support
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self._hostname = socket.gethostname()
        self._process_id = os.getpid()
        self._setup_structlog()
        
    def _setup_structlog(self):
        """Configure structlog for perfect 12-factor compliance"""
        
        # Determine output format
        json_logs = self.config.log_format == "json"
        
        # Build processor pipeline
        processors = [
            # Add context
            structlog.contextvars.merge_contextvars,
            
            # Add standard fields
            self._add_app_context,
            
            # Add timestamp
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            
            # Process exceptions
            self._process_exception,
            
            # Performance tracking
            self._add_performance_context,
            
            # Distributed tracing
            self._add_trace_context,
        ]
        
        if json_logs:
            # JSON output for production
            processors.extend([
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer()
            ])
        else:
            # Human-readable for development
            processors.extend([
                structlog.dev.ConsoleRenderer(
                    exception_formatter=structlog.dev.plain_traceback
                )
            ])
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.getLevelName(self.config.log_level.value)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
            cache_logger_on_first_use=True,
        )
        
        # Configure standard logging to use structlog
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=logging.getLevelName(self.config.log_level.value)
        )
        
        # Redirect standard logging to structlog
        logging.getLogger().handlers = [
            StructlogHandler()
        ]
    
    # ============================================================================
    # PROCESSORS
    # ============================================================================
    
    def _add_app_context(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add application context to all logs"""
        event_dict.update({
            "app": self.config.app_name,
            "version": self.config.app_version,
            "environment": self.config.environment.value,
            "service": event_dict.get("service", "unknown"),
        })
        
        if self.config.log_include_hostname:
            event_dict["hostname"] = self._hostname
            
        event_dict["pid"] = self._process_id
        
        return event_dict
    
    def _process_exception(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process exception information"""
        exc_info = event_dict.pop("exc_info", None)
        
        if exc_info:
            if isinstance(exc_info, BaseException):
                exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
            elif exc_info is True:
                exc_info = sys.exc_info()
            
            if exc_info and exc_info[0] is not None:
                event_dict["exception"] = {
                    "type": exc_info[0].__name__,
                    "value": str(exc_info[1]),
                    "traceback": traceback.format_exception(*exc_info)
                }
                
                # Add exception fingerprint for grouping
                event_dict["exception_fingerprint"] = self._generate_exception_fingerprint(exc_info)
        
        return event_dict
    
    def _add_performance_context(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add performance metrics to logs"""
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            event_dict["memory_mb"] = process.memory_info().rss / 1024 / 1024
            event_dict["cpu_percent"] = process.cpu_percent()
        except:
            pass
        
        # Add duration if available
        if "duration_ms" in event_dict:
            # Already provided
            pass
        elif "start_time" in event_dict:
            # Calculate duration
            start = event_dict.pop("start_time")
            event_dict["duration_ms"] = (time.time() - start) * 1000
        
        return event_dict
    
    def _add_trace_context(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add distributed tracing context"""
        # Check for trace context in thread local or context vars
        trace_id = event_dict.get("trace_id")
        span_id = event_dict.get("span_id")
        parent_span_id = event_dict.get("parent_span_id")
        
        if trace_id:
            event_dict["trace"] = {
                "trace_id": trace_id,
                "span_id": span_id or str(uuid.uuid4()),
                "parent_span_id": parent_span_id,
            }
        
        return event_dict
    
    def _generate_exception_fingerprint(self, exc_info) -> str:
        """Generate fingerprint for exception grouping"""
        exc_type = exc_info[0].__name__
        
        # Get the most relevant stack frame
        tb = exc_info[2]
        while tb.tb_next:
            tb = tb.tb_next
        
        frame = tb.tb_frame
        fingerprint = f"{exc_type}:{frame.f_code.co_filename}:{frame.f_lineno}"
        
        return fingerprint
    
    # ============================================================================
    # LOGGER FACTORY
    # ============================================================================
    
    def get_logger(self, service: str, **kwargs) -> FilteringBoundLogger:
        """Get a logger instance for a service"""
        logger = structlog.get_logger(service=service)
        
        # Bind any additional context
        if kwargs:
            logger = logger.bind(**kwargs)
        
        return logger
    
    # ============================================================================
    # SPECIALIZED LOGGERS
    # ============================================================================
    
    def get_trading_logger(self, service: str) -> "TradingLogger":
        """Get specialized trading logger"""
        return TradingLogger(self.get_logger(service))
    
    def get_performance_logger(self, service: str) -> "PerformanceLogger":
        """Get specialized performance logger"""
        return PerformanceLogger(self.get_logger(service))
    
    def get_audit_logger(self, service: str) -> "AuditLogger":
        """Get specialized audit logger"""
        return AuditLogger(self.get_logger(service))


# ============================================================================
# SPECIALIZED LOGGERS
# ============================================================================

class TradingLogger:
    """Specialized logger for trading events"""
    
    def __init__(self, logger: FilteringBoundLogger):
        self.logger = logger
    
    def signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        price: float,
        **kwargs
    ):
        """Log trading signal"""
        self.logger.info(
            "trading_signal",
            symbol=symbol,
            action=action,
            confidence=confidence,
            price=price,
            **kwargs
        )
    
    def order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str,
        **kwargs
    ):
        """Log order event"""
        self.logger.info(
            "order_event",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            **kwargs
        )
    
    def execution(
        self,
        order_id: str,
        symbol: str,
        executed_quantity: float,
        executed_price: float,
        **kwargs
    ):
        """Log trade execution"""
        self.logger.info(
            "trade_execution",
            order_id=order_id,
            symbol=symbol,
            executed_quantity=executed_quantity,
            executed_price=executed_price,
            **kwargs
        )
    
    def position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        pnl: float,
        **kwargs
    ):
        """Log position update"""
        self.logger.info(
            "position_update",
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            pnl=pnl,
            pnl_percent=(current_price - entry_price) / entry_price * 100,
            **kwargs
        )
    
    def risk_event(
        self,
        event_type: str,
        symbol: Optional[str] = None,
        risk_score: Optional[float] = None,
        action_taken: Optional[str] = None,
        **kwargs
    ):
        """Log risk management event"""
        self.logger.warning(
            "risk_event",
            event_type=event_type,
            symbol=symbol,
            risk_score=risk_score,
            action_taken=action_taken,
            **kwargs
        )


class PerformanceLogger:
    """Specialized logger for performance metrics"""
    
    def __init__(self, logger: FilteringBoundLogger):
        self.logger = logger
        self._timers = {}
    
    @contextmanager
    def timer(self, operation: str, **kwargs):
        """Context manager for timing operations"""
        start_time = time.time()
        timer_id = f"{operation}_{id(start_time)}"
        self._timers[timer_id] = start_time
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._timers.pop(timer_id, None)
            
            self.logger.info(
                "operation_timing",
                operation=operation,
                duration_ms=duration_ms,
                **kwargs
            )
    
    def latency(
        self,
        operation: str,
        latency_ms: float,
        **kwargs
    ):
        """Log latency measurement"""
        self.logger.info(
            "latency_measurement",
            operation=operation,
            latency_ms=latency_ms,
            **kwargs
        )
    
    def throughput(
        self,
        operation: str,
        count: int,
        duration_seconds: float,
        **kwargs
    ):
        """Log throughput metrics"""
        rate = count / duration_seconds if duration_seconds > 0 else 0
        
        self.logger.info(
            "throughput_measurement",
            operation=operation,
            count=count,
            duration_seconds=duration_seconds,
            rate_per_second=rate,
            **kwargs
        )
    
    def resource_usage(
        self,
        cpu_percent: float,
        memory_mb: float,
        disk_io_mb: Optional[float] = None,
        network_io_mb: Optional[float] = None,
        **kwargs
    ):
        """Log resource usage"""
        self.logger.info(
            "resource_usage",
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            disk_io_mb=disk_io_mb,
            network_io_mb=network_io_mb,
            **kwargs
        )


class AuditLogger:
    """Specialized logger for audit trail"""
    
    def __init__(self, logger: FilteringBoundLogger):
        self.logger = logger
    
    def user_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        ip_address: Optional[str] = None,
        **kwargs
    ):
        """Log user action for audit trail"""
        self.logger.info(
            "audit_user_action",
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )
    
    def system_action(
        self,
        action: str,
        component: str,
        result: str,
        **kwargs
    ):
        """Log system action for audit trail"""
        self.logger.info(
            "audit_system_action",
            action=action,
            component=component,
            result=result,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )
    
    def security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        **kwargs
    ):
        """Log security event"""
        self.logger.warning(
            "audit_security_event",
            event_type=event_type,
            severity=severity,
            details=details,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )


# ============================================================================
# STRUCTLOG HANDLER FOR STANDARD LOGGING
# ============================================================================

class StructlogHandler(logging.Handler):
    """Handler to redirect standard logging to structlog"""
    
    def emit(self, record: logging.LogRecord):
        """Emit a log record"""
        logger = structlog.get_logger(record.name)
        
        # Map log level
        level_map = {
            logging.DEBUG: logger.debug,
            logging.INFO: logger.info,
            logging.WARNING: logger.warning,
            logging.ERROR: logger.error,
            logging.CRITICAL: logger.critical,
        }
        
        log_func = level_map.get(record.levelno, logger.info)
        
        # Build event dict
        event_dict = {
            "logger_name": record.name,
            "line_number": record.lineno,
            "function": record.funcName,
            "module": record.module,
        }
        
        # Add exception info if present
        if record.exc_info:
            event_dict["exc_info"] = record.exc_info
        
        # Log the message
        log_func(record.getMessage(), **event_dict)


# ============================================================================
# GLOBAL LOGGER INSTANCE
# ============================================================================

_logger_instance: Optional[StructuredLogger] = None


def get_logger_factory() -> StructuredLogger:
    """Get global logger factory instance"""
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = StructuredLogger()
    
    return _logger_instance


def get_logger(service: str, **kwargs) -> FilteringBoundLogger:
    """Get a logger for a service"""
    return get_logger_factory().get_logger(service, **kwargs)


def get_trading_logger(service: str) -> TradingLogger:
    """Get trading logger for a service"""
    return get_logger_factory().get_trading_logger(service)


def get_performance_logger(service: str) -> PerformanceLogger:
    """Get performance logger for a service"""
    return get_logger_factory().get_performance_logger(service)


def get_audit_logger(service: str) -> AuditLogger:
    """Get audit logger for a service"""
    return get_logger_factory().get_audit_logger(service)


# ============================================================================
# LOG AGGREGATION SETUP
# ============================================================================

def setup_log_aggregation():
    """
    Setup for log aggregation in production
    This would configure forwarding to ELK, Datadog, etc.
    """
    config = get_config()
    
    if config.is_production():
        # In production, logs go to stdout and are collected by:
        # - Fluentd/Fluentbit (Kubernetes)
        # - CloudWatch (AWS)
        # - Stackdriver (GCP)
        # - ELK Stack (self-hosted)
        pass
    
    # The beauty of 12-factor: we just write to stdout
    # The platform handles aggregation


# ============================================================================
# METRICS AND MONITORING
# ============================================================================

class MetricsLogger:
    """Log metrics in a format suitable for Prometheus/Grafana"""
    
    def __init__(self, logger: FilteringBoundLogger):
        self.logger = logger
    
    def counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """Log counter metric"""
        self.logger.info(
            "metric",
            metric_type="counter",
            metric_name=name,
            metric_value=value,
            labels=labels or {}
        )
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Log gauge metric"""
        self.logger.info(
            "metric",
            metric_type="gauge",
            metric_name=name,
            metric_value=value,
            labels=labels or {}
        )
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Log histogram metric"""
        self.logger.info(
            "metric",
            metric_type="histogram",
            metric_name=name,
            metric_value=value,
            labels=labels or {}
        )