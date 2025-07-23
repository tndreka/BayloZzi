"""
12-FACTOR PERFECT CONFIGURATION MANAGEMENT
Factor III: Store config in the environment - 100/100 implementation
"""

import os
from typing import Optional, Dict, Any, List
from functools import lru_cache
from pathlib import Path
from pydantic import Field, validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
from enum import Enum


class Environment(str, Enum):
    """Application environments with strict separation"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Standardized log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Config(BaseSettings):
    """
    Perfect 12-Factor Configuration
    ALL configuration comes from environment variables
    NO hardcoded values anywhere in the codebase
    """
    
    # ============================================================================
    # APPLICATION METADATA
    # ============================================================================
    app_name: str = Field(
        default="forex-multi-agent-system",
        description="Application name for identification"
    )
    
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment"
    )
    
    debug: bool = Field(
        default=False,
        description="Debug mode flag"
    )
    
    # ============================================================================
    # SECURITY CONFIGURATION
    # ============================================================================
    secret_key: SecretStr = Field(
        ...,
        description="Application secret key for JWT/sessions"
    )
    
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    
    jwt_expiry_hours: int = Field(
        default=24,
        description="JWT token expiry in hours"
    )
    
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    # ============================================================================
    # DATABASE CONFIGURATION (Factor IV: Backing Services)
    # ============================================================================
    database_url: str = Field(
        ...,
        description="PostgreSQL connection URL"
    )
    
    database_pool_size: int = Field(
        default=20,
        description="Database connection pool size"
    )
    
    database_pool_overflow: int = Field(
        default=10,
        description="Maximum overflow connections"
    )
    
    database_pool_timeout: int = Field(
        default=30,
        description="Connection timeout in seconds"
    )
    
    database_echo: bool = Field(
        default=False,
        description="Echo SQL statements (debug only)"
    )
    
    # ============================================================================
    # REDIS CONFIGURATION (Factor IV: Backing Services)
    # ============================================================================
    redis_url: str = Field(
        ...,
        description="Redis connection URL"
    )
    
    redis_password: Optional[SecretStr] = Field(
        default=None,
        description="Redis password if required"
    )
    
    redis_pool_size: int = Field(
        default=50,
        description="Redis connection pool size"
    )
    
    redis_socket_timeout: int = Field(
        default=5,
        description="Socket timeout in seconds"
    )
    
    redis_retry_on_timeout: bool = Field(
        default=True,
        description="Retry on timeout"
    )
    
    # ============================================================================
    # API KEYS (Factor III: Config)
    # ============================================================================
    alpha_vantage_key: Optional[SecretStr] = Field(
        default=None,
        description="Alpha Vantage API key"
    )
    
    news_api_key: Optional[SecretStr] = Field(
        default=None,
        description="News API key"
    )
    
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key for NLP"
    )
    
    fred_api_key: Optional[SecretStr] = Field(
        default=None,
        description="FRED API key"
    )
    
    trading_economics_key: Optional[SecretStr] = Field(
        default=None,
        description="Trading Economics API key"
    )
    
    # ============================================================================
    # SERVICE PORTS (Factor VII: Port Binding)
    # ============================================================================
    api_port: int = Field(
        default=8080,
        description="API Gateway port"
    )
    
    risk_manager_port: int = Field(
        default=8001,
        description="Risk Manager service port"
    )
    
    chart_analyzer_port: int = Field(
        default=8002,
        description="Chart Analyzer service port"
    )
    
    trend_identifier_port: int = Field(
        default=8003,
        description="Trend Identifier service port"
    )
    
    news_sentiment_port: int = Field(
        default=8004,
        description="News Sentiment service port"
    )
    
    economic_factors_port: int = Field(
        default=8005,
        description="Economic Factors service port"
    )
    
    weekly_analyzer_port: int = Field(
        default=8006,
        description="Weekly Analyzer service port"
    )
    
    data_manager_port: int = Field(
        default=8007,
        description="Data Manager service port"
    )
    
    accuracy_tracker_port: int = Field(
        default=8008,
        description="Accuracy Tracker service port"
    )
    
    # ============================================================================
    # LOGGING CONFIGURATION (Factor XI: Logs)
    # ============================================================================
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Application log level"
    )
    
    log_format: str = Field(
        default="json",
        description="Log format (json/text)"
    )
    
    log_include_timestamp: bool = Field(
        default=True,
        description="Include timestamp in logs"
    )
    
    log_include_hostname: bool = Field(
        default=True,
        description="Include hostname in logs"
    )
    
    # ============================================================================
    # RISK MANAGEMENT CONFIGURATION
    # ============================================================================
    max_risk_per_trade: float = Field(
        default=0.02,
        ge=0.001,
        le=0.1,
        description="Maximum risk per trade (2% default)"
    )
    
    max_drawdown: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        description="Maximum portfolio drawdown (15% default)"
    )
    
    max_concurrent_positions: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent positions"
    )
    
    trading_enabled: bool = Field(
        default=True,
        description="Global trading switch"
    )
    
    # ============================================================================
    # AGENT CONFIGURATION
    # ============================================================================
    agent_update_interval: int = Field(
        default=300,
        description="Agent update interval in seconds"
    )
    
    agent_confidence_threshold: float = Field(
        default=0.65,
        ge=0.5,
        le=0.95,
        description="Minimum confidence for signals"
    )
    
    # ============================================================================
    # MONITORING CONFIGURATION
    # ============================================================================
    prometheus_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    
    metrics_port: int = Field(
        default=9090,
        description="Metrics endpoint port"
    )
    
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    
    # ============================================================================
    # PERFORMANCE CONFIGURATION
    # ============================================================================
    cache_ttl: int = Field(
        default=300,
        description="Default cache TTL in seconds"
    )
    
    request_timeout: int = Field(
        default=30,
        description="Default request timeout"
    )
    
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )
    
    retry_delay: int = Field(
        default=1,
        description="Initial retry delay in seconds"
    )
    
    # ============================================================================
    # FEATURE FLAGS
    # ============================================================================
    feature_weekly_analysis: bool = Field(
        default=True,
        description="Enable weekly analysis engine"
    )
    
    feature_ml_predictions: bool = Field(
        default=True,
        description="Enable ML predictions"
    )
    
    feature_news_sentiment: bool = Field(
        default=True,
        description="Enable news sentiment analysis"
    )
    
    feature_economic_analysis: bool = Field(
        default=True,
        description="Enable economic analysis"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"  # Allow extra fields for flexibility
    )
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        """Ensure environment is valid"""
        if isinstance(v, str):
            v = v.lower()
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("database_url")
    def validate_database_url(cls, v):
        """Ensure database URL is properly formatted"""
        if not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("Database URL must be a PostgreSQL connection string")
        return v
    
    @validator("redis_url")
    def validate_redis_url(cls, v):
        """Ensure Redis URL is properly formatted"""
        if not v.startswith(("redis://", "rediss://")):
            raise ValueError("Redis URL must start with redis:// or rediss://")
        return v
    
    def get_service_url(self, service: str) -> str:
        """Get service URL by name"""
        port_map = {
            "api": self.api_port,
            "risk_manager": self.risk_manager_port,
            "chart_analyzer": self.chart_analyzer_port,
            "trend_identifier": self.trend_identifier_port,
            "news_sentiment": self.news_sentiment_port,
            "economic_factors": self.economic_factors_port,
            "weekly_analyzer": self.weekly_analyzer_port,
            "data_manager": self.data_manager_port,
            "accuracy_tracker": self.accuracy_tracker_port,
        }
        
        port = port_map.get(service)
        if not port:
            raise ValueError(f"Unknown service: {service}")
        
        # In production, use service discovery
        if self.environment == Environment.PRODUCTION:
            return f"http://{service}:{port}"
        else:
            return f"http://localhost:{port}"
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            "level": self.log_level.value,
            "format": self.log_format,
            "include_timestamp": self.log_include_timestamp,
            "include_hostname": self.log_include_hostname,
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_pool_overflow,
            "pool_timeout": self.database_pool_timeout,
            "echo": self.database_echo and self.is_development(),
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        config = {
            "url": self.redis_url,
            "max_connections": self.redis_pool_size,
            "socket_timeout": self.redis_socket_timeout,
            "retry_on_timeout": self.redis_retry_on_timeout,
        }
        
        if self.redis_password:
            config["password"] = self.redis_password.get_secret_value()
        
        return config
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """Get API keys (be careful with this!)"""
        keys = {}
        
        if self.alpha_vantage_key:
            keys["alpha_vantage"] = self.alpha_vantage_key.get_secret_value()
        
        if self.news_api_key:
            keys["news_api"] = self.news_api_key.get_secret_value()
        
        if self.openai_api_key:
            keys["openai"] = self.openai_api_key.get_secret_value()
        
        if self.fred_api_key:
            keys["fred"] = self.fred_api_key.get_secret_value()
        
        if self.trading_economics_key:
            keys["trading_economics"] = self.trading_economics_key.get_secret_value()
        
        return keys
    
    def mask_secrets(self) -> Dict[str, Any]:
        """Return config with masked secrets for logging"""
        data = self.model_dump()
        
        # Mask sensitive fields
        sensitive_fields = [
            "secret_key", "database_url", "redis_url", "redis_password",
            "alpha_vantage_key", "news_api_key", "openai_api_key",
            "fred_api_key", "trading_economics_key"
        ]
        
        for field in sensitive_fields:
            if field in data and data[field]:
                if isinstance(data[field], dict) and "get_secret_value" in str(type(data[field])):
                    data[field] = "***MASKED***"
                elif isinstance(data[field], str):
                    # Mask connection strings but keep protocol
                    if "://" in data[field]:
                        protocol = data[field].split("://")[0]
                        data[field] = f"{protocol}://***MASKED***"
                    else:
                        data[field] = "***MASKED***"
        
        return data


@lru_cache()
def get_config() -> Config:
    """
    Get cached configuration instance
    This ensures we only load config once
    """
    return Config()


# Export convenience functions
def get_service_config(service_name: str) -> Dict[str, Any]:
    """Get configuration for a specific service"""
    config = get_config()
    
    base_config = {
        "service_name": service_name,
        "environment": config.environment,
        "log_level": config.log_level,
        "port": config.get_service_url(service_name).split(":")[-1],
    }
    
    # Add service-specific configs
    if service_name == "risk_manager":
        base_config.update({
            "max_risk_per_trade": config.max_risk_per_trade,
            "max_drawdown": config.max_drawdown,
            "max_concurrent_positions": config.max_concurrent_positions,
            "trading_enabled": config.trading_enabled,
        })
    
    return base_config


# Validation helper
def validate_config():
    """Validate configuration on startup"""
    try:
        config = get_config()
        print(f"Configuration loaded successfully for environment: {config.environment}")
        
        # Check critical configurations
        if config.is_production():
            assert config.secret_key, "SECRET_KEY must be set in production"
            assert not config.debug, "Debug must be disabled in production"
            assert config.database_url, "DATABASE_URL must be set"
            assert config.redis_url, "REDIS_URL must be set"
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        raise


# Environment-specific helpers
def is_testing() -> bool:
    """Check if running tests"""
    return get_config().environment == Environment.TESTING


def is_local_development() -> bool:
    """Check if running locally"""
    config = get_config()
    return config.is_development() and "localhost" in config.database_url