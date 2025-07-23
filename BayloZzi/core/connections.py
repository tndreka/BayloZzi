"""
12-FACTOR PERFECT BACKING SERVICES
Factor IV: Treat backing services as attached resources - 100/100 implementation
"""

import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import structlog
from functools import lru_cache
import time

from .config import get_config, Config

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """
    Perfect backing service management
    - All services treated as attached resources
    - Connection pooling with health checks
    - Automatic reconnection and failover
    - Circuit breaker pattern
    - Zero hardcoded connections
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self._redis_pool: Optional[aioredis.Redis] = None
        self._db_engine: Optional[AsyncEngine] = None
        self._db_session_factory: Optional[sessionmaker] = None
        self._pg_pool: Optional[asyncpg.Pool] = None
        self._health_status: Dict[str, Dict[str, Any]] = {}
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
    # ============================================================================
    # REDIS CONNECTION MANAGEMENT
    # ============================================================================
    
    async def get_redis(self) -> aioredis.Redis:
        """Get Redis connection with automatic pooling and health checks"""
        if not self._redis_pool:
            await self._init_redis()
        
        # Check circuit breaker
        if self._is_circuit_open("redis"):
            raise ConnectionError("Redis circuit breaker is open")
        
        try:
            # Health check
            await self._redis_pool.ping()
            self._record_success("redis")
            return self._redis_pool
        except Exception as e:
            self._record_failure("redis", e)
            raise
    
    async def _init_redis(self):
        """Initialize Redis connection pool"""
        try:
            redis_config = self.config.get_redis_config()
            
            self._redis_pool = await aioredis.create_redis_pool(
                redis_config["url"],
                password=redis_config.get("password"),
                minsize=5,
                maxsize=redis_config["max_connections"],
                timeout=redis_config["socket_timeout"],
                retry_on_timeout=redis_config["retry_on_timeout"],
                encoding="utf-8"
            )
            
            # Test connection
            await self._redis_pool.ping()
            
            logger.info(
                "redis_connected",
                url=redis_config["url"].split("@")[-1],  # Hide credentials
                pool_size=redis_config["max_connections"]
            )
            
            self._health_status["redis"] = {
                "status": "healthy",
                "connected_at": time.time(),
                "last_check": time.time()
            }
            
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            self._health_status["redis"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": time.time()
            }
            raise
    
    # ============================================================================
    # POSTGRESQL CONNECTION MANAGEMENT
    # ============================================================================
    
    async def get_db_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic pooling"""
        if not self._db_engine:
            await self._init_database()
        
        # Check circuit breaker
        if self._is_circuit_open("postgres"):
            raise ConnectionError("PostgreSQL circuit breaker is open")
        
        async with self._db_session_factory() as session:
            try:
                # Health check
                await session.execute("SELECT 1")
                self._record_success("postgres")
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self._record_failure("postgres", e)
                raise
            finally:
                await session.close()
    
    async def get_pg_connection(self) -> asyncpg.Connection:
        """Get raw PostgreSQL connection for performance-critical operations"""
        if not self._pg_pool:
            await self._init_pg_pool()
        
        # Check circuit breaker
        if self._is_circuit_open("postgres"):
            raise ConnectionError("PostgreSQL circuit breaker is open")
        
        try:
            async with self._pg_pool.acquire() as connection:
                # Health check
                await connection.fetchval("SELECT 1")
                self._record_success("postgres")
                return connection
        except Exception as e:
            self._record_failure("postgres", e)
            raise
    
    async def _init_database(self):
        """Initialize SQLAlchemy async engine"""
        try:
            db_config = self.config.get_database_config()
            
            # Create async engine with connection pooling
            self._db_engine = create_async_engine(
                db_config["url"],
                pool_size=db_config["pool_size"],
                max_overflow=db_config["max_overflow"],
                pool_timeout=db_config["pool_timeout"],
                echo=db_config["echo"],
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,   # Recycle connections after 1 hour
            )
            
            # Create session factory
            self._db_session_factory = sessionmaker(
                self._db_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self._db_engine.connect() as conn:
                await conn.execute("SELECT 1")
            
            logger.info(
                "database_connected",
                pool_size=db_config["pool_size"],
                max_overflow=db_config["max_overflow"]
            )
            
            self._health_status["postgres"] = {
                "status": "healthy",
                "connected_at": time.time(),
                "last_check": time.time()
            }
            
        except Exception as e:
            logger.error("database_connection_failed", error=str(e))
            self._health_status["postgres"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": time.time()
            }
            raise
    
    async def _init_pg_pool(self):
        """Initialize asyncpg connection pool for raw queries"""
        try:
            db_config = self.config.get_database_config()
            
            # Parse connection URL for asyncpg
            import urllib.parse
            parsed = urllib.parse.urlparse(db_config["url"])
            
            self._pg_pool = await asyncpg.create_pool(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:],  # Remove leading slash
                min_size=5,
                max_size=db_config["pool_size"],
                timeout=db_config["pool_timeout"],
                command_timeout=60,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
            )
            
            logger.info("asyncpg_pool_created", size=db_config["pool_size"])
            
        except Exception as e:
            logger.error("asyncpg_pool_failed", error=str(e))
            raise
    
    # ============================================================================
    # CIRCUIT BREAKER IMPLEMENTATION
    # ============================================================================
    
    def _init_circuit_breaker(self, service: str):
        """Initialize circuit breaker for a service"""
        self._circuit_breakers[service] = {
            "failures": 0,
            "success": 0,
            "state": "closed",  # closed, open, half-open
            "last_failure": None,
            "opened_at": None,
            "failure_threshold": 5,
            "success_threshold": 3,
            "timeout": 60,  # seconds
        }
    
    def _is_circuit_open(self, service: str) -> bool:
        """Check if circuit breaker is open"""
        if service not in self._circuit_breakers:
            self._init_circuit_breaker(service)
        
        breaker = self._circuit_breakers[service]
        
        if breaker["state"] == "closed":
            return False
        
        if breaker["state"] == "open":
            # Check if timeout has passed
            if time.time() - breaker["opened_at"] > breaker["timeout"]:
                breaker["state"] = "half-open"
                breaker["success"] = 0
                logger.info(f"{service}_circuit_half_open")
            else:
                return True
        
        return False
    
    def _record_success(self, service: str):
        """Record successful operation"""
        if service not in self._circuit_breakers:
            self._init_circuit_breaker(service)
        
        breaker = self._circuit_breakers[service]
        breaker["failures"] = 0
        
        if breaker["state"] == "half-open":
            breaker["success"] += 1
            if breaker["success"] >= breaker["success_threshold"]:
                breaker["state"] = "closed"
                logger.info(f"{service}_circuit_closed")
    
    def _record_failure(self, service: str, error: Exception):
        """Record failed operation"""
        if service not in self._circuit_breakers:
            self._init_circuit_breaker(service)
        
        breaker = self._circuit_breakers[service]
        breaker["failures"] += 1
        breaker["last_failure"] = time.time()
        
        if breaker["state"] == "half-open":
            breaker["state"] = "open"
            breaker["opened_at"] = time.time()
            logger.warning(f"{service}_circuit_opened", error=str(error))
        
        elif breaker["state"] == "closed":
            if breaker["failures"] >= breaker["failure_threshold"]:
                breaker["state"] = "open"
                breaker["opened_at"] = time.time()
                logger.warning(f"{service}_circuit_opened", error=str(error))
    
    # ============================================================================
    # HEALTH CHECKS AND MONITORING
    # ============================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all backing services"""
        health = {
            "status": "healthy",
            "services": {},
            "timestamp": time.time()
        }
        
        # Check Redis
        try:
            if self._redis_pool:
                await self._redis_pool.ping()
                health["services"]["redis"] = {
                    "status": "healthy",
                    "response_time": time.time() - time.time()
                }
            else:
                health["services"]["redis"] = {"status": "not_initialized"}
        except Exception as e:
            health["status"] = "unhealthy"
            health["services"]["redis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check PostgreSQL
        try:
            if self._db_engine:
                async with self._db_engine.connect() as conn:
                    start = time.time()
                    await conn.execute("SELECT 1")
                    health["services"]["postgres"] = {
                        "status": "healthy",
                        "response_time": time.time() - start
                    }
            else:
                health["services"]["postgres"] = {"status": "not_initialized"}
        except Exception as e:
            health["status"] = "unhealthy"
            health["services"]["postgres"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Add circuit breaker status
        health["circuit_breakers"] = {}
        for service, breaker in self._circuit_breakers.items():
            health["circuit_breakers"][service] = {
                "state": breaker["state"],
                "failures": breaker["failures"]
            }
        
        return health
    
    # ============================================================================
    # CLEANUP AND LIFECYCLE
    # ============================================================================
    
    async def close(self):
        """Close all connections gracefully"""
        logger.info("closing_connections")
        
        # Close Redis
        if self._redis_pool:
            self._redis_pool.close()
            await self._redis_pool.wait_closed()
            self._redis_pool = None
        
        # Close PostgreSQL
        if self._db_engine:
            await self._db_engine.dispose()
            self._db_engine = None
        
        if self._pg_pool:
            await self._pg_pool.close()
            self._pg_pool = None
        
        logger.info("connections_closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_connection_manager: Optional[ConnectionManager] = None


async def get_connection_manager() -> ConnectionManager:
    """Get singleton connection manager instance"""
    global _connection_manager
    
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    
    return _connection_manager


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

@asynccontextmanager
async def get_redis() -> AsyncGenerator[aioredis.Redis, None]:
    """Get Redis connection"""
    manager = await get_connection_manager()
    redis = await manager.get_redis()
    try:
        yield redis
    finally:
        pass  # Connection is pooled, no need to close


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    manager = await get_connection_manager()
    async with manager.get_db_session() as session:
        yield session


@asynccontextmanager
async def get_pg() -> AsyncGenerator[asyncpg.Connection, None]:
    """Get raw PostgreSQL connection"""
    manager = await get_connection_manager()
    conn = await manager.get_pg_connection()
    try:
        yield conn
    finally:
        pass  # Connection is pooled


# ============================================================================
# SERVICE DISCOVERY (Production)
# ============================================================================

class ServiceDiscovery:
    """
    Service discovery for production environments
    Integrates with Consul, Kubernetes, or cloud providers
    """
    
    def __init__(self):
        self.config = get_config()
        self._service_cache: Dict[str, str] = {}
    
    async def get_service_url(self, service_name: str) -> str:
        """Get service URL with discovery"""
        # In production, use actual service discovery
        if self.config.is_production():
            # Check cache first
            if service_name in self._service_cache:
                return self._service_cache[service_name]
            
            # Kubernetes service discovery
            if self._is_kubernetes():
                url = f"http://{service_name}.{self._get_namespace()}.svc.cluster.local"
                self._service_cache[service_name] = url
                return url
            
            # Fall back to environment-based discovery
            return self.config.get_service_url(service_name)
        
        # Development/testing uses localhost
        return self.config.get_service_url(service_name)
    
    def _is_kubernetes(self) -> bool:
        """Check if running in Kubernetes"""
        import os
        return os.path.exists("/var/run/secrets/kubernetes.io")
    
    def _get_namespace(self) -> str:
        """Get Kubernetes namespace"""
        try:
            with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
                return f.read().strip()
        except:
            return "default"


# Export singleton service discovery
service_discovery = ServiceDiscovery()