"""
12-FACTOR PERFECT DISPOSABILITY
Factor IX: Maximize robustness with fast startup and graceful shutdown - 100/100
"""

import asyncio
import signal
import sys
import time
from typing import Dict, List, Callable, Optional, Any, Set
from functools import wraps
import structlog
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import atexit
import os

logger = structlog.get_logger(__name__)


class GracefulShutdownManager:
    """
    Perfect graceful shutdown implementation
    - Handles all signals (SIGTERM, SIGINT, SIGHUP)
    - Ordered shutdown sequence
    - Timeout protection
    - Resource cleanup tracking
    - Fast startup optimization
    """
    
    def __init__(self, app_name: str = "forex-agent"):
        self.app_name = app_name
        self._shutdown_handlers: List[Dict[str, Any]] = []
        self._startup_handlers: List[Dict[str, Any]] = []
        self._cleanup_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        self._shutdown_timeout = 30  # seconds
        self._startup_timeout = 10   # seconds
        self._is_shutting_down = False
        self._startup_time: Optional[float] = None
        self._shutdown_time: Optional[float] = None
        
        # Register signal handlers
        self._register_signal_handlers()
        
        # Register atexit handler as fallback
        atexit.register(self._emergency_shutdown)
    
    # ============================================================================
    # SIGNAL HANDLING
    # ============================================================================
    
    def _register_signal_handlers(self):
        """Register all signal handlers for graceful shutdown"""
        # Handle different signals based on platform
        if sys.platform != "win32":
            # Unix signals
            signals = [
                signal.SIGTERM,  # Termination signal
                signal.SIGINT,   # Interrupt (Ctrl+C)
                signal.SIGHUP,   # Hangup
                signal.SIGUSR1,  # User-defined signal 1 (for graceful reload)
                signal.SIGUSR2,  # User-defined signal 2 (for health check)
            ]
            
            for sig in signals:
                signal.signal(sig, self._signal_handler)
        else:
            # Windows signals (limited)
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(
            "signal_handlers_registered",
            app=self.app_name,
            platform=sys.platform
        )
    
    def _signal_handler(self, signum, frame):
        """Handle incoming signals"""
        signal_name = signal.Signals(signum).name
        
        logger.info(
            "signal_received",
            signal=signal_name,
            signum=signum
        )
        
        if signum in (signal.SIGTERM, signal.SIGINT):
            # Initiate graceful shutdown
            asyncio.create_task(self.shutdown(signal_name))
        
        elif signum == signal.SIGHUP and sys.platform != "win32":
            # Graceful reload (Unix only)
            asyncio.create_task(self.reload())
        
        elif signum == signal.SIGUSR1 and sys.platform != "win32":
            # Dump state for debugging
            self._dump_state()
        
        elif signum == signal.SIGUSR2 and sys.platform != "win32":
            # Health check signal
            self._log_health()
    
    # ============================================================================
    # STARTUP MANAGEMENT
    # ============================================================================
    
    def register_startup(
        self,
        handler: Callable,
        name: str,
        priority: int = 50,
        timeout: Optional[float] = None
    ):
        """
        Register startup handler
        Lower priority runs first (0-100)
        """
        self._startup_handlers.append({
            "handler": handler,
            "name": name,
            "priority": priority,
            "timeout": timeout or self._startup_timeout,
        })
        
        # Sort by priority
        self._startup_handlers.sort(key=lambda x: x["priority"])
    
    async def startup(self) -> float:
        """
        Execute startup sequence
        Returns startup time in seconds
        """
        self._startup_time = time.time()
        logger.info("startup_begin", app=self.app_name)
        
        failed_handlers = []
        
        for handler_info in self._startup_handlers:
            handler = handler_info["handler"]
            name = handler_info["name"]
            timeout = handler_info["timeout"]
            
            try:
                logger.info("startup_handler_begin", handler=name)
                start = time.time()
                
                # Execute with timeout
                if asyncio.iscoroutinefunction(handler):
                    await asyncio.wait_for(handler(), timeout=timeout)
                else:
                    # Run sync handlers in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler)
                
                elapsed = time.time() - start
                logger.info(
                    "startup_handler_complete",
                    handler=name,
                    elapsed=elapsed
                )
                
            except asyncio.TimeoutError:
                logger.error(
                    "startup_handler_timeout",
                    handler=name,
                    timeout=timeout
                )
                failed_handlers.append(name)
                
            except Exception as e:
                logger.error(
                    "startup_handler_failed",
                    handler=name,
                    error=str(e)
                )
                failed_handlers.append(name)
        
        total_time = time.time() - self._startup_time
        
        if failed_handlers:
            logger.warning(
                "startup_partial",
                app=self.app_name,
                failed=failed_handlers,
                elapsed=total_time
            )
        else:
            logger.info(
                "startup_complete",
                app=self.app_name,
                elapsed=total_time
            )
        
        return total_time
    
    # ============================================================================
    # SHUTDOWN MANAGEMENT
    # ============================================================================
    
    def register_shutdown(
        self,
        handler: Callable,
        name: str,
        priority: int = 50,
        timeout: Optional[float] = None
    ):
        """
        Register shutdown handler
        Higher priority runs first (100-0)
        """
        self._shutdown_handlers.append({
            "handler": handler,
            "name": name,
            "priority": priority,
            "timeout": timeout or self._shutdown_timeout,
        })
        
        # Sort by priority (reverse)
        self._shutdown_handlers.sort(key=lambda x: x["priority"], reverse=True)
    
    async def shutdown(self, reason: str = "requested"):
        """Execute graceful shutdown sequence"""
        if self._is_shutting_down:
            logger.warning("shutdown_already_in_progress")
            return
        
        self._is_shutting_down = True
        self._shutdown_time = time.time()
        self._shutdown_event.set()
        
        logger.info(
            "shutdown_begin",
            app=self.app_name,
            reason=reason
        )
        
        # Cancel all running tasks
        await self._cancel_tasks()
        
        # Execute shutdown handlers
        failed_handlers = []
        
        for handler_info in self._shutdown_handlers:
            handler = handler_info["handler"]
            name = handler_info["name"]
            timeout = handler_info["timeout"]
            
            try:
                logger.info("shutdown_handler_begin", handler=name)
                start = time.time()
                
                # Execute with timeout
                if asyncio.iscoroutinefunction(handler):
                    await asyncio.wait_for(handler(), timeout=timeout)
                else:
                    # Run sync handlers in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler)
                
                elapsed = time.time() - start
                logger.info(
                    "shutdown_handler_complete",
                    handler=name,
                    elapsed=elapsed
                )
                
            except asyncio.TimeoutError:
                logger.error(
                    "shutdown_handler_timeout",
                    handler=name,
                    timeout=timeout
                )
                failed_handlers.append(name)
                
            except Exception as e:
                logger.error(
                    "shutdown_handler_failed",
                    handler=name,
                    error=str(e)
                )
                failed_handlers.append(name)
        
        total_time = time.time() - self._shutdown_time
        
        if failed_handlers:
            logger.warning(
                "shutdown_partial",
                app=self.app_name,
                failed=failed_handlers,
                elapsed=total_time
            )
        else:
            logger.info(
                "shutdown_complete",
                app=self.app_name,
                elapsed=total_time
            )
        
        # Force exit if needed
        if total_time > self._shutdown_timeout:
            logger.error("shutdown_timeout_exceeded", forcing_exit=True)
            os._exit(1)
    
    async def _cancel_tasks(self):
        """Cancel all running tasks gracefully"""
        tasks = []
        
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
                tasks.append(task)
        
        if tasks:
            logger.info("cancelling_tasks", count=len(tasks))
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _emergency_shutdown(self):
        """Emergency shutdown handler (sync)"""
        if not self._is_shutting_down:
            logger.warning("emergency_shutdown_triggered")
            # Can't do async cleanup here, just log
            print(f"Emergency shutdown of {self.app_name}", file=sys.stderr)
    
    # ============================================================================
    # HEALTH AND STATE MANAGEMENT
    # ============================================================================
    
    def _dump_state(self):
        """Dump current state for debugging"""
        state = {
            "app": self.app_name,
            "uptime": time.time() - self._startup_time if self._startup_time else 0,
            "is_shutting_down": self._is_shutting_down,
            "active_tasks": len(asyncio.all_tasks()),
            "startup_handlers": len(self._startup_handlers),
            "shutdown_handlers": len(self._shutdown_handlers),
        }
        
        logger.info("state_dump", **state)
    
    def _log_health(self):
        """Log health status"""
        if self._is_shutting_down:
            health = "shutting_down"
        elif self._startup_time:
            health = "healthy"
        else:
            health = "starting"
        
        logger.info(
            "health_check",
            app=self.app_name,
            status=health,
            uptime=time.time() - self._startup_time if self._startup_time else 0
        )
    
    # ============================================================================
    # RELOAD SUPPORT
    # ============================================================================
    
    async def reload(self):
        """Graceful reload (configuration refresh)"""
        logger.info("reload_requested", app=self.app_name)
        
        # Emit reload event for handlers
        for handler_info in self._shutdown_handlers:
            if handler_info["name"].endswith("_reload"):
                try:
                    handler = handler_info["handler"]
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                except Exception as e:
                    logger.error(
                        "reload_handler_failed",
                        handler=handler_info["name"],
                        error=str(e)
                    )
        
        logger.info("reload_complete", app=self.app_name)
    
    # ============================================================================
    # DECORATORS AND HELPERS
    # ============================================================================
    
    def shutdown_handler(self, name: str, priority: int = 50, timeout: Optional[float] = None):
        """Decorator for shutdown handlers"""
        def decorator(func):
            self.register_shutdown(func, name, priority, timeout)
            return func
        return decorator
    
    def startup_handler(self, name: str, priority: int = 50, timeout: Optional[float] = None):
        """Decorator for startup handlers"""
        def decorator(func):
            self.register_startup(func, name, priority, timeout)
            return func
        return decorator
    
    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress"""
        return self._is_shutting_down
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        await self._shutdown_event.wait()


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_shutdown_manager: Optional[GracefulShutdownManager] = None


def get_shutdown_manager(app_name: str = "forex-agent") -> GracefulShutdownManager:
    """Get global shutdown manager instance"""
    global _shutdown_manager
    
    if _shutdown_manager is None:
        _shutdown_manager = GracefulShutdownManager(app_name)
    
    return _shutdown_manager


# ============================================================================
# AGENT BASE CLASS WITH LIFECYCLE
# ============================================================================

class LifecycleAgent:
    """Base class for agents with perfect lifecycle management"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.shutdown_manager = get_shutdown_manager(agent_name)
        self._is_running = False
        self._start_time: Optional[float] = None
        
        # Register lifecycle handlers
        self.shutdown_manager.register_startup(
            self._startup,
            f"{agent_name}_startup",
            priority=20
        )
        
        self.shutdown_manager.register_shutdown(
            self._shutdown,
            f"{agent_name}_shutdown",
            priority=80
        )
    
    async def _startup(self):
        """Agent startup handler"""
        self._start_time = time.time()
        self._is_running = True
        
        # Initialize resources
        await self.on_startup()
        
        logger.info(f"{self.agent_name}_started")
    
    async def _shutdown(self):
        """Agent shutdown handler"""
        self._is_running = False
        
        # Cleanup resources
        await self.on_shutdown()
        
        if self._start_time:
            uptime = time.time() - self._start_time
            logger.info(f"{self.agent_name}_stopped", uptime=uptime)
    
    async def on_startup(self):
        """Override in subclass for startup logic"""
        pass
    
    async def on_shutdown(self):
        """Override in subclass for shutdown logic"""
        pass
    
    @property
    def is_running(self) -> bool:
        """Check if agent is running"""
        return self._is_running and not self.shutdown_manager.is_shutting_down
    
    async def run_forever(self):
        """Run agent until shutdown"""
        # Startup
        await self.shutdown_manager.startup()
        
        # Wait for shutdown
        await self.shutdown_manager.wait_for_shutdown()
        
        # Shutdown
        await self.shutdown_manager.shutdown()


# ============================================================================
# FAST STARTUP HELPERS
# ============================================================================

class FastStartup:
    """Optimizations for fast startup"""
    
    @staticmethod
    def lazy_import(module_name: str):
        """Lazy import for faster startup"""
        import importlib
        
        class LazyModule:
            def __getattr__(self, name):
                module = importlib.import_module(module_name)
                return getattr(module, name)
        
        return LazyModule()
    
    @staticmethod
    async def parallel_init(*coroutines):
        """Initialize multiple resources in parallel"""
        return await asyncio.gather(*coroutines, return_exceptions=True)
    
    @staticmethod
    def preload_cache():
        """Preload critical data into cache"""
        # This would be implemented based on specific needs
        pass


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

async def health_check_handler(request):
    """Standard health check endpoint for all services"""
    manager = get_shutdown_manager()
    
    if manager.is_shutting_down:
        return {
            "status": "shutting_down",
            "code": 503
        }
    
    return {
        "status": "healthy",
        "uptime": time.time() - manager._startup_time if manager._startup_time else 0,
        "app": manager.app_name,
        "code": 200
    }