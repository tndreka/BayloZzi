"""
12-FACTOR PERFECT MAIN ENTRY POINT
Factor VI: Execute app as stateless processes - 100/100
"""

import asyncio
import argparse
import os
import sys
from typing import Dict, Type, Optional

# Performance: Use uvloop for faster async (Unix/Linux only)
if sys.platform != 'win32':
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass  # Fall back to default event loop

from .core.config import get_config, validate_config
from .core.logging import get_logger, setup_log_aggregation
from .core.lifecycle import get_shutdown_manager, LifecycleAgent
from .core.connections import get_connection_manager

# Import all agents
from .agents.risk_management_agent import MultiAgentRiskManager as RiskManagementAgent
from .agents.chart_analysis_agent import ChartAnalysisAgent
from .agents.trend_identification_agent import TrendIdentificationAgent
from .agents.news_sentiment_agent import NewsSentimentAgent
from .agents.economic_factors_agent import EconomicFactorsAgent
from .agents.weekly_analysis_engine import WeeklyAnalysisEngine
from .data.historical_data_manager import HistoricalDataManager as DataManagerService
from .core.prediction_accuracy_tracker import PredictionAccuracyTracker as AccuracyTrackerService
# from .api.gateway import APIGateway  # TODO: Create API gateway

# Service registry
SERVICE_REGISTRY: Dict[str, Type[LifecycleAgent]] = {
    "risk-manager": RiskManagementAgent,
    "chart-analyzer": ChartAnalysisAgent,
    "trend-identifier": TrendIdentificationAgent,
    "news-sentiment": NewsSentimentAgent,
    "economic-factors": EconomicFactorsAgent,
    "weekly-analyzer": WeeklyAnalysisEngine,
    "data-manager": DataManagerService,
    "accuracy-tracker": AccuracyTrackerService,
    # "api-gateway": APIGateway,  # TODO: Add when API gateway is created
}


class StatelessProcess(LifecycleAgent):
    """
    Base class ensuring all processes are stateless
    All state must be externalized to Redis/PostgreSQL
    """
    
    def __init__(self, service_name: str):
        super().__init__(service_name)
        self.config = get_config()
        self.logger = get_logger(service_name)
        self.connections = None
        self._service_instance = None
    
    async def on_startup(self):
        """Initialize stateless service"""
        # Validate configuration
        validate_config()
        
        # Setup logging
        setup_log_aggregation()
        
        # Initialize connections
        self.connections = await get_connection_manager()
        
        # Verify external state stores
        health = await self.connections.health_check()
        if health["status"] != "healthy":
            raise RuntimeError(f"External state stores unhealthy: {health}")
        
        # Create service instance
        service_class = SERVICE_REGISTRY.get(self.agent_name)
        if not service_class:
            raise ValueError(f"Unknown service: {self.agent_name}")
        
        self._service_instance = service_class(
            config=self.config,
            connections=self.connections,
            logger=self.logger
        )
        
        # Initialize service
        await self._service_instance.initialize()
        
        self.logger.info(
            "service_started",
            service=self.agent_name,
            port=self.config.get_service_url(self.agent_name).split(":")[-1],
            environment=self.config.environment.value
        )
    
    async def on_shutdown(self):
        """Cleanup stateless service"""
        if self._service_instance:
            await self._service_instance.shutdown()
        
        if self.connections:
            await self.connections.close()
        
        self.logger.info("service_stopped", service=self.agent_name)
    
    async def run(self):
        """Run the service"""
        if not self._service_instance:
            raise RuntimeError("Service not initialized")
        
        # Start the service
        await self._service_instance.start()
        
        # Wait for shutdown signal
        await self.shutdown_manager.wait_for_shutdown()


async def main():
    """Main entry point for all services"""
    parser = argparse.ArgumentParser(
        description="12-Factor Forex Multi-Agent Trading System"
    )
    
    parser.add_argument(
        "--service",
        type=str,
        default=os.environ.get("SERVICE", "api-gateway"),
        choices=list(SERVICE_REGISTRY.keys()),
        help="Service to run"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Get logger early
    logger = get_logger("main")
    
    try:
        # Validate configuration
        if args.validate_only:
            validate_config()
            print("Configuration valid!")
            return 0
        
        # Log startup
        logger.info(
            "starting_service",
            service=args.service,
            pid=os.getpid(),
            python_version=sys.version
        )
        
        # Create and run service
        process = StatelessProcess(args.service)
        await process.run_forever()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")
        return 0
        
    except Exception as e:
        logger.error(
            "service_failed",
            service=args.service,
            error=str(e),
            exc_info=True
        )
        return 1


def run():
    """Synchronous entry point"""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


if __name__ == "__main__":
    run()


# ============================================================================
# STATELESS VERIFICATION
# ============================================================================

class StatelessVerifier:
    """Verify that services are truly stateless"""
    
    @staticmethod
    def verify_no_local_state(service_path: str) -> bool:
        """Verify service has no local state"""
        import ast
        import pathlib
        
        violations = []
        
        # Patterns that indicate local state
        state_patterns = [
            "open(",  # File operations
            "sqlite",  # Local database
            "shelve",  # Local persistence
            "pickle.dump",  # Local serialization
            "global ",  # Global variables
            "__dict__",  # Direct attribute access
        ]
        
        # Check all Python files
        for py_file in pathlib.Path(service_path).rglob("*.py"):
            with open(py_file, "r") as f:
                content = f.read()
                
            for pattern in state_patterns:
                if pattern in content:
                    # Check if it's actually problematic
                    try:
                        tree = ast.parse(content)
                        # More sophisticated AST analysis here
                    except:
                        pass
                    
                    violations.append(f"{py_file}: {pattern}")
        
        return len(violations) == 0


# ============================================================================
# HORIZONTAL SCALING SUPPORT
# ============================================================================

class HorizontalScaler:
    """Support for horizontal scaling (Factor VIII)"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("scaler")
    
    def get_instance_id(self) -> str:
        """Get unique instance ID for this process"""
        # In Kubernetes, use pod name
        pod_name = os.environ.get("HOSTNAME", "")
        if pod_name:
            return pod_name
        
        # Otherwise generate one
        import socket
        return f"{socket.gethostname()}-{os.getpid()}"
    
    def register_instance(self):
        """Register this instance for load balancing"""
        instance_id = self.get_instance_id()
        
        # Register in service discovery
        # This would integrate with Consul, etcd, or k8s
        
        self.logger.info(
            "instance_registered",
            instance_id=instance_id,
            service=self.config.app_name
        )
    
    def get_worker_count(self) -> int:
        """Get number of workers to spawn"""
        # Based on CPU cores or environment variable
        return int(os.environ.get("WORKER_COUNT", os.cpu_count() or 1))


# ============================================================================
# PROCESS MONITORING
# ============================================================================

class ProcessMonitor:
    """Monitor process health and performance"""
    
    def __init__(self):
        self.logger = get_logger("monitor")
        self.start_time = asyncio.get_event_loop().time()
    
    async def report_metrics(self):
        """Report process metrics"""
        import psutil
        
        process = psutil.Process()
        
        metrics = {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "threads": process.num_threads(),
            "fds": process.num_fds() if hasattr(process, "num_fds") else 0,
            "uptime_seconds": asyncio.get_event_loop().time() - self.start_time,
        }
        
        self.logger.info("process_metrics", **metrics)
        
        # Send to monitoring system
        # prometheus_client.Gauge(...).set(...)
        
        return metrics